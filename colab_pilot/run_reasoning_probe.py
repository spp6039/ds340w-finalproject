"""One-record probe (#328, A2 and B): is the budget spent on reasoning or lost to a
prompt problem?

Re-sends the agents' exact #328 prompts at num_predict=8192 with everything else
unchanged, and captures the normally-discarded `thinking` field next to `response`
so we can see where the tokens went. Each prompt is taken straight from the agent's
own generate() (via a capturing transport, no network call), so it is byte-for-byte
what the agent would send.

Diagnostic only: no changes to agents/prompts/DEFAULT_CONFIG/scoring; thinking is
never fed to the parser; done_reason=length is recorded as truncation, not a parse
error; no gold access or SQL execution. Live inference is opt-in (--live); the
default just builds and verifies the prompts and writes nothing. Outputs go to a
separate dir with reasoning_probe_* names and never overwrite existing files.

    python run_reasoning_probe.py                         # dry: verify prompts, no inference
    python run_reasoning_probe.py --live --out <dir>      # send the 2 requests (after approval)
"""
import os, sys, json, time, hashlib, argparse, datetime, platform
from dataclasses import replace

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import agent_a, agent_a2, agent_b            # noqa: E402
from config import DEFAULT_CONFIG            # noqa: E402
import ollama_health as H                    # noqa: E402

PROBE_NUM_PREDICT = 8192
RECORD_CATEGORY = "col-synonyms"
RECORD_INDEX = 328
GEN_PREFIX = "reasoning_probe_"
META_PREFIX = "reasoning_probe_meta_"

# (agent, module, parser): A2 -> list[str]; B -> list[(desc, sql)].
PROBES = [
    ("A2", agent_a2, "extract_candidates"),
    ("B", agent_b, "parse_interpretations"),
]


def probe_config(num_predict=PROBE_NUM_PREDICT):
    """DEFAULT_CONFIG with only num_predict overridden (think/temp/top_p kept)."""
    return replace(DEFAULT_CONFIG, num_predict=num_predict)


class _Captured(Exception):
    pass


def capture_agent_prompt(mod, question, schema):
    """The exact prompt mod.generate would send, captured without any network call."""
    holder = {}

    def _capturing_transport(prompt, config):
        holder["prompt"] = prompt
        raise _Captured()

    # generate() builds the prompt, calls the transport (which raises), and returns a
    # request_error output we ignore -- we only want the captured prompt.
    mod.generate(question, schema, config=probe_config(), transport=_capturing_transport)
    if "prompt" not in holder:
        raise RuntimeError(f"failed to capture prompt for {mod.__name__}")
    return holder["prompt"]


def _parse_response(agent_name, mod, parse_fn_name, response_text):
    """Run the agent's parser on the RESPONSE only. Returns (passed, sqls, descs)."""
    parsed = getattr(mod, parse_fn_name)(response_text or "")
    if agent_name == "B":                     # list[(description, sql)]
        descs = [d for d, _ in parsed]
        sqls = [s for _, s in parsed]
        return (len(parsed) > 0, sqls, descs)
    return (len(parsed) > 0, list(parsed), None)   # A2: list[str]


def do_request(agent_name, mod, parse_fn_name, question, schema, cfg, transport):
    """Build the exact prompt, send one request, return a filled result row. Captures
    thinking and response separately; classifies truncation vs parse_error; never raises."""
    prompt = capture_agent_prompt(mod, question, schema)
    row = {
        "agent": agent_name, "category": RECORD_CATEGORY, "index": RECORD_INDEX,
        "record": f"{RECORD_CATEGORY}#{RECORD_INDEX}",
        "model": cfg.model, "num_predict": cfg.num_predict,
        "generation_settings": cfg.generation_settings(),
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_char_len": len(prompt),
        "status": None, "done_reason": None, "prompt_eval_count": None,
        "eval_count": None, "latency": None,
        "thinking": None, "response": None,
        "parser_passed": None, "num_candidates": 0,
        "candidate_sqls": [], "interpretation_descriptions": None, "error": None,
    }
    t0 = time.perf_counter()
    try:
        resp = transport(prompt, cfg)
    except Exception as e:
        row["latency"] = time.perf_counter() - t0
        row["status"] = "request_error"
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    row["latency"] = time.perf_counter() - t0
    if not isinstance(resp, dict):
        row["status"] = "request_error"
        row["error"] = f"transport returned {type(resp).__name__}, expected dict"
        return row

    row["thinking"] = resp.get("thinking")        # kept separate; never parsed
    row["response"] = resp.get("response")
    row["done_reason"] = resp.get("done_reason")
    row["prompt_eval_count"] = resp.get("prompt_eval_count")
    row["eval_count"] = resp.get("eval_count")

    passed, sqls, descs = _parse_response(agent_name, mod, parse_fn_name, row["response"])
    row["parser_passed"] = passed
    row["num_candidates"] = len(sqls)
    row["candidate_sqls"] = sqls
    row["interpretation_descriptions"] = descs

    if row["done_reason"] == "length":
        row["status"] = "truncated_length"        # budget exhausted, not a parse failure
    else:
        row["status"] = "ok" if passed else "parse_error"
    return row


def _load_record():
    base = os.path.join(ROOT, "AmbiQT-master", "AmbiQT-master", "benchmark")
    val = json.load(open(os.path.join(base, RECORD_CATEGORY, "validation.json"), encoding="utf-8"))
    rec = val[RECORD_INDEX]
    if rec is None:
        raise RuntimeError(f"record {RECORD_CATEGORY}#{RECORD_INDEX} missing from validation.json")
    return rec.get("question", ""), rec["schema_without_content"]


def prepare(question, schema):
    """Build + audit the two exact prompts (no inference)."""
    info = []
    for name, mod, _ in PROBES:
        prompt = capture_agent_prompt(mod, question, schema)
        info.append({"agent": name, "prompt_char_len": len(prompt),
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "prompt": prompt})
    return info


def run(out_dir, live=False, transport=None, skip_preflight=False):
    cfg = probe_config()
    question, schema = _load_record()

    prepared = prepare(question, schema)
    print(f"record {RECORD_CATEGORY}#{RECORD_INDEX}: num_predict={cfg.num_predict} "
          f"think={cfg.think} temp={cfg.temperature} top_p={cfg.top_p}")
    for pi in prepared:
        print(f"  prompt[{pi['agent']}]: {pi['prompt_char_len']} chars  sha256={pi['prompt_sha256'][:16]}…")

    if not live and transport is None:
        print("\nDRY MODE (no inference, nothing written). Re-run with --live --out <dir> "
              "to send the 2 requests.")
        return None

    os.makedirs(out_dir, exist_ok=True)
    tx = transport if transport is not None else agent_a._ollama_transport

    if live and transport is None and not skip_preflight:
        ok, detail = H.preflight(cfg, verbose=True)
        if not ok:
            print(f"\nABORTING: Ollama preflight FAILED (stage={detail.get('stage')}). No requests sent.")
            sys.exit(2)

    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    gen_path = os.path.join(out_dir, f"{GEN_PREFIX}{run_id}.jsonl")
    meta_path = os.path.join(out_dir, f"{META_PREFIX}{run_id}.json")
    for pth in (gen_path, meta_path):
        if os.path.exists(pth):
            print(f"REFUSING to overwrite existing file: {pth}")
            sys.exit(3)

    meta = {"run_id": run_id, "probe": "reasoning-budget #328", "model": cfg.model,
            "record": f"{RECORD_CATEGORY}#{RECORD_INDEX}",
            "num_predict": cfg.num_predict,
            "default_config_num_predict": DEFAULT_CONFIG.num_predict,
            "generation_settings": cfg.generation_settings(), "host": cfg.url,
            "python": platform.python_version(), "platform": platform.platform(),
            "agents": [a for a, _, _ in PROBES],
            "prompt_sha256": {pi["agent"]: pi["prompt_sha256"] for pi in prepared},
            "note": "DIAGNOSTIC ONLY; captures the normally-discarded `thinking` field. "
                    f"DEFAULT_CONFIG.num_predict stays {DEFAULT_CONFIG.num_predict}."}
    json.dump(meta, open(meta_path, "w"), indent=2)

    n = 0
    with open(gen_path, "w", encoding="utf-8") as fh:
        for name, mod, parse_fn in PROBES:          # sequential, one request each
            row = do_request(name, mod, parse_fn, question, schema, cfg, tx)
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()   # preserve partial
            n += 1
            th = len(row["thinking"] or ""); rs = len(row["response"] or "")
            print(f"[{n}/{len(PROBES)}] {name:<2} status={row['status']:<16} "
                  f"done={row['done_reason']} eval={row['eval_count']} "
                  f"think_len={th} resp_len={rs} cands={row['num_candidates']}")
    print("\nwrote", gen_path)
    print("wrote", meta_path)
    return gen_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reasoning_probe_out",
                    help="results dir (separate from pilot/diagnostic outputs)")
    ap.add_argument("--live", action="store_true",
                    help="actually send the 2 requests to Ollama (default: dry, no inference)")
    ap.add_argument("--skip-preflight", action="store_true", help="skip server preflight")
    a = ap.parse_args()
    run(a.out, live=a.live, skip_preflight=a.skip_preflight)
