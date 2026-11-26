"""Tests for run_reasoning_probe.py. No live Ollama: requests use injected mocks.
Checks thinking/response captured separately, done_reason=length vs parse_error,
the real A2/B parsers, partial-result preservation, the 8192 config, and that the
exact agent prompts are reused."""
import os, sys, json, glob, tempfile, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_CONFIG            # noqa: E402
import agent_a2, agent_b                     # noqa: E402
import run_reasoning_probe as P              # noqa: E402

FAILS = []
def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond: FAILS.append(name)

SCHEMA = "singer :\n  singer_id\n  artist_name\n  age"   # newline -> format_schema skipped
Q = "Show singer names."
CFG = P.probe_config()


print("[5] probe_config 8192 / DEFAULT_CONFIG immutability")
check("probe num_predict == 8192", CFG.num_predict == 8192)
check("think/temp/top_p/model preserved",
      CFG.think is True and CFG.temperature == 0.0 and CFG.top_p == 1.0
      and CFG.model == DEFAULT_CONFIG.model)
import dataclasses as _dc
diffs = {f.name for f in _dc.fields(CFG) if getattr(CFG, f.name) != getattr(DEFAULT_CONFIG, f.name)}
check("ONLY num_predict differs from DEFAULT_CONFIG", diffs == {"num_predict"})
_ = P.probe_config()
check("DEFAULT_CONFIG.num_predict still 2048 after probe_config()", DEFAULT_CONFIG.num_predict == 2048)


print("[6] exact agent prompts reused")
a2_cap = P.capture_agent_prompt(agent_a2, Q, SCHEMA)
b_cap = P.capture_agent_prompt(agent_b, Q, SCHEMA)
# newline schema -> generate() leaves it unchanged, so build_prompt(Q, SCHEMA) must match
check("A2 captured prompt == agent_a2.build_prompt(Q, SCHEMA)", a2_cap == agent_a2.build_prompt(Q, SCHEMA))
check("B captured prompt == agent_b.build_prompt(Q, SCHEMA)", b_cap == agent_b.build_prompt(Q, SCHEMA))
check("A2 prompt carries the question and the A2 template tail 'SQL:'",
      Q in a2_cap and a2_cap.rstrip().endswith("SQL:"))
check("B prompt carries the question and the B template tail 'JSON:'",
      Q in b_cap and b_cap.rstrip().endswith("JSON:"))
check("A2 and B prompts differ (distinct instructions)", a2_cap != b_cap)
check("capture is deterministic (same string twice)",
      P.capture_agent_prompt(agent_b, Q, SCHEMA) == b_cap)


print("[1] thinking and response captured separately")
def tx_think(prompt, cfg):
    return {"thinking": "let me reason about singers ...", "response": "SELECT artist_name FROM singer",
            "done_reason": "stop", "prompt_eval_count": 40, "eval_count": 120}
row = P.do_request("A2", agent_a2, "extract_candidates", Q, SCHEMA, CFG, tx_think)
check("thinking stored separately and non-empty", row["thinking"] == "let me reason about singers ...")
check("response stored separately", row["response"] == "SELECT artist_name FROM singer")
check("thinking != response (distinct fields)", row["thinking"] != row["response"])
check("prompt_sha256 + prompt recorded", len(row["prompt_sha256"]) == 64 and row["prompt"] == a2_cap)
check("tokens captured", row["prompt_eval_count"] == 40 and row["eval_count"] == 120)


print("[2] done_reason=length classified as truncation, not parse_error")
def tx_length(prompt, cfg):
    return {"thinking": "x" * 500, "response": "", "done_reason": "length",
            "prompt_eval_count": 199, "eval_count": cfg.num_predict}
row = P.do_request("B", agent_b, "parse_interpretations", Q, SCHEMA, CFG, tx_length)
check("status == truncated_length (budget exhausted)", row["status"] == "truncated_length")
check("done_reason length recorded", row["done_reason"] == "length")
check("eval_count == budget (8192)", row["eval_count"] == 8192)
check("NOT labeled parse_error despite 0 candidates", row["status"] != "parse_error")
check("num_candidates == 0 on empty response", row["num_candidates"] == 0)
check("thinking preserved even when response empty", len(row["thinking"]) == 500)


print("[3] real parsers accept valid A2 / B responses")
a2_resp = "SELECT artist_name FROM singer\n---\nSELECT artist_name, age FROM singer"
row = P.do_request("A2", agent_a2, "extract_candidates", Q, SCHEMA, CFG,
                   lambda p, c: {"thinking": "t", "response": a2_resp, "done_reason": "stop",
                                 "prompt_eval_count": 40, "eval_count": 90})
check("A2 valid response -> status ok", row["status"] == "ok")
check("A2 two candidates parsed", row["num_candidates"] == 2 and row["parser_passed"])
check("A2 has no interpretation_descriptions (not applicable)", row["interpretation_descriptions"] is None)

b_resp = ('{"interpretations": [{"description": "stage name", "sql": "SELECT artist_name FROM singer"},'
          ' {"description": "performer name", "sql": "SELECT performer_name FROM singer"}]}')
row = P.do_request("B", agent_b, "parse_interpretations", Q, SCHEMA, CFG,
                   lambda p, c: {"thinking": "t", "response": b_resp, "done_reason": "stop",
                                 "prompt_eval_count": 60, "eval_count": 140})
check("B valid response -> status ok", row["status"] == "ok")
check("B two candidates parsed", row["num_candidates"] == 2 and row["parser_passed"])
check("B interpretation_descriptions captured", row["interpretation_descriptions"] == ["stage name", "performer name"])
check("B candidate_sqls captured",
      row["candidate_sqls"] == ["SELECT artist_name FROM singer", "SELECT performer_name FROM singer"])


print("[4] partial results preserved when second request fails")
_calls = {"n": 0}
def tx_first_ok_second_raises(prompt, cfg):
    _calls["n"] += 1
    if _calls["n"] >= 2:
        raise ConnectionError("simulated drop on second request")
    return {"thinking": "t", "response": "SELECT artist_name FROM singer",
            "done_reason": "stop", "prompt_eval_count": 40, "eval_count": 90}

tmp = tempfile.mkdtemp(prefix="reasonprobe_")
outdir = os.path.join(tmp, "out")
code = None
try:
    P.run(outdir, live=True, transport=tx_first_ok_second_raises)
except SystemExit as e:
    code = e.code
gens = glob.glob(os.path.join(outdir, P.GEN_PREFIX + "*.jsonl"))
rows = [json.loads(l) for l in open(gens[0], encoding="utf-8")] if gens else []
check("both rows written (first preserved, second recorded)", len(rows) == 2)
if len(rows) == 2:
    check("first row (A2) succeeded and is preserved", rows[0]["agent"] == "A2" and rows[0]["status"] == "ok")
    check("second row (B) recorded as request_error, not lost", rows[1]["agent"] == "B"
          and rows[1]["status"] == "request_error" and "simulated drop" in rows[1]["error"])
metas = glob.glob(os.path.join(outdir, P.META_PREFIX + "*.json"))
check("meta written with num_predict=8192", bool(metas) and json.load(open(metas[0]))["num_predict"] == 8192)
check("output files are reasoning_probe_* (not pilot_/budget_diag_)",
      all(os.path.basename(f).startswith("reasoning_probe_") for f in gens + metas))

shutil.rmtree(tmp, ignore_errors=True)

print("\nRESULT:", "ALL REASONING-PROBE TESTS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(0 if not FAILS else 1)
