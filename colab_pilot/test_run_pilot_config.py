"""Tests for run_pilot.py's configurable generation budget. No live Ollama: agents
run on an injected transport. Checks the shared 4096 config, DEFAULT_CONFIG
immutability, one config across all agents, the budget in metadata, and --num-predict."""
import os, sys, json, glob, tempfile, shutil
import dataclasses as _dc

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_CONFIG            # noqa: E402
import agent_a                              # noqa: E402  (shared transport for all agents)
import run_pilot as RP                      # noqa: E402

FAILS = []
def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond: FAILS.append(name)


print("[1] pilot_config shared budget")
cfg = RP.pilot_config()
check("default num_predict == 4096", cfg.num_predict == 4096)
check("think preserved True", cfg.think is True)
check("model/temp/top_p/host/endpoint/timeout preserved",
      cfg.model == DEFAULT_CONFIG.model and cfg.temperature == DEFAULT_CONFIG.temperature
      and cfg.top_p == DEFAULT_CONFIG.top_p and cfg.host == DEFAULT_CONFIG.host
      and cfg.endpoint == DEFAULT_CONFIG.endpoint
      and cfg.request_timeout == DEFAULT_CONFIG.request_timeout)
diffs = {f.name for f in _dc.fields(cfg) if getattr(cfg, f.name) != getattr(DEFAULT_CONFIG, f.name)}
check("ONLY num_predict differs from DEFAULT_CONFIG", diffs == {"num_predict"})
check("ollama_options sends num_predict=4096", cfg.ollama_options()["num_predict"] == 4096)
check("RP default budget constant is 4096", RP.DEFAULT_NUM_PREDICT == 4096)


print("[2] DEFAULT_CONFIG immutability")
_ = RP.pilot_config(); _ = RP.pilot_config(1234)
check("DEFAULT_CONFIG.num_predict still 2048", DEFAULT_CONFIG.num_predict == 2048)
check("DEFAULT_CONFIG.think still True", DEFAULT_CONFIG.think is True)


seen = {}
def spy(prompt, config):
    seen.setdefault(id(config), {"count": 0, "num_predict": config.num_predict, "think": config.think})
    seen[id(config)]["count"] += 1
    return {"response": "SELECT 1;", "model": DEFAULT_CONFIG.model, "done_reason": "stop",
            "prompt_eval_count": 5, "eval_count": 7}
orig = agent_a._ollama_transport


def run_and_collect(out_dir, num_predict=None):
    global seen
    seen = {}
    agent_a._ollama_transport = spy
    code = None
    try:
        kw = {} if num_predict is None else {"num_predict": num_predict}
        RP.run(out_dir, timeout=10.0, skip_preflight=True, **kw)
    except SystemExit as e:
        code = e.code
    finally:
        agent_a._ollama_transport = orig
    gens = glob.glob(os.path.join(out_dir, "pilot_generations_*.jsonl"))
    metas = glob.glob(os.path.join(out_dir, "pilot_meta_*.json"))
    rows = [json.loads(l) for l in open(gens[0], encoding="utf-8")] if gens else []
    meta = json.load(open(metas[0])) if metas else {}
    return code, rows, meta


print("[3] all agents share one 4096 config (default)")
tmp = tempfile.mkdtemp(prefix="runpilotcfg_")
code, rows, meta = run_and_collect(os.path.join(tmp, "out"))
check("exit 0 on clean 36/36 run", code in (0, None))
check("exactly ONE config object used across all agent calls", len(seen) == 1)
only = next(iter(seen.values()))
check("that single config was used 36 times (12 records x 3 agents)", only["count"] == 36)
check("that config carried num_predict=4096", only["num_predict"] == 4096)
check("that config carried think=True", only["think"] is True)
check("36 rows written", len(rows) == 36)
check("every row's generation_settings.num_predict == 4096",
      all(r["generation_settings"]["num_predict"] == 4096 for r in rows))
by_agent = {}
for r in rows:
    by_agent.setdefault(r["agent"], set()).add(r["generation_settings"]["num_predict"])
check("A, A2, B each present and all at 4096",
      set(by_agent) == {"A", "A2", "B"} and all(v == {4096} for v in by_agent.values()))


print("[4] metadata records the budget")
check("meta.num_predict == 4096", meta.get("num_predict") == 4096)
check("meta.generation_settings.num_predict == 4096",
      meta.get("generation_settings", {}).get("num_predict") == 4096)
check("meta records DEFAULT_CONFIG baseline (2048) for provenance",
      meta.get("default_config_num_predict") == 2048)


print("[5] --num-predict override is honored")
tmp2 = tempfile.mkdtemp(prefix="runpilotcfg_ov_")
code2, rows2, meta2 = run_and_collect(os.path.join(tmp2, "out"), num_predict=1234)
only2 = next(iter(seen.values()))
check("override: single config used, num_predict=1234", len(seen) == 1 and only2["num_predict"] == 1234)
check("override: meta.num_predict == 1234", meta2.get("num_predict") == 1234)
check("override: all rows at 1234",
      bool(rows2) and all(r["generation_settings"]["num_predict"] == 1234 for r in rows2))
check("override did NOT mutate DEFAULT_CONFIG (still 2048)", DEFAULT_CONFIG.num_predict == 2048)

for d in (tmp, tmp2):
    shutil.rmtree(d, ignore_errors=True)

print("\nRESULT:", "ALL RUN_PILOT CONFIG TESTS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(0 if not FAILS else 1)
