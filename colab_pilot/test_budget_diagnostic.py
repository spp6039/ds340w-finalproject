"""Tests for run_budget_diagnostic.py. No live Ollama: agents run on an injected
transport. Checks the 4096 config, DEFAULT_CONFIG immutability, one shared config
across all agents, truncation labeling, and the non-overwrite guard."""
import os, sys, json, glob, tempfile, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DEFAULT_CONFIG            # noqa: E402
import agent_a, agent_a2, agent_b            # noqa: E402
import run_budget_diagnostic as D            # noqa: E402

FAILS = []
def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond: FAILS.append(name)


print("[1] diagnostic config (dataclasses.replace)")
cfg = D.diag_config()
check("num_predict == 4096", cfg.num_predict == 4096)
check("think preserved True", cfg.think is True)
check("model unchanged", cfg.model == DEFAULT_CONFIG.model)
check("temperature unchanged", cfg.temperature == DEFAULT_CONFIG.temperature)
check("top_p unchanged", cfg.top_p == DEFAULT_CONFIG.top_p)
check("host/endpoint/url unchanged", cfg.host == DEFAULT_CONFIG.host
      and cfg.endpoint == DEFAULT_CONFIG.endpoint and cfg.url == DEFAULT_CONFIG.url)
check("ollama_options sends num_predict=4096", cfg.ollama_options()["num_predict"] == 4096)
import dataclasses as _dc
diffs = {f.name for f in _dc.fields(cfg) if getattr(cfg, f.name) != getattr(DEFAULT_CONFIG, f.name)}
check("ONLY num_predict differs from DEFAULT_CONFIG", diffs == {"num_predict"})


print("[2] DEFAULT_CONFIG immutability")
_ = D.diag_config(); _ = D.diag_config(4096)
check("DEFAULT_CONFIG.num_predict still 2048", DEFAULT_CONFIG.num_predict == 2048)
check("DEFAULT_CONFIG.think still True", DEFAULT_CONFIG.think is True)


print("[3] agents receive identical diagnostic config")
seen = {}
def spy_transport(prompt, config):
    seen.setdefault(id(config), 0)
    seen[id(config)] += 1
    return {"response": "SELECT 1;", "model": DEFAULT_CONFIG.model, "done_reason": "stop",
            "prompt_eval_count": 5, "eval_count": 7}

# A2/B default to agent_a._ollama_transport, so patching that one covers all three.
orig = agent_a._ollama_transport
agent_a._ollama_transport = spy_transport

tmp = tempfile.mkdtemp(prefix="budgetdiag_")
code = None
try:
    D.run(os.path.join(tmp, "out"), timeout=10.0, skip_preflight=True)
except SystemExit as e:
    code = e.code
finally:
    agent_a._ollama_transport = orig

check("exit 0 on clean 36/36 run (no request_error)", code in (0, None))
check("exactly ONE config object used for all 36 agent calls", len(seen) == 1)
check("that config was used 36 times", sum(seen.values()) == 36)

out1 = os.path.join(tmp, "out")
gens = glob.glob(os.path.join(out1, D.GEN_PREFIX + "*.jsonl"))
metas = glob.glob(os.path.join(out1, D.META_PREFIX + "*.json"))
rows = [json.loads(l) for l in open(gens[0], encoding="utf-8")] if gens else []
check("wrote exactly one budget_diag generations file", len(gens) == 1)
check("wrote exactly one budget_diag meta file", len(metas) == 1)
check("36 rows written (no record discarded)", len(rows) == 36)
check("every row stamps num_predict=4096 in generation_settings",
      all(r["generation_settings"]["num_predict"] == 4096 for r in rows))
check("filenames are budget_diag_* (NOT pilot_*)",
      all(os.path.basename(g).startswith("budget_diag_") for g in gens + metas))
if metas:
    m = json.load(open(metas[0]))
    check("meta records baseline=2048 and diagnostic=4096",
          m["baseline_num_predict"] == 2048 and m["diagnostic_num_predict"] == 4096)


print("[4] truncation labeling preserved under diagnostic config")
def trunc_transport(prompt, config):
    return {"response": "", "model": DEFAULT_CONFIG.model, "done_reason": "length",
            "prompt_eval_count": 5, "eval_count": config.num_predict}
agent_a._ollama_transport = trunc_transport
tmp2 = tempfile.mkdtemp(prefix="budgetdiag_tr_")
code2 = None
try:
    D.run(os.path.join(tmp2, "out"), timeout=10.0, skip_preflight=True)
except SystemExit as e:
    code2 = e.code
finally:
    agent_a._ollama_transport = orig
gens2 = glob.glob(os.path.join(tmp2, "out", D.GEN_PREFIX + "*.jsonl"))
rows2 = [json.loads(l) for l in open(gens2[0], encoding="utf-8")] if gens2 else []
check("all 36 rows present on truncation (nothing discarded)", len(rows2) == 36)
check("all truncated rows labeled generation_status='truncated'",
      bool(rows2) and all(r["generation_status"] == "truncated" for r in rows2))
check("all truncated rows carry done_reason='length'",
      bool(rows2) and all(r["done_reason"] == "length" for r in rows2))
check("truncation does NOT cause nonzero exit (valid diagnostic outcome)", code2 in (0, None))


print("[5] non-overwrite guard")
tmp3 = tempfile.mkdtemp(prefix="budgetdiag_ow_")
outdir = os.path.join(tmp3, "out"); os.makedirs(outdir)
pilot_file = os.path.join(outdir, "pilot_generations_EXISTING.jsonl")
with open(pilot_file, "w") as f: f.write('{"keep":"me"}\n')
before = open(pilot_file).read()
agent_a._ollama_transport = spy_transport
try:
    D.run(outdir, timeout=10.0, skip_preflight=True)
except SystemExit:
    pass
finally:
    agent_a._ollama_transport = orig
check("pre-existing pilot_ output untouched (byte-identical)", open(pilot_file).read() == before)
check("diagnostic wrote only budget_diag_* files alongside it",
      all(os.path.basename(f).startswith(("budget_diag_", "pilot_")) for f in os.listdir(outdir)))

for d in (tmp, tmp2, tmp3):
    shutil.rmtree(d, ignore_errors=True)

print("\nRESULT:", "ALL BUDGET-DIAGNOSTIC TESTS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(0 if not FAILS else 1)
