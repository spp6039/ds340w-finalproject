"""Check the bundle is self-contained and gold prep works -- no model calls.

Verifies imports, the 12 DB paths, gold preparation, gold<->validation mapping,
agent isolation (agents can only see question + schema), and scorer access. Exits
0 only if every check passes. Server health is a separate step (smoke_test.py).
"""
import os, sys, json, inspect

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

CATS = ["col-synonyms", "tbl-synonyms", "tbl-split", "tbl-agg"]
fails = []
def ok(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond: fails.append(name)

print("[1] imports")
import compare, execute, interpret, safe_execute, score, golds          # noqa: E402
import config, agent_a, agent_a2, agent_b, adapter                      # noqa: E402
ok("all required modules import", True)

print("[2] selected DB paths resolve")
man = json.load(open(os.path.join(ROOT, "option_c", "manifest_unified.json"), encoding="utf-8"))
ok(f"manifest has 12 ELIGIBLE records ({len(man)})",
   len(man) == 12 and all(m["eligibility"] == "ELIGIBLE" for m in man))
for m in man:
    ok(f"DB resolves: {m['category']}#{m['index']} {os.path.basename(m['perturbed_db_path'])}",
       os.path.isfile(os.path.join(ROOT, m["perturbed_db_path"])))

print("[3] gold preparation (no model)")
golds_d, rep = golds.prepare_all()
ok(f"prepared 12, skipped 0 (prepared={rep['prepared']} skipped={rep['skipped']})",
   rep["prepared"] == 12 and rep["skipped"] == 0)

print("[4] gold <-> validation mapping")
base = os.path.join(ROOT, "AmbiQT-master", "AmbiQT-master", "benchmark")
val = {c: json.load(open(os.path.join(base, c, "validation.json"), encoding="utf-8")) for c in CATS}
for (cat, idx), g in sorted(golds_d.items()):
    rec = val[cat][idx]
    ok(f"{cat}#{idx}: query1/query2 match validation & db_id={g.db_id}",
       rec is not None and rec["query1"] == g.query1 and rec["query2"] == g.query2
       and "schema_without_content" in rec)

print("[5] agents cannot receive gold")
for nm, mod in [("A", agent_a), ("A2", agent_a2), ("B", agent_b)]:
    params = list(inspect.signature(mod.generate).parameters)
    ok(f"agent {nm}.generate params == [question, schema, config, transport]  ({params})",
       params == ["question", "schema", "config", "transport"])

print("[6] scoring pipeline can access a prepared gold")
g = next(iter(golds_d.values()))
shim = type("O", (), {"candidates": [g.query1, g.query2]})()
cs = adapter.score_agent_candidates(shim, g)
ok("adapter+real scorer returns 2 CandidateScore for the two gold queries", len(cs) == 2)
ok("empty candidates -> no scoring", adapter.score_agent_candidates(type("E", (), {"candidates": []})(), g) == [])

print("\nRESULT:", "ALL CHECKS PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(0 if not fails else 1)
