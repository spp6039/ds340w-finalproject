"""Gold-result preparation for the AmbiQT evaluator (Stage 1).

For each ELIGIBLE manifest record this:
  1. joins the manifest entry to its validation.json record by (category, index),
  2. resolves the durable perturbed DB path (project-root-relative),
  3. extracts the DB schema (safe_execute.schema_from_db) for interpret.py,
  4. executes query1 and query2 read-only, with a timeout,
  5. compares their results with the existing policy (compare.compare), and
  6. establishes the reusable gold result + ordering policy, preserving the
     validated invariant query1_rows == query2_rows on the perturbed DB.

Records whose preparation fails are SKIPPED and reported explicitly (never
silently dropped). The returned GoldRecord objects are meant to be prepared
ONCE and reused by the later scoring pipeline (gold rows are not recomputed
per candidate).

Reuses: compare.has_order_by / compare.compare, interpret.metadata_from_record,
safe_execute.execute_sql / schema_from_db.
"""
import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import compare as C                       # noqa: E402
import interpret as I                     # noqa: E402
import safe_execute as SE                 # noqa: E402

CATS = ["col-synonyms", "tbl-synonyms", "tbl-split", "tbl-agg"]


@dataclass
class GoldRecord:
    category: str
    index: int
    db_id: str
    perturbed_db_path: str                 # project-relative (as in the manifest)
    db_abspath: str
    question: str
    query1: str
    query2: str
    ordered: bool                          # ORDER BY present in the gold -> row order matters
    gold_rows: List[Tuple]                 # the single gold result (query1 == query2)
    gold_rowcount: int
    interp_metadata: dict                  # for interpret.classify(...)
    schema: dict                           # {table: {col: type}} for interpret.classify(...)
    status: str = "ready"                  # "ready" | "skipped"
    skip_reason: Optional[str] = None
    q1_status: str = "ok"
    q2_status: str = "ok"


def _skipped(cat, idx, db_id, path, reason, q1s="", q2s=""):
    return GoldRecord(category=cat, index=idx, db_id=db_id, perturbed_db_path=path,
                      db_abspath="", question="", query1="", query2="", ordered=False,
                      gold_rows=[], gold_rowcount=0, interp_metadata={}, schema={},
                      status="skipped", skip_reason=reason, q1_status=q1s, q2_status=q2s)


def _load_validation():
    return {c: json.load(open(os.path.join(PROJECT, "AmbiQT-master", "AmbiQT-master",
                                            "benchmark", c, "validation.json"), encoding="utf-8"))
            for c in CATS}


def prepare_one(m, rec, timeout=10.0, schema_cache=None):
    """Prepare a single gold record. `m` = manifest entry, `rec` = validation record."""
    cat, idx, db_id = m["category"], m["index"], m["db_id"]
    rel = m["perturbed_db_path"]
    if not rel:
        return _skipped(cat, idx, db_id, rel, "no perturbed_db_path in manifest")
    abspath = os.path.join(PROJECT, rel)
    if not os.path.isfile(abspath):
        return _skipped(cat, idx, db_id, rel, f"perturbed DB not found: {abspath}")

    # schema (cached per db file)
    try:
        if schema_cache is not None and abspath in schema_cache:
            schema = schema_cache[abspath]
        else:
            schema = SE.schema_from_db(abspath)
            if schema_cache is not None:
                schema_cache[abspath] = schema
    except Exception as e:
        return _skipped(cat, idx, db_id, rel, f"schema extraction failed: {type(e).__name__}: {e}")

    q1, q2 = rec["query1"], rec["query2"]
    r1 = SE.execute_sql(abspath, q1, timeout=timeout)
    r2 = SE.execute_sql(abspath, q2, timeout=timeout)
    if not r1.ok:
        return _skipped(cat, idx, db_id, rel, f"query1 {r1.status}: {r1.error}", r1.status, r2.status)
    if not r2.ok:
        return _skipped(cat, idx, db_id, rel, f"query2 {r2.status}: {r2.error}", r1.status, r2.status)

    ordered = C.has_order_by(q1)
    equal, _ = C.compare(r1.rows, r2.rows, ordered)
    if not equal:
        return _skipped(cat, idx, db_id, rel,
                        f"query1/query2 results differ (invariant violated): "
                        f"n1={r1.rowcount} n2={r2.rowcount}", r1.status, r2.status)

    try:
        meta = I.metadata_from_record(cat, rec)
    except Exception as e:
        return _skipped(cat, idx, db_id, rel, f"metadata build failed: {type(e).__name__}: {e}",
                        r1.status, r2.status)

    return GoldRecord(category=cat, index=idx, db_id=db_id, perturbed_db_path=rel,
                      db_abspath=abspath, question=rec.get("question", ""),
                      query1=q1, query2=q2, ordered=ordered,
                      gold_rows=r1.rows, gold_rowcount=r1.rowcount,
                      interp_metadata=meta, schema=schema,
                      status="ready", q1_status=r1.status, q2_status=r2.status)


def prepare_all(limit=None, timeout=10.0, categories=None):
    """Prepare all ELIGIBLE golds. Returns (golds_dict, report).
    golds_dict maps (category, index) -> GoldRecord (only status=='ready')."""
    manifest = json.load(open(os.path.join(PROJECT, "option_c", "manifest_unified.json"), encoding="utf-8"))
    val = _load_validation()
    cats = set(categories or CATS)
    elig = [m for m in manifest if m["eligibility"] == "ELIGIBLE" and m["category"] in cats]
    if limit:
        elig = elig[:limit]

    schema_cache: Dict[str, dict] = {}
    golds: Dict[Tuple[str, int], GoldRecord] = {}
    skipped: List[GoldRecord] = []
    t0 = time.perf_counter()
    for m in elig:
        rec = val[m["category"]][m["index"]]
        gr = prepare_one(m, rec, timeout=timeout, schema_cache=schema_cache)
        if gr.status == "ready":
            golds[(gr.category, gr.index)] = gr
        else:
            skipped.append(gr)

    report = _build_report(elig, golds, skipped, round(time.perf_counter() - t0, 2))
    return golds, report


def _build_report(elig, golds, skipped, runtime):
    from collections import Counter, defaultdict
    per = defaultdict(lambda: {"eligible": 0, "prepared": 0})
    for m in elig:
        per[m["category"]]["eligible"] += 1
    for (cat, idx) in golds:
        per[cat]["prepared"] += 1
    reasons = Counter()
    q1_fail = q2_fail = mismatch = timeout_ct = path_fail = schema_fail = 0
    for s in skipped:
        r = s.skip_reason or ""
        if "not found" in r or "no perturbed_db_path" in r:
            path_fail += 1; reasons["path"] += 1
        elif "schema extraction" in r:
            schema_fail += 1; reasons["schema"] += 1
        elif r.startswith("query1"):
            q1_fail += 1; reasons["query1"] += 1
            if "timeout" in r:
                timeout_ct += 1
        elif r.startswith("query2"):
            q2_fail += 1; reasons["query2"] += 1
            if "timeout" in r:
                timeout_ct += 1
        elif "results differ" in r:
            mismatch += 1; reasons["mismatch"] += 1
        else:
            reasons["other"] += 1
    return {
        "eligible": len(elig), "prepared": len(golds), "skipped": len(skipped),
        "query1_failures": q1_fail, "query2_failures": q2_fail,
        "result_mismatches": mismatch, "timeouts": timeout_ct,
        "path_failures": path_fail, "schema_failures": schema_fail,
        "per_category": {c: dict(v) for c, v in sorted(per.items())},
        "skip_examples": [(s.category, s.index, s.skip_reason) for s in skipped[:25]],
        "runtime_sec": runtime,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--categories", nargs="+", default=None)
    args = ap.parse_args()

    golds, rep = prepare_all(limit=args.limit, timeout=args.timeout, categories=args.categories)
    print("GOLD PREPARATION REPORT")
    print(f"  eligible loaded          : {rep['eligible']}")
    print(f"  successfully prepared     : {rep['prepared']}")
    print(f"  skipped                   : {rep['skipped']}")
    print(f"    query1 exec failures    : {rep['query1_failures']}")
    print(f"    query2 exec failures    : {rep['query2_failures']}")
    print(f"    result mismatches       : {rep['result_mismatches']}")
    print(f"    timeouts                : {rep['timeouts']}")
    print(f"    path-resolution fails   : {rep['path_failures']}")
    print(f"    schema-extraction fails : {rep['schema_failures']}")
    print("  per-category (eligible -> prepared):")
    for c, v in rep["per_category"].items():
        print(f"    {c:14} {v['eligible']:>5} -> {v['prepared']:>5}")
    if rep["skip_examples"]:
        print("  skip examples:")
        for ex in rep["skip_examples"]:
            print(f"    {ex}")
    print(f"  runtime: {rep['runtime_sec']}s")
    ok = (rep["prepared"] == 2993 and rep["skipped"] == 0) or args.limit or args.categories
    print("\nRESULT:", "ALL 2,993 PREPARED" if (rep["prepared"] == 2993 and rep["skipped"] == 0)
          else f"{rep['prepared']} prepared, {rep['skipped']} skipped")
    return golds, rep


if __name__ == "__main__":
    main()
