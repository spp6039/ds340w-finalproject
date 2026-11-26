"""Diagnostic: does raising num_predict 2048 -> 4096 reduce truncations?

Runs all three agents on the same 12 records under a throwaway
replace(DEFAULT_CONFIG, num_predict=4096) copy and reports per-agent completion vs
truncation. Diagnostic only: it does not change DEFAULT_CONFIG, the agents, the
prompts, scoring, or the pilot runner, and writes budget_diag_* files to a separate
directory so it can never overwrite the real pilot outputs.

Exit codes: 0 = all 36 generated with the server available (truncations are a valid
result, not a failure); 2 = preflight failed; 3 = <36 rows or any request_error.

    python run_budget_diagnostic.py --out <dir> [--timeout 10] [--num-predict 4096]
"""
import os, sys, json, argparse, datetime, platform
from dataclasses import replace

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import agent_a, agent_a2, agent_b          # noqa: E402
import adapter, golds, score               # noqa: E402
from config import DEFAULT_CONFIG          # noqa: E402
import ollama_health as H                  # noqa: E402

CATS = ["col-synonyms", "tbl-synonyms", "tbl-split", "tbl-agg"]
AGENTS = [("A", agent_a), ("A2", agent_a2), ("B", agent_b)]
DIAG_NUM_PREDICT = 4096
# Not "pilot_" on purpose, so this can't overwrite the official pilot outputs.
GEN_PREFIX = "budget_diag_generations_"
META_PREFIX = "budget_diag_meta_"


def diag_config(num_predict=DIAG_NUM_PREDICT):
    """DEFAULT_CONFIG with only num_predict raised; think=True kept explicitly."""
    return replace(DEFAULT_CONFIG, num_predict=num_predict, think=True)


def _load_validation():
    base = os.path.join(ROOT, "AmbiQT-master", "AmbiQT-master", "benchmark")
    return {c: json.load(open(os.path.join(base, c, "validation.json"), encoding="utf-8")) for c in CATS}


def _key_map():
    man = json.load(open(os.path.join(ROOT, "option_c", "manifest_unified.json"), encoding="utf-8"))
    return {(m["category"], m["index"]): m.get("perturbation_key") for m in man}


def _score_row(scores):
    rec = score.score_record(scores) if scores else None
    return (None if rec is None else {
        "A_recovered": rec.A_recovered, "B_recovered": rec.B_recovered,
        "either_recovered": rec.either_recovered, "both_recovered": rec.both_recovered,
        "distinct_sql_count": rec.distinct_sql_count,
        "a_credit_count": rec.a_credit_count, "b_credit_count": rec.b_credit_count,
        "label_counts": rec.label_counts})


def run(out_dir, timeout=10.0, num_predict=DIAG_NUM_PREDICT, skip_preflight=False):
    cfg = diag_config(num_predict)
    os.makedirs(out_dir, exist_ok=True)

    if not skip_preflight:
        print(f"preflight: readiness + model + smoke (num_predict={cfg.num_predict}) ...")
        ok, detail = H.preflight(DEFAULT_CONFIG, verbose=True)   # same host/model
        if not ok:
            print("\n" + "=" * 70)
            print(f"ABORTING before any record: preflight FAILED (stage={detail.get('stage')}). "
                  "No records written.")
            print("=" * 70)
            sys.exit(2)
        print("preflight OK.\n")

    print("preparing gold records (local SQLite, read-only; no model) ...")
    golds_d, rep = golds.prepare_all(timeout=timeout)
    print(f"  prepared {rep['prepared']} / eligible {rep['eligible']} (skipped {rep['skipped']})")
    val = _load_validation()
    keys = _key_map()

    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    gen_path = os.path.join(out_dir, f"{GEN_PREFIX}{run_id}.jsonl")
    meta_path = os.path.join(out_dir, f"{META_PREFIX}{run_id}.json")
    for pth in (gen_path, meta_path):
        if os.path.exists(pth):
            print(f"REFUSING to overwrite existing file: {pth}")
            sys.exit(3)

    meta = {"run_id": run_id, "diagnostic": "num_predict budget sweep", "model": cfg.model,
            "baseline_num_predict": DEFAULT_CONFIG.num_predict,
            "diagnostic_num_predict": cfg.num_predict,
            "generation_settings": cfg.generation_settings(),
            "host": cfg.url, "python": platform.python_version(),
            "platform": platform.platform(), "num_records": len(golds_d),
            "agents": [a for a, _ in AGENTS], "timeout": timeout,
            "note": "DIAGNOSTIC ONLY -- does not change the official config "
                    f"(DEFAULT_CONFIG.num_predict stays {DEFAULT_CONFIG.num_predict})."}
    json.dump(meta, open(meta_path, "w"), indent=2)

    total = len(golds_d) * len(AGENTS)
    n = 0
    status_counts = {a: {"ok": 0, "truncated": 0, "parse_error": 0, "request_error": 0}
                     for a, _ in AGENTS}
    with open(gen_path, "w", encoding="utf-8") as fh:
        for (cat, idx), gold in sorted(golds_d.items()):
            rec = val[cat][idx]
            question = rec.get("question", "")
            schema = rec["schema_without_content"]          # agent-facing schema only
            for name, mod in AGENTS:
                out = mod.generate(question, schema, config=cfg)   # diagnostic config; no gold
                scores = adapter.score_agent_candidates(out, gold, timeout=timeout)
                st = out.generation_status
                if st in status_counts[name]:
                    status_counts[name][st] += 1
                row = {
                    "run_id": run_id, "diagnostic_num_predict": cfg.num_predict,
                    "agent": name, "category": cat, "index": idx,
                    "db_id": gold.db_id, "perturbation_key": keys.get((cat, idx)),
                    "question": question,
                    "generation_status": st,
                    "generation_error": out.generation_error,
                    "done_reason": out.done_reason, "model": out.model,
                    "generation_settings": out.generation_settings,
                    "prompt_tokens": out.prompt_tokens,
                    "total_generated_tokens": out.total_generated_tokens,
                    "latency": out.latency,
                    "num_generation_calls": getattr(out, "num_generation_calls", 1),
                    "num_candidates": len(out.candidates),
                    "distinct_candidate_count": getattr(out, "distinct_candidate_count", None),
                    "has_duplicates": getattr(out, "has_duplicates", None),
                    "candidate_sqls": list(out.candidates),
                    "interpretation_descriptions": getattr(out, "interpretation_descriptions", None),
                    "raw_response": out.raw_response,
                    "record_score": _score_row(scores),
                    "candidate_scores": [
                        {"sql": cs.candidate_sql, "execution_status": cs.execution_status,
                         "execution_error": cs.execution_error, "execution_time": cs.execution_time,
                         "result_match": cs.result_match, "label": cs.interpretation_label,
                         "subtype": cs.interpretation_subtype,
                         "a_credit": cs.a_credit, "b_credit": cs.b_credit} for cs in scores],
                }
                fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
                n += 1
                print(f"[{n:>2}/{total}] {cat}#{idx} {name:<2} {st:<12} "
                      f"lat={out.latency:6.1f}s cand={len(out.candidates)} "
                      f"done={out.done_reason} tok={out.total_generated_tokens}")

    print("\n" + "=" * 70)
    print(f"BUDGET DIAGNOSTIC  num_predict={cfg.num_predict}  (baseline "
          f"{DEFAULT_CONFIG.num_predict})   run_id={run_id}")
    print("-" * 70)
    print(f"{'agent':6} {'ok':>4} {'truncated':>10} {'parse_err':>10} {'req_err':>8}  of {len(golds_d)}")
    req_err_total = 0
    for name, _ in AGENTS:
        c = status_counts[name]
        req_err_total += c["request_error"]
        print(f"{name:6} {c['ok']:>4} {c['truncated']:>10} {c['parse_error']:>10} {c['request_error']:>8}")
    print("=" * 70)
    print("wrote", gen_path)
    print("wrote", meta_path)

    if n != total:
        print(f"INCOMPLETE: wrote {n}/{total} rows. Exit 3.")
        sys.exit(3)
    if req_err_total > 0:
        print(f"SERVER/MODEL UNAVAILABLE DURING RUN: {req_err_total} request_error "
              f"across {total} attempts. Exit 3.")
        sys.exit(3)
    print(f"DIAGNOSTIC OK: {n}/{total} attempts generated (0 request_error). "
          "Truncations above are a valid diagnostic signal, not a failure.")
    return gen_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="budget_diag_out",
                    help="results dir (SEPARATE from pilot outputs; use a Drive path in Colab)")
    ap.add_argument("--timeout", type=float, default=10.0, help="per-query SQL execution timeout (s)")
    ap.add_argument("--num-predict", type=int, default=DIAG_NUM_PREDICT,
                    help="diagnostic generation budget (default 4096)")
    ap.add_argument("--skip-preflight", action="store_true", help="skip server preflight (NOT recommended)")
    a = ap.parse_args()
    run(a.out, timeout=a.timeout, num_predict=a.num_predict, skip_preflight=a.skip_preflight)
