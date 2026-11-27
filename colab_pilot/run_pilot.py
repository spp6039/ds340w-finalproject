"""Run Agents A, A2, B on the 12 pilot records via Ollama and score each candidate.

Writes one JSONL row per (record, agent). The agent only ever sees the question and
the agent-facing schema; gold data is used only by the scorer, after generation.

Generation budget: num_predict is set with --num-predict (default 4096); the runner
builds one replace(DEFAULT_CONFIG, num_predict=...) copy and passes that same config
to all three agents and to preflight/metadata. DEFAULT_CONFIG is never modified.

Safeguards: a readiness/model/smoke preflight before any record (exit 2 if the
server is down), and a fail-fast that stops the run if the request_error rate goes
over 20% (exit 3), preserving whatever rows were already written.

    python run_pilot.py --out <dir> [--timeout 10] [--num-predict 4096]
                        [--error-threshold 0.20] [--skip-preflight]
"""
import os, sys, json, time, argparse, datetime, platform
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
DEFAULT_NUM_PREDICT = 4096   # validated by the 4096 budget diagnostic

should_abort = H.should_abort


def pilot_config(num_predict=DEFAULT_NUM_PREDICT):
    """One config for all three agents: DEFAULT_CONFIG with num_predict overridden."""
    return replace(DEFAULT_CONFIG, num_predict=num_predict)


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


def run(out_dir, timeout=10.0, error_threshold=0.20, skip_preflight=False,
        num_predict=DEFAULT_NUM_PREDICT):
    os.makedirs(out_dir, exist_ok=True)
    cfg = pilot_config(num_predict)          # shared by A, A2, B + preflight/meta

    if not skip_preflight:
        print(f"preflight: Ollama readiness + model + smoke (num_predict={cfg.num_predict}) ...")
        ok, detail = H.preflight(cfg, smoke_timeout=180, verbose=True)  # 180s: covers T4 model warm-up
        if not ok:
            print("\n" + "=" * 70)
            print(f"ABORTING before any record: Ollama preflight FAILED (stage={detail.get('stage')}).")
            print("No experimental records were written.")
            print("=" * 70)
            sys.exit(2)
        print("preflight OK.\n")

    print("preparing gold records (local SQLite, read-only; no model) ...")
    golds_d, rep = golds.prepare_all(timeout=timeout)
    print(f"  prepared {rep['prepared']} / eligible {rep['eligible']} (skipped {rep['skipped']})")
    val = _load_validation()
    keys = _key_map()
    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    gen_path = os.path.join(out_dir, f"pilot_generations_{run_id}.jsonl")
    meta = {"run_id": run_id, "model": cfg.model,
            "num_predict": cfg.num_predict,
            "default_config_num_predict": DEFAULT_CONFIG.num_predict,
            "generation_settings": cfg.generation_settings(),
            "host": cfg.url, "python": platform.python_version(),
            "platform": platform.platform(), "num_records": len(golds_d),
            "agents": [a for a, _ in AGENTS], "timeout": timeout,
            "error_threshold": error_threshold}
    json.dump(meta, open(os.path.join(out_dir, f"pilot_meta_{run_id}.json"), "w"), indent=2)

    total = len(golds_d) * len(AGENTS)
    n = 0            # requests written
    n_err = 0        # of which request_error
    aborted = False
    with open(gen_path, "w", encoding="utf-8") as fh:
        for (cat, idx), gold in sorted(golds_d.items()):
            if aborted:
                break
            rec = val[cat][idx]
            question = rec.get("question", "")
            schema = rec["schema_without_content"]          # agent-facing schema only
            for name, mod in AGENTS:
                out = mod.generate(question, schema, config=cfg)   # shared cfg; no gold
                scores = adapter.score_agent_candidates(out, gold, timeout=timeout)
                row = {
                    "run_id": run_id, "agent": name, "category": cat, "index": idx,
                    "db_id": gold.db_id, "perturbation_key": keys.get((cat, idx)),
                    "question": question,
                    "generation_status": out.generation_status,
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
                if out.generation_status == "request_error":
                    n_err += 1
                print(f"[{n:>2}/{total}] {cat}#{idx} {name:<2} "
                      f"{out.generation_status:<12} lat={out.latency:6.1f}s "
                      f"cand={len(out.candidates)} done={out.done_reason}")
                if out.generation_status == "request_error":
                    print(f"     request_error: {out.generation_error}")

                if should_abort(n, n_err, threshold=error_threshold):
                    print("\n" + "!" * 70)
                    print(f"FAIL-FAST: {n_err}/{n} requests returned request_error "
                          f"({100*n_err/n:.0f}% > {100*error_threshold:.0f}% threshold).")
                    print(H.diagnose(cfg))
                    print(f"Partial results preserved: {gen_path}")
                    print("!" * 70)
                    aborted = True
                    break

    print("\nwrote", gen_path)
    print("wrote", os.path.join(out_dir, f"pilot_meta_{run_id}.json"))
    if aborted:
        print(f"RUN ABORTED after {n}/{total} requests ({n_err} request_error). Exit code 3.")
        sys.exit(3)
    if n_err and should_abort(n, n_err, threshold=error_threshold):
        print(f"RUN COMPLETED BUT UNHEALTHY: {n_err}/{n} request_error. Exit code 3.")
        sys.exit(3)
    print(f"RUN OK: {n}/{total} requests written, {n_err} request_error (num_predict={cfg.num_predict}).")
    return gen_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results directory (use a mounted Drive path in Colab)")
    ap.add_argument("--timeout", type=float, default=10.0, help="per-query SQL execution timeout (s)")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT,
                    help=f"generation token budget shared by all agents (default {DEFAULT_NUM_PREDICT})")
    ap.add_argument("--error-threshold", type=float, default=0.20,
                    help="abort if request_error rate exceeds this (default 0.20)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the readiness/model/smoke preflight (NOT recommended)")
    a = ap.parse_args()
    run(a.out, timeout=a.timeout, error_threshold=a.error_threshold,
        skip_preflight=a.skip_preflight, num_predict=a.num_predict)
