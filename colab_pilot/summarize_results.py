"""Summarize pilot run files into the metrics reported in the paper.

No model and no network are needed: this only reads pilot_generations_*.jsonl files
written by run_pilot.py (per-candidate scores are already stored in each row).

Usage
-----
    # Paper tables from the committed result files
    python summarize_results.py ../results/pilot_generations_20261119T191703Z.jsonl \
                                ../results/pilot_generations_20261124T194002Z.jsonl

    # Check a fresh re-run against the committed 8,192-token paper run
    python summarize_results.py <new_run>.jsonl \
        --reference ../results/pilot_generations_20261124T194002Z.jsonl

Metric definitions (match the paper's Evaluation Metrics section)
-----------------------------------------------------------------
completed        generation_status == "ok" and at least one SQL candidate
truncated        done_reason == "length" or generation_status == "truncated"
any-alt coverage questions where >= 1 gold alternative was recovered (either_recovered)
both-alt cov.    questions where BOTH gold alternatives were recovered (both_recovered)
alt-interp recall  (#gold alternatives recovered) / (2 x #questions), each counted once
exec success     executed-OK candidates / all generated candidates
recall per 1k tok  gold alternatives recovered per 1,000 generated tokens (incl. thinking)

"Recovered" uses the scorer's primary credit (execution OK AND exact result match AND
structural label A_ONLY / B_ONLY), not result matching alone.
"""
import argparse
import json
import os
import sys

AGENTS = ("A", "A2", "B")


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta_path = os.path.join(os.path.dirname(path),
                             os.path.basename(path).replace("pilot_generations_", "pilot_meta_")
                             .replace(".jsonl", ".json"))
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return rows, meta


def agent_stats(rows, agent):
    g = [r for r in rows if r.get("agent") == agent]
    rs = [(r.get("record_score") or {}) for r in g]
    cands = [c for r in g for c in r.get("candidate_scores", [])]
    n = len(g)
    recovered = sum(bool(x.get("A_recovered")) + bool(x.get("B_recovered")) for x in rs)
    tokens = sum(r.get("total_generated_tokens") or 0 for r in g)
    return {
        "agent": agent,
        "questions": n,
        "completed": sum(r.get("generation_status") == "ok" and r.get("num_candidates", 0) > 0 for r in g),
        "truncated": sum(r.get("done_reason") == "length" or r.get("generation_status") == "truncated" for r in g),
        "parse_err": sum(r.get("generation_status") == "parse_error" for r in g),
        "candidates": len(cands),
        "exec_ok": sum(c.get("execution_status") == "ok" for c in cands),
        "any_alt": sum(bool(x.get("either_recovered")) for x in rs),
        "both_alt": sum(bool(x.get("both_recovered")) for x in rs),
        "alts_recovered": recovered,
        "recall": recovered / (2 * n) if n else 0.0,
        "tokens": tokens,
        "per_1k_tok": 1000 * recovered / tokens if tokens else 0.0,
        "latency_s": sum(r.get("latency") or 0 for r in g),
    }


def print_table(rows, meta, label):
    print("=" * 96)
    print(f"{label}   run_id={meta.get('run_id', '?')}   num_predict={meta.get('num_predict', '?')}   "
          f"model={meta.get('model', '?')}   rows={len(rows)}")
    print("=" * 96)
    hdr = (f"{'Agent':<6}{'Done':>6}{'Trunc':>7}{'Parse':>7}{'ExecOK/Cand':>13}"
           f"{'Any-alt':>9}{'Both-alt':>10}{'Recall':>14}{'Tokens':>9}{'Rec/1k':>8}{'Time':>8}")
    print(hdr)
    print("-" * len(hdr))
    for a in AGENTS:
        s = agent_stats(rows, a)
        n = s["questions"]
        print(f"{a:<6}{s['completed']:>3}/{n:<2}{s['truncated']:>5}/{n:<1}{s['parse_err']:>6} "
              f"{s['exec_ok']:>6}/{s['candidates']:<5}"
              f"{s['any_alt']:>5}/{n:<3}{s['both_alt']:>6}/{n:<3}"
              f"{s['alts_recovered']:>5}/{2*n} {100*s['recall']:>4.0f}%"
              f"{s['tokens']:>9}{s['per_1k_tok']:>8.2f}{s['latency_s']/60:>6.1f}m")
    print()


def per_question(rows):
    out = {}
    for r in rows:
        rs = r.get("record_score") or {}
        out[(r["agent"], r["category"], r["index"])] = (
            r.get("generation_status"),
            bool(rs.get("A_recovered")), bool(rs.get("B_recovered")),
            tuple(r.get("candidate_sqls") or ()),
        )
    return out


def compare(new_rows, ref_rows):
    new, ref = per_question(new_rows), per_question(ref_rows)
    keys = sorted(set(new) | set(ref))
    same_score = same_sql = 0
    diffs = []
    for k in keys:
        a, b = new.get(k), ref.get(k)
        if a and b and a[:3] == b[:3]:
            same_score += 1
        else:
            diffs.append((k, a[:3] if a else None, b[:3] if b else None))
        if a and b and a[3] == b[3]:
            same_sql += 1
    print("=" * 96)
    print("Re-run vs committed paper run (per agent x question)")
    print("=" * 96)
    print(f"identical status + recovered interpretations: {same_score}/{len(keys)}")
    print(f"identical SQL candidates (exact text):         {same_sql}/{len(keys)}")
    if diffs:
        print("\nDifferences (status, A_recovered, B_recovered)  new  vs  paper:")
        for (ag, cat, idx), a, b in diffs:
            print(f"  {ag:<2} {cat}#{idx:<5} {a}  vs  {b}")
    else:
        print("\nAll per-question outcomes match the paper run.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="pilot_generations_*.jsonl files")
    ap.add_argument("--reference", help="committed run to compare the (first) run against")
    a = ap.parse_args()

    loaded = []
    for p in a.runs:
        rows, meta = load(p)
        loaded.append((rows, meta))
        print_table(rows, meta, os.path.basename(p))

    if a.reference:
        ref_rows, ref_meta = load(a.reference)
        print_table(ref_rows, ref_meta, "REFERENCE: " + os.path.basename(a.reference))
        compare(loaded[0][0], ref_rows)


if __name__ == "__main__":
    sys.exit(main())
