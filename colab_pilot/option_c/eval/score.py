"""Candidate scoring infrastructure (Stage 2).

Scores one SQL candidate against one PREPARED GoldRecord (from golds.py), and
aggregates a list of scored candidates to the record level. No LLM logic here.

Reuses, without duplicating: safe_execute.execute_sql (read-only + timeout),
compare.compare (result policy), interpret.classify (+ labels).

Primary interpretation credit requires ALL of:
  1. execution_status == "ok",
  2. result_match is True (candidate rows == gold rows under the gold's ordering),
  3. structural label is EXACTLY A_ONLY (-> A credit) or B_ONLY with NO subtype
     (-> B credit).  B_ONLY + subtype "B_WITH_EXTRA_AGG" gets NO primary credit.
Execution correctness, result match, structural identity, and final credit are
kept as separate fields so every decision is auditable.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import compare as C            # noqa: E402
import interpret as I          # noqa: E402
import safe_execute as SE      # noqa: E402


@dataclass
class CandidateScore:
    candidate_sql: str
    execution_status: str                       # "ok" | "error" | "timeout"
    execution_error: Optional[str] = None
    execution_time: float = 0.0
    result_match: Optional[bool] = None         # None when execution did not succeed
    interpretation_label: str = I.UNCLASSIFIABLE
    interpretation_subtype: Optional[str] = None
    interpretation_reason: str = ""
    a_credit: bool = False
    b_credit: bool = False

    @property
    def any_credit(self) -> bool:
        return self.a_credit or self.b_credit


def score_candidate(candidate_sql: str, gold, timeout: float = 10.0) -> CandidateScore:
    """Score one candidate against a prepared GoldRecord. Does not reload the
    manifest or recompute gold queries."""
    # 1-2. execution (read-only, timeout)
    ex = SE.execute_sql(gold.db_abspath, candidate_sql, timeout=timeout)

    # 3. result match (only meaningful when execution succeeded)
    result_match: Optional[bool] = None
    if ex.status == "ok":
        equal, _ = C.compare(ex.rows, gold.gold_rows, gold.ordered)
        result_match = bool(equal)

    # 4. structural interpretation (independent of execution)
    res = I.classify(candidate_sql, gold.category, gold.interp_metadata, gold.schema)
    label, subtype, reason = res.label, res.subtype, res.reason

    # 5. combine -> primary credit
    exec_ok = ex.status == "ok"
    a_credit = bool(exec_ok and result_match and label == I.A_ONLY)
    b_credit = bool(exec_ok and result_match and label == I.B_ONLY and subtype is None)

    return CandidateScore(
        candidate_sql=candidate_sql,
        execution_status=ex.status,
        execution_error=ex.error,
        execution_time=ex.elapsed,
        result_match=result_match,
        interpretation_label=label,
        interpretation_subtype=subtype,
        interpretation_reason=reason,
        a_credit=a_credit,
        b_credit=b_credit,
    )


def score_candidates(candidate_sqls: List[str], gold, timeout: float = 10.0) -> List[CandidateScore]:
    return [score_candidate(sql, gold, timeout=timeout) for sql in candidate_sqls]


# ----------------------------------------------------------------- record level
@dataclass
class RecordScore:
    A_recovered: bool
    B_recovered: bool
    either_recovered: bool
    both_recovered: bool
    num_candidates: int
    a_credit_count: int
    b_credit_count: int
    distinct_sql_count: int
    label_counts: Dict[str, int] = field(default_factory=dict)
    a_candidate_indices: Tuple[int, ...] = ()
    b_candidate_indices: Tuple[int, ...] = ()


def _norm_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def score_record(scored_candidates: List[CandidateScore]) -> RecordScore:
    """Aggregate a list of CandidateScore (for ONE benchmark record) to
    record-level interpretation recovery. Agent-agnostic."""
    a_idx = tuple(i for i, c in enumerate(scored_candidates) if c.a_credit)
    b_idx = tuple(i for i, c in enumerate(scored_candidates) if c.b_credit)

    A_recovered = len(a_idx) > 0
    B_recovered = len(b_idx) > 0
    either = A_recovered or B_recovered
    # both requires TWO DISTINCT candidates, one with A credit and one with B
    # credit. a_credit and b_credit are mutually exclusive per candidate, so any
    # pair (ai, bi) is already at distinct list positions.
    both = any(ai != bi for ai in a_idx for bi in b_idx)

    labels: Dict[str, int] = {}
    for c in scored_candidates:
        labels[c.interpretation_label] = labels.get(c.interpretation_label, 0) + 1

    return RecordScore(
        A_recovered=A_recovered,
        B_recovered=B_recovered,
        either_recovered=either,
        both_recovered=both,
        num_candidates=len(scored_candidates),
        a_credit_count=len(a_idx),
        b_credit_count=len(b_idx),
        distinct_sql_count=len({_norm_sql(c.candidate_sql) for c in scored_candidates}),
        label_counts=labels,
        a_candidate_indices=a_idx,
        b_candidate_indices=b_idx,
    )
