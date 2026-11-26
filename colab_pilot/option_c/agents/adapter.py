"""Adapter: pass an agent's generated candidate to the Stage 2 scorer.

This is the ONLY link between generation and evaluation. The agent itself never
receives the GoldRecord; the caller holds the (already-prepared) gold and uses
this adapter to score the candidate. Generation and scoring stay separate.
"""
import os
import sys

_EVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import score as _score  # noqa: E402


def score_agent_output(agent_output, gold, timeout: float = 10.0):
    """Score an AgentOutput's single candidate against a prepared GoldRecord.

    Returns a CandidateScore, or None when the agent produced no candidate
    (failed/parse_error) -- a failed generation is NOT given a fabricated query.
    """
    sql = getattr(agent_output, "candidate_sql", None)
    if not sql:
        return None
    return _score.score_candidate(sql, gold, timeout=timeout)


def score_agent_candidates(agent_output, gold, timeout: float = 10.0):
    """Score ALL of a multi-candidate agent's candidates against a prepared
    GoldRecord. Works for any agent exposing `.candidates` (Agent A: <=1,
    Agent A2: <=2). Returns a list of CandidateScore (empty when the agent
    produced none -- failed/parse_error/truncated are never given a fabricated
    query). Reuses Stage 2's score_candidates unchanged; aggregate the result
    with score.score_record(). Candidates are scored exactly as generated, so any
    duplicates stay visible to score_record (distinct_sql_count / both_recovered).
    """
    cands = list(getattr(agent_output, "candidates", []) or [])
    if not cands:
        return []
    return _score.score_candidates(cands, gold, timeout=timeout)
