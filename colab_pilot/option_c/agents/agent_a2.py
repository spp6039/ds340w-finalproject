"""Agent A2 - multi-candidate Text-to-SQL baseline.

Generates UP TO TWO SQL candidates for a question WITHOUT any ambiguity
detection, interpretation enumeration, or explanation. A2 is the control for
Agent B: it isolates the effect of "being allowed more than one candidate" from
B's "explicit ambiguity reasoning".

It REUSES Agent A's shared infrastructure and never re-implements it more weakly:
  * config.DEFAULT_CONFIG            same model / temperature / top_p / think /
                                     num_predict / request_timeout (no A2-specific
                                     decoding advantage, no larger budget)
  * agent_a._ollama_transport        same stdlib-urllib Ollama call
  * agent_a.extract_sql + helpers    same hardened, quote-aware SQL extraction
  * done_reason truncation guard     truncated output -> zero valid candidates
  * token accounting                 total_generated_tokens = eval_count (incl.
                                     discarded thinking tokens)

Generation strategy: ONE Ollama request returning up to two queries separated by
a line containing only `---`. One request keeps call / latency / compute / token-
BUDGET parity with Agent A and with a one-call Agent B, so the A2-vs-B comparison
isolates reasoning rather than resources; temperature=0.0 determinism also makes
two independent identical-prompt requests collapse to duplicates. The cost is
higher truncation risk (both candidates share one num_predict budget); that is
detected and recorded, never silently promoted. See the Stage 3B report for the
full trade-off and the shared budget policy.

A2 deliberately does NOT ask the model to identify interpretations, enumerate
meanings, explain ambiguity, or label candidates -- those are Agent B's job.

Public API:
  build_prompt(question, schema_text) -> str
  extract_candidates(text, maximum=2) -> list[str]
  generate(question, schema, config=DEFAULT_CONFIG, transport=None) -> AgentOutput
"""
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import agent_a as A                      # noqa: E402  (reuse, not re-implement)
from config import DEFAULT_CONFIG        # noqa: E402

AGENT_NAME = "A2"
MAX_CANDIDATES = 2


@dataclass
class AgentOutput:
    agent_name: str
    candidate_sqls: List[str]            # 0..MAX_CANDIDATES, in generation order
    raw_response: Optional[str]
    model: str
    generation_settings: dict
    # "ok" | "parse_error" | "truncated" | "request_error"
    generation_status: str
    latency: float
    num_generation_calls: int = 0        # actual Ollama calls made (audit/budget)
    prompt_tokens: Optional[int] = None
    # Ollama eval_count: TOTAL generated tokens, INCLUDING discarded thinking.
    total_generated_tokens: Optional[int] = None
    done_reason: Optional[str] = None
    generation_error: Optional[str] = None

    @property
    def candidates(self) -> List[str]:
        """Parsed candidates (same attribute name Agent A exposes), so the shared
        scoring adapter works for A and A2 alike. Empty on any failure."""
        return list(self.candidate_sqls)

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_sqls)

    @property
    def distinct_candidate_count(self) -> int:
        return len({_norm(c) for c in self.candidate_sqls})

    @property
    def has_duplicates(self) -> bool:
        return self.num_candidates != self.distinct_candidate_count


# ----------------------------------------------------------------- prompt
# NOTE: no "interpretation", "meaning", "ambiguous", or "distinct reading" wording
# -- A2 is a blind multi-candidate baseline, not an ambiguity-aware agent.
PROMPT_TEMPLATE = """You are an expert data analyst who writes SQLite SQL.
Given a database schema and a question, write up to TWO candidate SQL queries that answer it.

Rules:
- Output up to two SQLite SELECT queries (two if you reasonably can, otherwise one).
- Each query must use only the tables and columns that appear in the schema below.
- Separate the two queries with a line containing only ---
- Do not explain, comment, number, or output anything except the SQL queries and the separator.

Schema:
{schema}

Question: {question}

SQL:"""


def build_prompt(question: str, schema_text: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema_text, question=question)


# ----------------------------------------------------------------- extraction
_SEP = re.compile(r"(?m)^[ \t]*(?:-{3,}|={3,})[ \t]*$")   # a '---' / '===' line


def _norm(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _split_statements(text: str, maximum: int) -> List[str]:
    """Split a single blob into up to `maximum` SQL-statement substrings using
    Agent A's quote-aware statement scanner (so ';' inside a literal is safe and a
    second statement after an unquoted ';' is separated). Fuzzy only when there is
    no terminator -- then we confidently return just the first region."""
    s = A._STMT.search(text)
    if not s:
        return [text]                     # no SELECT/WITH -> extract_sql filters it
    region = text[s.start():]
    out: List[str] = []
    while region and len(out) < maximum:
        stmt, terminated = A._first_statement(region)
        out.append(stmt)
        if not terminated:
            break
        region = region[len(stmt) + 1:]
        nxt = A._STMT.search(region)
        if not nxt:
            break
        region = region[nxt.start():]
    return out


def extract_candidates(text: Optional[str], maximum: int = MAX_CANDIDATES) -> List[str]:
    """Extract up to `maximum` SQL candidates from a model response, reusing Agent
    A's hardened single-statement extractor on each block. Returns [] if none.

    Block detection (in order of preference):
      1. explicit `---` / `===` separator lines (the instructed format);
      2. multiple fenced ```code``` blocks;
      3. sequential statements in one blob (quote-aware ';' split).
    Each block is then parsed by A.extract_sql (never a weaker parser).
    """
    if not text:
        return []
    t = A._THINK.sub(" ", text)           # drop reasoning trace once, up front

    blocks = [b for b in _SEP.split(t) if b.strip()]
    if len(blocks) < 2:
        fences = [f for f in A._FENCE.findall(t) if f.strip()]
        if len(fences) >= 2:
            blocks = fences
        else:
            blocks = _split_statements(t, maximum)

    candidates: List[str] = []
    for b in blocks:
        if len(candidates) >= maximum:
            break
        sql = A.extract_sql(b)            # hardened, quote-aware, no fabrication
        if sql:
            candidates.append(sql)
    return candidates


# ----------------------------------------------------------------- generate
def generate(question: str, schema, config=DEFAULT_CONFIG, transport=None) -> AgentOutput:
    """Generate up to two SQL candidates with ONE Ollama request. `schema` is the
    agent-facing schema string (AmbiQT schema_without_content) or pre-formatted."""
    schema_text = A.format_schema(schema) if ("|" in schema or ":" in schema) and "\n" not in schema else schema
    prompt = build_prompt(question, schema_text)
    settings = config.generation_settings()
    t0 = time.perf_counter()
    try:
        resp = (transport or A._ollama_transport)(prompt, config)
    except Exception as e:
        return AgentOutput(
            agent_name=AGENT_NAME, candidate_sqls=[], raw_response=None,
            model=config.model, generation_settings=settings,
            generation_status="request_error", latency=time.perf_counter() - t0,
            num_generation_calls=1, done_reason=None,
            generation_error=f"{type(e).__name__}: {e}")
    latency = time.perf_counter() - t0
    if not isinstance(resp, dict):
        resp = {"response": str(resp)}
    raw = resp.get("response", "") or ""
    done_reason = resp.get("done_reason")
    prompt_tokens = resp.get("prompt_eval_count")
    total_tokens = resp.get("eval_count")        # incl. discarded thinking tokens

    # Truncation guard (same policy as Agent A): an explicit length-cap means the
    # generation was cut off -- a partial second (or first) query is NOT a valid
    # candidate. Missing done_reason is handled conservatively (not flagged).
    if done_reason == "length":
        return AgentOutput(
            agent_name=AGENT_NAME, candidate_sqls=[], raw_response=raw,
            model=config.model, generation_settings=settings,
            generation_status="truncated", latency=latency,
            num_generation_calls=1, prompt_tokens=prompt_tokens,
            total_generated_tokens=total_tokens, done_reason=done_reason,
            generation_error="generation truncated by token budget (done_reason=length)")

    candidates = extract_candidates(raw, MAX_CANDIDATES)
    status = "ok" if candidates else "parse_error"
    return AgentOutput(
        agent_name=AGENT_NAME, candidate_sqls=candidates, raw_response=raw,
        model=config.model, generation_settings=settings,
        generation_status=status, latency=latency,
        num_generation_calls=1, prompt_tokens=prompt_tokens,
        total_generated_tokens=total_tokens, done_reason=done_reason,
        generation_error=None if candidates else "no SQL statement found in response")
