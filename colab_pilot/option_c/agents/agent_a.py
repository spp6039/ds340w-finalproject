"""Agent A - single-candidate Text-to-SQL baseline.

Receives only a question and the agent-facing schema; produces exactly ONE SQL
candidate via local Ollama (qwen3:4b). No ambiguity detection, no multi-candidate
generation, no access to gold queries/results/metadata/evaluator/DB.

Public API:
  format_schema(schema_without_content) -> str      # agent-facing schema text
  build_prompt(question, schema_text) -> str
  generate(question, schema, config=DEFAULT_CONFIG, transport=None) -> AgentOutput

`transport` lets tests inject a mock Ollama; default uses a stdlib urllib call
(no extra dependency). A failed generation yields candidate_sql=None (never a
fabricated fallback).
"""
import os
import re
import sys
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from config import DEFAULT_CONFIG  # noqa: E402

AGENT_NAME = "A"


@dataclass
class AgentOutput:
    agent_name: str
    candidate_sql: Optional[str]
    raw_response: Optional[str]
    model: str
    generation_settings: dict
    # "ok" | "parse_error" | "truncated" | "request_error"
    generation_status: str
    latency: float
    prompt_tokens: Optional[int] = None
    # Ollama eval_count: TOTAL generated tokens, INCLUDING discarded `thinking`
    # tokens. Not an answer/SQL-only count; Ollama does not report one.
    total_generated_tokens: Optional[int] = None
    # Ollama stop reason: "stop" (natural) | "length" (hit num_predict budget) |
    # None when the transport did not report it.
    done_reason: Optional[str] = None
    generation_error: Optional[str] = None

    @property
    def candidates(self):
        """At most one candidate; empty on any failure incl. truncation (never
        fabricated)."""
        return [self.candidate_sql] if self.candidate_sql else []


# ----------------------------------------------------------------- schema / prompt
def format_schema(schema_without_content: str) -> str:
    """Turn AmbiQT's `schema_without_content` ("t : a , b | u : c , d") into a
    compact per-table listing. This is the AGENT-FACING schema and already hides
    retained original objects (synonyms replace them in AmbiQT's presentation)."""
    lines = []
    for part in schema_without_content.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        tname, cols = part.split(":", 1)
        cols = [c.strip() for c in cols.split(",") if c.strip()]
        lines.append(f"{tname.strip()}({', '.join(cols)})")
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are an expert data analyst who writes SQLite SQL.
Given a database schema and a question, write ONE SQL query that answers it.

Rules:
- Output exactly one SQLite SELECT query.
- Use only the tables and columns that appear in the schema below.
- Do not explain, comment, or output anything except the SQL query.

Schema:
{schema}

Question: {question}

SQL:"""


def build_prompt(question: str, schema_text: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema_text, question=question)


# ----------------------------------------------------------------- SQL extraction
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)
_STMT = re.compile(r"\b(with|select)\b", re.I)
_FENCE_LEAD = re.compile(r"^```(?:sql)?\s*", re.I)
_FENCE_TRAIL = re.compile(r"\s*```$")


def _first_statement(region: str) -> "tuple[str, bool]":
    """Return (first_statement_text, terminated) for `region`, which begins at a
    SELECT/WITH. Scans quote-aware so a ';' inside a '...' or "..." string literal
    (incl. SQL '' / "" escapes) does NOT end the statement. `terminated` is True
    iff an unquoted ';' was found (a confident right boundary)."""
    quote = None
    i, n = 0, len(region)
    while i < n:
        ch = region[i]
        if quote is not None:
            if ch == quote:
                if i + 1 < n and region[i + 1] == quote:   # doubled = escaped quote
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == ";":
            return region[:i], True
        i += 1
    return region, False


def _is_word(c: str) -> bool:
    return c.isalnum() or c == "_"


def _trim_trailing_prose(stmt: str) -> str:
    """For an UNTERMINATED statement (no closing ';'), cut at the first unquoted
    sentence-style period so trailing explanatory prose ("... desc . This lists
    singers.") does not leak into the SQL. A '.' inside a string literal, between
    word chars (qualified name a.b), or between digits (decimal 1.5) is kept."""
    quote = None
    i, n = 0, len(stmt)
    while i < n:
        ch = stmt[i]
        if quote is not None:
            if ch == quote:
                if i + 1 < n and stmt[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == ".":
            prev = stmt[i - 1] if i > 0 else ""
            nxt = stmt[i + 1] if i + 1 < n else ""
            if _is_word(prev) and _is_word(nxt):
                i += 1          # qualified name  a.b
                continue
            if prev.isdigit() and nxt.isdigit():
                i += 1          # decimal literal 1.5
                continue
            return stmt[:i]     # sentence boundary -> drop prose from here
        i += 1
    return stmt


def extract_sql(text: Optional[str]) -> Optional[str]:
    """Extract exactly ONE complete SQL statement from a model response, or None
    if one cannot be confidently identified. Never invents or repairs SQL.

    Strategy (lightweight, no parser dependency):
      1. strip any <think>...</think> reasoning trace;
      2. prefer a fenced ```code``` block (fences bound out surrounding prose);
      3. start at the first SELECT/WITH;
      4. end at the first *unquoted* ';' (quote-aware -> literals with ';' survive;
         a second statement after the ';' is dropped);
      5. if there was no ';' terminator, trim trailing sentence-style prose.
    """
    if not text:
        return None
    body = _THINK.sub(" ", text)
    m = _FENCE.search(body)                     # prefer fenced code
    if m:
        body = m.group(1)
    s = _STMT.search(body)                      # first SELECT/WITH
    if not s:
        return None
    region = body[s.start():]
    stmt, terminated = _first_statement(region)
    if not terminated:                          # may carry trailing prose
        stmt = _trim_trailing_prose(stmt)
    stmt = _FENCE_TRAIL.sub("", _FENCE_LEAD.sub("", stmt.strip())).strip()
    return stmt or None


# ----------------------------------------------------------------- Ollama call
def _ollama_transport(prompt: str, config) -> dict:
    """Default transport: POST to Ollama /api/generate (stdlib urllib)."""
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "think": config.think,
        "options": config.ollama_options(),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(config.url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=config.request_timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------------------------------------------- generate
def generate(question: str, schema, config=DEFAULT_CONFIG, transport=None) -> AgentOutput:
    """Generate exactly one SQL candidate. `schema` is the agent-facing schema
    string (AmbiQT schema_without_content) or an already-formatted schema text."""
    schema_text = format_schema(schema) if ("|" in schema or ":" in schema) and "\n" not in schema else schema
    prompt = build_prompt(question, schema_text)
    settings = config.generation_settings()
    t0 = time.perf_counter()
    try:
        resp = (transport or _ollama_transport)(prompt, config)
    except Exception as e:
        return AgentOutput(agent_name=AGENT_NAME, candidate_sql=None, raw_response=None,
                           model=config.model, generation_settings=settings,
                           generation_status="request_error", latency=time.perf_counter() - t0,
                           done_reason=None,
                           generation_error=f"{type(e).__name__}: {e}")
    latency = time.perf_counter() - t0
    if not isinstance(resp, dict):
        resp = {"response": str(resp)}
    raw = resp.get("response", "") or ""
    done_reason = resp.get("done_reason")
    prompt_tokens = resp.get("prompt_eval_count")
    total_tokens = resp.get("eval_count")       # includes discarded thinking tokens

    # Truncation guard: an explicit length-cap means the generation was cut off at
    # num_predict. Even if extract_sql could pull a syntactically plausible partial
    # statement out of `raw`, it is NOT a valid candidate -> status "truncated",
    # zero candidates, raw preserved for debugging. A MISSING done_reason is handled
    # conservatively: we only flag truncation on an EXPLICIT "length" signal and do
    # not reject otherwise-valid output on an unprovable truncation.
    if done_reason == "length":
        return AgentOutput(
            agent_name=AGENT_NAME, candidate_sql=None, raw_response=raw,
            model=config.model, generation_settings=settings,
            generation_status="truncated", latency=latency,
            prompt_tokens=prompt_tokens, total_generated_tokens=total_tokens,
            done_reason=done_reason,
            generation_error="generation truncated by token budget (done_reason=length)",
        )

    sql = extract_sql(raw)
    status = "ok" if sql else "parse_error"
    return AgentOutput(
        agent_name=AGENT_NAME, candidate_sql=sql, raw_response=raw,
        model=config.model, generation_settings=settings,
        generation_status=status, latency=latency,
        prompt_tokens=prompt_tokens, total_generated_tokens=total_tokens,
        done_reason=done_reason,
        generation_error=None if sql else "no SQL statement found in response",
    )
