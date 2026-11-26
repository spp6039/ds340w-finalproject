"""Agent B - explicit ambiguity-reasoning Text-to-SQL agent.

Agent B explicitly assesses whether a question has more than one genuinely
plausible reading given the schema, identifies AT MOST two distinct readings
(one when only one is reasonable -- it does not invent ambiguity), and emits one
SQL candidate per reading. It is the "with explicit ambiguity reasoning" arm; the
A2-vs-B comparison isolates that reasoning because B gets NO extra generation
resources (one call, one shared budget) and NO evaluation information.

Reuses the shared infrastructure, nothing re-implemented more weakly:
  * config.DEFAULT_CONFIG            same model / temperature / top_p / think /
                                     num_predict / request_timeout (no B-specific
                                     decoding setting, no larger budget, no retry)
  * agent_a._ollama_transport        same stdlib-urllib Ollama call (one request)
  * agent_a.extract_sql + helpers    same hardened, quote-aware SQL extraction
                                     (no silent repair of malformed SQL)
  * done_reason truncation guard     truncated output -> zero candidates
  * token accounting                 total_generated_tokens = eval_count (incl.
                                     discarded thinking tokens; CoT is NOT stored)

Output format: ONE JSON object
    {"interpretations": [{"description": "...", "sql": "SELECT ..."}, ...]}
with one or two pairs. Each kept pair must have a nonempty description AND a SQL
value that the shared extractor accepts; malformed pairs are dropped, and because
the pairs are independent JSON array elements they can be isolated unambiguously.
If the whole response does not yield at least one valid pair, Agent B returns zero
candidates with an explicit parse error -- it never fabricates a pair.

Interpretation descriptions are retained ONLY for later qualitative auditing of
B's reasoning. They never award evaluation credit: the adapter passes SQL alone
to the unchanged Stage 2 scorer, which decides A/B credit from gold execution +
the structural classifier. (Two readings may yield identical SQL, and two SQL
queries may not reflect distinct readings -- so reasoning and SQL credit are kept
separate.)

Public API:
  build_prompt(question, schema_text) -> str
  parse_interpretations(text, maximum=2) -> list[(description, sql)]
  generate(question, schema, config=DEFAULT_CONFIG, transport=None) -> AgentOutput
"""
import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import agent_a as A                      # noqa: E402  (reuse, not re-implement)
from config import DEFAULT_CONFIG        # noqa: E402

AGENT_NAME = "B"
MAX_CANDIDATES = 2

# Accept the spec'd keys plus a couple of common aliases, so a minor key choice
# does not discard an otherwise-valid pair. Kept small and explicit.
_DESC_KEYS = ("description", "interpretation", "reading")
_SQL_KEYS = ("sql", "query")


@dataclass
class AgentOutput:
    agent_name: str
    candidate_sqls: List[str]                   # 0..MAX_CANDIDATES, in order
    interpretation_descriptions: List[str]      # aligned 1:1 with candidate_sqls
    raw_response: Optional[str]
    model: str
    generation_settings: dict
    # "ok" | "parse_error" | "truncated" | "request_error"
    generation_status: str
    latency: float
    num_generation_calls: int = 0
    prompt_tokens: Optional[int] = None
    total_generated_tokens: Optional[int] = None   # eval_count, incl. thinking
    done_reason: Optional[str] = None
    generation_error: Optional[str] = None

    @property
    def candidates(self) -> List[str]:
        """SQL candidates only (same attribute name A/A2 expose) so the shared,
        agent-agnostic adapter works unchanged. Descriptions are NOT included --
        they never reach the scorer."""
        return list(self.candidate_sqls)

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_sqls)

    @property
    def interpretations(self) -> List[dict]:
        """Audit view: description<->sql pairs, for qualitative review only."""
        return [{"description": d, "sql": s}
                for d, s in zip(self.interpretation_descriptions, self.candidate_sqls)]

    @property
    def distinct_candidate_count(self) -> int:
        return len({_norm(c) for c in self.candidate_sqls})

    @property
    def has_duplicates(self) -> bool:
        """True when distinct readings produced identical SQL (reasoning != SQL).
        Named to match Agent A2's `has_duplicates` so the experiment runner can
        read one attribute across agents."""
        return self.num_candidates != self.distinct_candidate_count

    @property
    def has_duplicate_sql(self) -> bool:
        """Backward-compatible alias of `has_duplicates`."""
        return self.has_duplicates


# ----------------------------------------------------------------- prompt
PROMPT_TEMPLATE = """You are an expert data analyst who writes SQLite SQL.

Step 1 - Interpretation assessment:
Decide whether the question below has more than one genuinely plausible meaning given the schema. A genuine ambiguity changes which columns, tables, or rows are returned (or the SQL operation) -- not mere wording or formatting. Identify at most TWO distinct plausible readings. If only one reasonable reading exists, identify only one. Do not invent a distinction that does not materially change the requested information.

Step 2 - SQL generation:
For each reading you identified, write exactly ONE SQLite SELECT query that answers it, using only the tables and columns in the schema below. Do not use tables or columns that are not in the schema. Do not put explanations inside the SQL.

Output ONLY a single JSON object, and nothing before or after it, of exactly this shape:
{{"interpretations": [{{"description": "<one concise sentence naming the reading>", "sql": "<one SQLite SELECT query>"}}]}}
Include ONE object if there is a single reading, or TWO objects if there are two distinct readings. Never more than two.

Schema:
{schema}

Question: {question}

JSON:"""


def build_prompt(question: str, schema_text: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema_text, question=question)


# ----------------------------------------------------------------- JSON parsing
def _norm(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _iter_json_objects(text: str):
    """Yield each balanced top-level {...} substring in order, string/escape aware
    (braces and quotes inside JSON string literals do not affect nesting). A stray
    or truncated object does not stop the scan: a complete object is yielded and
    the scan resumes after it; an unbalanced trailing '{' ends the scan. Objects
    are yielded, never merged."""
    i = 0
    n = len(text)
    while True:
        start = text.find("{", i)
        if start < 0:
            return
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(start, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:                     # unbalanced from here on -> no more objects
            return
        yield text[start:end + 1]
        i = end + 1


def _iter_json_arrays(text: str):
    """Yield each balanced top-level [...] substring in order, string/escape aware
    (brackets and quotes inside JSON string literals do not affect nesting). Mirrors
    _iter_json_objects for the case where the model emits a BARE top-level array of
    interpretation pairs instead of the {"interpretations": [...]} wrapper. A stray
    or truncated array does not stop the scan; an unbalanced trailing '[' ends it."""
    i = 0
    n = len(text)
    while True:
        start = text.find("[", i)
        if start < 0:
            return
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(start, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:                     # unbalanced from here on -> no more arrays
            return
        yield text[start:end + 1]
        i = end + 1


def _pick(d: dict, keys) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def parse_interpretations(text: Optional[str], maximum: int = MAX_CANDIDATES
                          ) -> List[Tuple[str, str]]:
    """Parse the model's JSON into up to `maximum` (description, sql) pairs.

    Conservative: a pair is kept only if it has a nonempty string description AND
    a `sql` value that the shared hardened extractor turns into a real statement.
    Malformed pairs are dropped (independent array elements -> safe isolation); if
    nothing parses, returns []. Never repairs SQL, never fabricates a pair.
    """
    if not text:
        return []
    t = A._THINK.sub(" ", text)                 # discard reasoning trace, never stored
    # Scan balanced top-level objects in order (fenced content first, then the
    # whole text) and select the FIRST that is a dict with an `interpretations`
    # list. This skips a brace-containing preamble (invalid or a valid-but-
    # unrelated object) instead of giving up on it. Objects are never merged, and
    # a dict lacking the required structure is not accepted.
    items = None
    sources = []
    m = A._FENCE.search(t)
    if m:
        sources.append(m.group(1))
    sources.append(t)
    for src in sources:
        for js in _iter_json_objects(src):
            try:
                obj = json.loads(js)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("interpretations"), list):
                items = obj["interpretations"]
                break
        if items is not None:
            break
    # Fallback: the model may emit a BARE top-level JSON array of pair objects
    # ([{...}, {...}]) instead of the {"interpretations": [...]} wrapper (observed
    # with qwen3:4b). Accept the first balanced top-level array that parses to a
    # list containing at least one object. The strict per-pair validation below
    # still applies, so an array that is not interpretation pairs yields zero pairs
    # (-> parse_error), never a silent or ambiguous acceptance. Runs ONLY when the
    # wrapper search above found nothing, so wrapper outputs are unaffected.
    if items is None:
        for src in sources:
            for js in _iter_json_arrays(src):
                try:
                    arr = json.loads(js)
                except (ValueError, TypeError):
                    continue
                if isinstance(arr, list) and any(isinstance(e, dict) for e in arr):
                    items = arr
                    break
            if items is not None:
                break
    if items is None:
        return []

    pairs: List[Tuple[str, str]] = []
    for it in items:
        if len(pairs) >= maximum:
            break
        if not isinstance(it, dict):
            continue
        desc = _pick(it, _DESC_KEYS)
        sqlv = _pick(it, _SQL_KEYS)
        if desc is None or sqlv is None:        # missing/empty/invalid-type field
            continue
        sql = A.extract_sql(sqlv)               # hardened; None if not real SQL
        if not sql:
            continue
        pairs.append((desc.strip(), sql))
    return pairs


# ----------------------------------------------------------------- generate
def generate(question: str, schema, config=DEFAULT_CONFIG, transport=None) -> AgentOutput:
    """One Ollama request -> up to two interpretation/SQL pairs. `schema` is the
    agent-facing schema string (AmbiQT schema_without_content) or pre-formatted."""
    schema_text = A.format_schema(schema) if ("|" in schema or ":" in schema) and "\n" not in schema else schema
    prompt = build_prompt(question, schema_text)
    settings = config.generation_settings()
    t0 = time.perf_counter()
    try:
        resp = (transport or A._ollama_transport)(prompt, config)
    except Exception as e:
        return AgentOutput(
            agent_name=AGENT_NAME, candidate_sqls=[], interpretation_descriptions=[],
            raw_response=None, model=config.model, generation_settings=settings,
            generation_status="request_error", latency=time.perf_counter() - t0,
            num_generation_calls=1, done_reason=None,
            generation_error=f"{type(e).__name__}: {e}")
    latency = time.perf_counter() - t0
    if not isinstance(resp, dict):
        resp = {"response": str(resp)}
    raw = resp.get("response", "") or ""
    done_reason = resp.get("done_reason")
    prompt_tokens = resp.get("prompt_eval_count")
    total_tokens = resp.get("eval_count")

    # Truncation guard (same policy as A/A2): a length-cap means cut-off output;
    # a partial interpretation/SQL pair is never a valid candidate.
    if done_reason == "length":
        return AgentOutput(
            agent_name=AGENT_NAME, candidate_sqls=[], interpretation_descriptions=[],
            raw_response=raw, model=config.model, generation_settings=settings,
            generation_status="truncated", latency=latency, num_generation_calls=1,
            prompt_tokens=prompt_tokens, total_generated_tokens=total_tokens,
            done_reason=done_reason,
            generation_error="generation truncated by token budget (done_reason=length)")

    pairs = parse_interpretations(raw, MAX_CANDIDATES)
    descs = [d for d, _ in pairs]
    sqls = [s for _, s in pairs]
    status = "ok" if pairs else "parse_error"
    return AgentOutput(
        agent_name=AGENT_NAME, candidate_sqls=sqls, interpretation_descriptions=descs,
        raw_response=raw, model=config.model, generation_settings=settings,
        generation_status=status, latency=latency, num_generation_calls=1,
        prompt_tokens=prompt_tokens, total_generated_tokens=total_tokens,
        done_reason=done_reason,
        generation_error=None if pairs else "no valid interpretation/SQL pair parsed from response")
