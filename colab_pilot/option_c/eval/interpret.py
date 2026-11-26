"""Structural interpretation classifier for AmbiQT candidate SQL.

Classifies a SQL string by the interpretation it STRUCTURALLY expresses, using
sqlglot (SQLite dialect) AST + scope resolution. It does NOT execute SQL and it
does NOT judge execution correctness (that is a separate stage).

Labels (exactly one per candidate):
  A_ONLY, B_ONLY, BOTH_REFERENCE, NEITHER, UNCLASSIFIABLE

Mapping (verified against all 2,993 eligible golds): syn0 == query1 == A ;
syn1 == query2 == B.

Category rules (metadata is the source of truth, never the question):
  col-synonyms : column syn0 on owning table -> A ; syn1 on owning table -> B.
                 Same-named columns on OTHER tables are NOT evidence.
  tbl-synonyms : table syn0 as a source -> A ; syn1 -> B.
  tbl-split    : base column `col` read from base table -> A ;
                 split table `{t}_{col}` referenced -> B.
  tbl-agg      : live aggregate over base table(s) -> A ; reading precomputed
                 table -> B. Reading precomputed AND re-aggregating => B_ONLY
                 with diagnostic subtype B_WITH_EXTRA_AGG (NOT primary success).

Robustness: unused CTEs are eliminated; `*` is NOT expanded (so it cannot inject
synonym columns); columns are resolved to their owning base table through alias
and subquery scopes. If ownership/interpretation cannot be determined, returns
UNCLASSIFIABLE (never a bare-identifier guess).
"""
from dataclasses import dataclass, field
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope, Scope
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes

A_ONLY = "A_ONLY"
B_ONLY = "B_ONLY"
BOTH_REFERENCE = "BOTH_REFERENCE"
NEITHER = "NEITHER"
UNCLASSIFIABLE = "UNCLASSIFIABLE"
LABELS = {A_ONLY, B_ONLY, BOTH_REFERENCE, NEITHER, UNCLASSIFIABLE}


@dataclass
class Result:
    label: str
    subtype: Optional[str] = None        # e.g. "B_WITH_EXTRA_AGG"
    reason: str = ""
    refs_cols: tuple = field(default=())  # (table, col) referenced, resolved
    refs_tables: tuple = field(default=())


# ----------------------------------------------------------------- metadata
def metadata_from_record(category, rec):
    """Build the classifier metadata from a benchmark validation record."""
    if category == "col-synonyms":
        (t, cm), = rec["extra_map"].items()
        (oc, syns), = cm.items()
        return {"owning_table": t.lower(), "syn0": syns[0].lower(), "syn1": syns[1].lower()}
    if category == "tbl-synonyms":
        (t, syns), = rec["extra_table_map"].items()
        return {"syn0_table": syns[0].lower(), "syn1_table": syns[1].lower()}
    if category == "tbl-split":
        (t, col), = rec["split_map"].items()
        return {"base_table": t.lower(), "split_col": col.lower(),
                "split_table": f"{t.lower()}_{col.lower()}"}
    if category == "tbl-agg":
        return {"precomputed_table": rec["new_table_name"].lower()}
    raise ValueError(f"unknown category {category}")


# ----------------------------------------------------------------- parsing/scope
def _prepare(sql, schema):
    """Parse -> drop unused CTEs -> qualify columns (no star expansion)."""
    ast = sqlglot.parse_one(sql, read="sqlite")      # may raise ParseError
    try:
        ast = eliminate_ctes(ast)
    except Exception:
        pass
    ast = qualify(
        ast, schema=schema, dialect="sqlite",
        qualify_columns=True, validate_qualify_columns=False,
        expand_stars=False,
    )
    return ast


def _resolve_col_table(scope, col, _depth=0):
    """Resolve a column's base table through alias + subquery scopes.
    Returns a lowercased base table name, or None if unresolved."""
    if _depth > 12:
        return None
    qual = col.table
    if qual:
        src = scope.sources.get(qual)
    else:
        # unqualified column: resolvable only when the scope has a single source
        srcs = list(scope.sources.values())
        src = srcs[0] if len(srcs) == 1 else None
    if src is None:
        return None
    if isinstance(src, exp.Table):
        return src.name.lower()
    if isinstance(src, Scope):
        inner = src
        selects = inner.expression.selects if inner.expression else []
        for proj in selects:
            if proj.alias_or_name.lower() == col.name.lower():
                base = proj.unalias() if isinstance(proj, exp.Alias) else proj
                if isinstance(base, exp.Column):
                    return _resolve_col_table(inner, base, _depth + 1)
                return None  # projection is an expression, not a plain column
        # star projection over a single table source -> column flows from it
        if any(isinstance(p, exp.Star) or (isinstance(p, exp.Column) and p.name == "*")
               for p in selects):
            tabs = [s for s in inner.sources.values() if isinstance(s, exp.Table)]
            if len(tabs) == 1:
                return tabs[0].name.lower()
    return None


def _collect(ast):
    """Return (resolved_cols:set[(tbl,col)], source_tables:set[str],
    unresolved_col_names:set[str], has_aggfunc:bool, agg_arg_tables:set[str])."""
    cols, tables, unresolved = set(), set(), set()
    for sc in traverse_scope(ast):
        for s in sc.sources.values():
            if isinstance(s, exp.Table):
                tables.add(s.name.lower())
        for col in sc.columns:
            bt = _resolve_col_table(sc, col)
            if bt:
                cols.add((bt, col.name.lower()))
            else:
                unresolved.add(col.name.lower())
    aggs = list(ast.find_all(exp.AggFunc))
    # which base tables feed aggregate arguments (best-effort)
    agg_arg_tables = set()
    for a in aggs:
        for c in a.find_all(exp.Column):
            # resolve via any scope containing it
            for sc in traverse_scope(ast):
                if c in sc.columns:
                    bt = _resolve_col_table(sc, c)
                    if bt:
                        agg_arg_tables.add(bt)
                    break
    return cols, tables, unresolved, bool(aggs), agg_arg_tables


# ----------------------------------------------------------------- classify
def classify(sql, category, metadata, schema):
    """Classify `sql` for `category`. `metadata` from metadata_from_record;
    `schema` = {table: {col: type}} (from the perturbed DB). Returns Result."""
    try:
        ast = _prepare(sql, schema)
    except Exception as e:
        return Result(UNCLASSIFIABLE, reason=f"parse/qualify error: {type(e).__name__}: {e}")

    try:
        cols, tables, unresolved, has_agg, agg_tables = _collect(ast)
    except Exception as e:
        return Result(UNCLASSIFIABLE, reason=f"scope resolution error: {type(e).__name__}: {e}")

    rc = tuple(sorted(cols)); rt = tuple(sorted(tables))

    if category == "col-synonyms":
        t, s0, s1 = metadata["owning_table"], metadata["syn0"], metadata["syn1"]
        a = (t, s0) in cols
        b = (t, s1) in cols
        if a and b:
            return Result(BOTH_REFERENCE, reason="both synonym columns on owning table", refs_cols=rc, refs_tables=rt)
        if a:
            return Result(A_ONLY, refs_cols=rc, refs_tables=rt)
        if b:
            return Result(B_ONLY, refs_cols=rc, refs_tables=rt)
        # a synonym-named column that we could not pin to a table -> cannot rule out ownership
        if (s0 in unresolved) or (s1 in unresolved):
            return Result(UNCLASSIFIABLE, reason="synonym-named column could not be resolved to a table",
                          refs_cols=rc, refs_tables=rt)
        return Result(NEITHER, reason="no synonym column on owning table", refs_cols=rc, refs_tables=rt)

    if category == "tbl-synonyms":
        s0, s1 = metadata["syn0_table"], metadata["syn1_table"]
        a, b = s0 in tables, s1 in tables
        if a and b:
            return Result(BOTH_REFERENCE, reason="both synonym tables", refs_tables=rt)
        if a:
            return Result(A_ONLY, refs_tables=rt)
        if b:
            return Result(B_ONLY, refs_tables=rt)
        return Result(NEITHER, reason="neither synonym table referenced", refs_tables=rt)

    if category == "tbl-split":
        t, col, st = metadata["base_table"], metadata["split_col"], metadata["split_table"]
        b = st in tables
        a = (t, col) in cols
        if a and b:
            return Result(BOTH_REFERENCE, reason="base column and split table both referenced",
                          refs_cols=rc, refs_tables=rt)
        if b:
            return Result(B_ONLY, refs_cols=rc, refs_tables=rt)
        if a:
            return Result(A_ONLY, refs_cols=rc, refs_tables=rt)
        return Result(NEITHER, reason="neither base column nor split table", refs_cols=rc, refs_tables=rt)

    if category == "tbl-agg":
        ntn = metadata["precomputed_table"]
        b = ntn in tables
        base_tables = tables - {ntn}
        if b and has_agg and (agg_tables - {ntn}):
            return Result(BOTH_REFERENCE, reason="reads precomputed AND aggregates over base",
                          refs_tables=rt)
        if b:
            sub = "B_WITH_EXTRA_AGG" if has_agg else None
            return Result(B_ONLY, subtype=sub,
                          reason="reads precomputed table" + (" with extra aggregation" if sub else ""),
                          refs_tables=rt)
        if has_agg and base_tables:
            return Result(A_ONLY, reason="live aggregation over base", refs_tables=rt)
        return Result(NEITHER, reason="neither precomputed read nor live aggregation", refs_tables=rt)

    return Result(UNCLASSIFIABLE, reason=f"unknown category {category}")
