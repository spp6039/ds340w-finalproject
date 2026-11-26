"""Result-comparison policy. Assumptions are explicit and documented here.

POLICY (for AmbiQT col-synonyms / tbl-synonyms / tbl-split):
  * Column order IS significant: each result row is compared positionally as a
    tuple. Justification: query1, query2 and orig_query all list the same
    logical columns in the same order (verified from generate.py), so a
    positional tuple comparison is the correct, strict check.
  * Duplicate rows ARE preserved: we use multiset/bag comparison
    (collections.Counter over row tuples), never set() -- duplicates are never
    silently collapsed.
  * Row order is significant ONLY when the gold query has ORDER BY. We detect
    ' order by ' in the normalized gold SQL and, if present, require the exact
    ordered sequence; otherwise we require multiset equality.
  * NULLs: SQLite returns Python None; None==None inside tuples, so NULLs are
    compared structurally and never dropped.
  * Data types: values are compared by SQLite storage class as returned by the
    driver (int/float/str/bytes/None). For these three categories the
    perturbation introduces NO casting or aggregation (columns/tables are
    verbatim copies), so int-vs-float drift cannot occur and exact comparison
    is appropriate. (For a category that introduced aggregates -- e.g. tbl-agg,
    out of scope here -- a numeric-tolerant mode would be needed.)
"""
from collections import Counter


def has_order_by(sql):
    return " order by " in (" " + " ".join(sql.lower().split()) + " ")


def compare(rows_a, rows_b, ordered):
    """Return (equal: bool, detail: dict). rows_* are lists of row tuples/lists."""
    ta = [tuple(r) for r in rows_a]
    tb = [tuple(r) for r in rows_b]
    ordered_equal = (ta == tb)
    try:
        multiset_equal = (Counter(ta) == Counter(tb))
    except TypeError:
        # Unhashable cell (shouldn't happen with sqlite scalars); fall back to
        # sorted-list comparison using repr as key, still duplicate-preserving.
        multiset_equal = (sorted(map(repr, ta)) == sorted(map(repr, tb)))
    equal = ordered_equal if ordered else multiset_equal
    return equal, {
        "n_a": len(ta),
        "n_b": len(tb),
        "ordered_mode": ordered,
        "ordered_equal": ordered_equal,
        "multiset_equal": multiset_equal,
    }
