"""Safe SQLite connection + query execution.

Distinguishes an execution ERROR from a successful EMPTY result:
  run_query -> ('ok', rows)          # rows may be [] (genuine empty result)
            -> ('error', 'MsgText')  # exception during execute/fetch
"""
import sqlite3


def connect(db_path):
    con = sqlite3.connect(db_path)
    # Some Spider DBs contain non-UTF8 bytes; decode leniently so execution
    # never dies on a decode error (keeps comparison deterministic).
    con.text_factory = lambda b: b.decode("utf-8", "replace") if isinstance(b, (bytes, bytearray)) else b
    return con


def run_query(db_path, query):
    """Execute query, return ('ok', rows) or ('error', message).
    rows is a list of tuples; an empty list is a genuine empty result, NOT an error."""
    con = None
    try:
        con = connect(db_path)
        cur = con.execute(query)
        rows = cur.fetchall()
        return ("ok", rows)
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")
    finally:
        if con is not None:
            con.close()
