"""Safe, read-only, timeout-bounded SQL execution for evaluation.

Builds on option_c/execute.py's connection convention (lenient text_factory) but:
  * opens the database READ-ONLY (URI mode=ro) so no candidate SQL can ever
    modify the original or reconstructed databases, and
  * enforces a wall-clock timeout via a watchdog that calls Connection.interrupt()
    (safe to call from another thread), so a timed-out query is actually aborted
    and cannot keep running in the background.

Status is one of: "ok" | "error" | "timeout".
A successful query that returns zero rows is "ok" (rows == []), never an error.
Also provides schema_from_db(), the single schema-extraction helper reused by
golds.py and the later scoring pipeline.
"""
import os
import time
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# reuse the text_factory convention from the existing module
try:
    from execute import connect as _base_connect  # when run with option_c/eval on sys.path
except ImportError:  # pragma: no cover
    _base_connect = None


@dataclass
class ExecResult:
    status: str                       # "ok" | "error" | "timeout"
    rows: Optional[List[Tuple]] = None
    rowcount: Optional[int] = None
    error: Optional[str] = None
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _ro_uri(db_path: str) -> str:
    # absolute path -> read-only SQLite URI (forward slashes; works on Windows)
    p = os.path.abspath(db_path).replace("\\", "/")
    return "file:" + p + "?mode=ro"


def connect_ro(db_path: str) -> sqlite3.Connection:
    """Read-only connection with the lenient text_factory convention."""
    con = sqlite3.connect(_ro_uri(db_path), uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace") if isinstance(b, (bytes, bytearray)) else b
    return con


def execute_sql(db_path: str, query: str, timeout: float = 10.0) -> ExecResult:
    """Execute `query` read-only with a wall-clock `timeout` (seconds)."""
    t0 = time.perf_counter()
    con = None
    timed_out = {"v": False}
    timer = None
    try:
        con = connect_ro(db_path)

        def _watchdog():
            timed_out["v"] = True
            try:
                con.interrupt()
            except Exception:
                pass

        timer = threading.Timer(timeout, _watchdog)
        timer.start()
        cur = con.execute(query)
        rows = cur.fetchall()
        return ExecResult("ok", rows=rows, rowcount=len(rows),
                          elapsed=time.perf_counter() - t0)
    except Exception as e:
        if timed_out["v"]:
            return ExecResult("timeout", error=f"exceeded {timeout}s",
                              elapsed=time.perf_counter() - t0)
        return ExecResult("error", error=f"{type(e).__name__}: {e}",
                          elapsed=time.perf_counter() - t0)
    finally:
        if timer is not None:
            timer.cancel()
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def schema_from_db(db_path: str) -> dict:
    """Return {table: {column: type}} for sqlglot qualification. Read-only."""
    con = connect_ro(db_path)
    try:
        sch = {}
        for (t,) in con.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%'"):
            sch[t] = {r[1]: (r[2] or "TEXT") for r in con.execute(f'PRAGMA table_info("{t}")')}
        return sch
    finally:
        con.close()
