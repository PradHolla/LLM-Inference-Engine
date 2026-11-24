"""Phase 11 CRM agent environment: per-task in-memory copies of the derived DB with guarded sql() and search().

  env = CRMEnv("b2b"); s = env.open_task(); print(s.sql('SELECT COUNT(*) FROM "Case"').text); s.close()
  uv run python -m crm.env --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crmarena"
FOOTER = "[showing {n} of {m} rows; narrow the query]"
READONLY_PRAGMAS = {"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo", "foreign_key_list"}
ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
SNIPPET = "snippet(_fts_search, 3, '[', ']', '...', 24)"


@dataclass
class Observation:
    ok: bool
    text: str
    rows_total: int | None
    rows_shown: int | None
    truncated: bool
    elapsed_ms: float
    error: str | None


def schema_objects(org: str) -> list[dict]:
    """The org's schema list with null-description (not real) fields dropped."""
    objs = json.loads((DATA / f"{org}_schema.json").read_text())
    return [{"object": o["object"], "fields": {k: v for k, v in o["fields"].items() if v is not None}}
            for o in objs]


def render_rows(names: list[str], rows: list[tuple], total: int, char_cap: int) -> tuple[str, int, bool]:
    """Render rows as a repr-style list of dicts cut to whole rows within char_cap; return (text, shown, truncated)."""
    reprs = [repr(dict(zip(names, r))) for r in rows]
    full = "[" + ", ".join(reprs) + "]"
    if len(rows) == total and len(full) <= char_cap:
        return full, len(rows), False
    budget = char_cap - len("\n" + FOOTER.format(n=total, m=total))
    kept, size = [], 2
    for s in reprs:
        add = len(s) + (2 if kept else 0)
        if size + add > budget:
            break
        kept.append(s)
        size += add
    return "[" + ", ".join(kept) + "]\n" + FOOTER.format(n=len(kept), m=total), len(kept), True


def _dedupe(names: list[str]) -> list[str]:
    """Make column names unique (a join selecting two Id columns would collapse in a dict)."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


class TaskSession:
    """One task's private in-memory database; not shared with any other session."""

    def __init__(self, conn: sqlite3.Connection, org: str, row_cap: int, char_cap: int,
                 timeout_s: float, copy_ms: float):
        self.conn, self.org, self.copy_ms = conn, org, copy_ms
        self.row_cap, self.char_cap, self.timeout_s = row_cap, char_cap, timeout_s
        self._deadline: float | None = None
        self._timed_out = False
        self._internal = False
        self._denied: str | None = None
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(self._progress, 4000)
        conn.set_authorizer(self._authorize)

    def _progress(self) -> int:
        if self._deadline is not None and time.monotonic() > self._deadline:
            self._timed_out = True
            return 1
        return 0

    def _authorize(self, action, arg1, arg2, _db, _trigger) -> int:
        # query_only alone is not enough: "PRAGMA query_only = OFF" would lift it, so writes are denied here too.
        if self._internal:
            return sqlite3.SQLITE_OK
        internal = action in (sqlite3.SQLITE_READ, sqlite3.SQLITE_PRAGMA) and (
            (arg1 or "").startswith("_") if action == sqlite3.SQLITE_READ else (arg2 or "").startswith("_"))
        if internal:
            name = arg1 if action == sqlite3.SQLITE_READ else arg2
            self._denied = f"tables whose name starts with '_' are internal and cannot be queried ({name})"
        elif action == sqlite3.SQLITE_PRAGMA:
            if (arg1 or "").lower() in READONLY_PRAGMAS:
                return sqlite3.SQLITE_OK
            self._denied = f"PRAGMA {arg1} is not allowed; allowed: {', '.join(sorted(READONLY_PRAGMAS))}"
        elif action in ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        else:
            self._denied = "only read-only SELECT queries are allowed (the database is read-only)"
        return sqlite3.SQLITE_DENY

    def _run(self, query: str, params: tuple = (), internal: bool = False, cap: int | None = None,
             prefix: str = "") -> Observation:
        """Execute one statement under the timeout and caps; never raises on sqlite errors."""
        t0 = time.monotonic()
        self._deadline, self._timed_out, self._denied, self._internal = t0 + self.timeout_s, False, None, internal
        cap = self.row_cap if cap is None else cap
        try:
            cur = self.conn.execute(query, params)
            names = _dedupe([d[0] for d in cur.description or []])
            kept: list[tuple] = []
            total = 0
            while batch := cur.fetchmany(500):
                if len(kept) < cap:
                    kept.extend(batch[: cap - len(kept)])
                total += len(batch)
        except sqlite3.Error as e:
            if self._timed_out:
                msg = f"Query timed out after {float(self.timeout_s)} s"
            elif self._denied:
                msg = self._denied
            elif "one statement at a time" in str(e):
                msg = "only one SQL statement per call is allowed"
            else:
                msg = str(e)
            return Observation(False, f"Error: {msg}", None, None, False, (time.monotonic() - t0) * 1000, msg)
        finally:
            self._deadline, self._internal = None, False
        text, shown, truncated = render_rows(names, kept, total, self.char_cap - len(prefix))
        return Observation(True, prefix + text, total, shown, truncated, (time.monotonic() - t0) * 1000, None)

    def sql(self, query: str) -> Observation:
        """Run one read-only SQL statement; internal ('_'-prefixed) tables are refused."""
        if not query.strip().strip(";").strip():
            return Observation(False, "Error: empty query", None, None, False, 0.0, "empty query")
        return self._run(query)

    def search(self, query: str, objects: list[str] | None = None, limit: int = 20) -> Observation:
        """Full-text search over indexed text fields, best bm25 first; falls back to OR if AND finds nothing."""
        words = re.findall(r"\w+", query)
        if not words:
            msg = "search query has no searchable words"
            return Observation(False, f"Error: {msg}", None, None, False, 0.0, msg)
        limit = max(1, min(int(limit), self.row_cap))
        where, params = "_fts_search MATCH ?", []
        if objects:
            where += f" AND object IN ({', '.join('?' * len(objects))})"
            params = list(objects)
        stmt = (f"SELECT object, record_id AS Id, field, {SNIPPET} AS snippet FROM _fts_search "
                f"WHERE {where} ORDER BY bm25(_fts_search) LIMIT ?")
        quoted = ['"' + w + '"' for w in words]
        obs = self._run(stmt, (" ".join(quoted), *params, limit), internal=True, cap=limit)
        if obs.ok and obs.rows_total == 0 and len(quoted) > 1:
            obs = self._run(stmt, (" OR ".join(quoted), *params, limit), internal=True, cap=limit,
                            prefix="(no record matched all words; showing records matching any word)\n")
        return obs

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> TaskSession:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CRMEnv:
    """Factory for per-task sessions over one org's derived database."""

    def __init__(self, org: str, db_path: str | Path | None = None, row_cap: int = 200,
                 char_cap: int = 24_000, timeout_s: float = 5.0):
        if org not in ("b2b", "b2c"):
            raise ValueError(f"org must be b2b or b2c, got {org!r}")
        path = Path(db_path) if db_path else DATA / "derived" / f"{org}.db"
        self.db_path = path if path.is_absolute() else ROOT / path
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} missing; run: uv run tools/crmdb.py build --org {org}")
        self.org, self.row_cap, self.char_cap, self.timeout_s = org, row_cap, char_cap, timeout_s

    def open_task(self) -> TaskSession:
        """Copy the derived DB into a fresh in-memory database and wrap it in a TaskSession."""
        t0 = time.perf_counter()
        src = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        mem = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            src.backup(mem)
        finally:
            src.close()
        copy_ms = (time.perf_counter() - t0) * 1000
        return TaskSession(mem, self.org, self.row_cap, self.char_cap, self.timeout_s, copy_ms)


def _selftest() -> int:
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    for org in ("b2b", "b2c"):
        env = CRMEnv(org)
        times = []
        for _ in range(5):
            s = env.open_task()
            times.append(s.copy_ms)
            s.close()
        size_mb = env.db_path.stat().st_size / 1e6
        print(f"[{org}] per-task DB copy ({size_mb:.1f} MB): min {min(times):.1f} ms, "
              f"max {max(times):.1f} ms over 5 copies")
    env = CRMEnv("b2b")
    s = env.open_task()
    o = s.sql('SELECT COUNT(*) FROM "Case"')
    check("b2b Case count is 153", o.ok and o.text == "[{'COUNT(*)': 153}]", o.text)
    o = s.sql('SELECT AVG(julianday(ClosedDate) - julianday(CreatedDate)) AS d FROM "Case" '
              "WHERE ClosedDate IS NOT NULL;")
    check("julianday duration non-null", o.ok and "None" not in o.text, o.text)
    o = s.sql("SELECT * FROM VoiceCallTranscript__c")
    last = o.text.rsplit("\n", 1)[-1]
    m = re.fullmatch(r"\[showing (\d+) of (\d+) rows; narrow the query\]", last)
    check("VoiceCallTranscript__c truncated with footer, M=4033",
          o.ok and o.truncated and m is not None and int(m.group(2)) == 4033 and o.rows_total == 4033
          and int(m.group(1)) == o.rows_shown and len(o.text) <= env.char_cap,
          f"{last!r} len={len(o.text)}")
    o = s.sql("SELECT Id FROM Account LIMIT 3")
    check("small result is not truncated", o.ok and not o.truncated and o.rows_total == 3
          and "showing" not in o.text, o.text[:80])
    o = s.sql('SELECT NoSuchColumn FROM "Case"')
    check("bad column -> ok=False, Error:", not o.ok and o.text.startswith("Error:"), o.text)
    o = s.sql("SELECT 1; SELECT 2")
    check("two statements rejected", not o.ok and "one SQL statement" in o.text, o.text)
    for q in ("INSERT INTO Account(Id) VALUES ('x')", "UPDATE \"Case\" SET Status = 'x'",
              "PRAGMA query_only = OFF", "CREATE TEMP TABLE t(a)", "ATTACH ':memory:' AS z"):
        o = s.sql(q)
        check(f"write rejected: {q[:30]}", not o.ok and o.text.startswith("Error:"), o.text)
    o = s.sql('SELECT COUNT(*) FROM "Case"')
    check("data unchanged after write attempts", o.text == "[{'COUNT(*)': 153}]", o.text)
    for q in ("SELECT * FROM _fts_search", "SELECT COUNT(*) FROM _derived_manifest",
              "SELECT * FROM _fts_search_data", "PRAGMA table_info(_fts_search)"):
        o = s.sql(q)
        check(f"internal table rejected: {q}", not o.ok and "internal" in o.text, o.text)
    o = s.sql('PRAGMA table_info("Case")')
    check("PRAGMA table_info allowed", o.ok and "'Subject'" in o.text, f"rows={o.rows_total}")
    o = s.search("Quantum Circuits")
    check("search('Quantum Circuits') has a Knowledge__kav hit", o.ok and "'Knowledge__kav'" in o.text,
          o.text[:160])
    o = s.search('install "router": (setup) -- guide', objects=["Knowledge__kav"], limit=5)
    check("search with FTS syntax chars is sanitized", o.ok and 0 < o.rows_total <= 5
          and "'object': 'Case'" not in o.text, o.text[:120])
    o = s.search("zzqqxx Quantum")
    check("search falls back to OR", o.ok and o.text.startswith("(no record matched all words")
          and o.rows_total > 0, o.text[:100])
    s2 = env.open_task()
    s.conn.set_authorizer(None)
    s.conn.execute("PRAGMA query_only = OFF")
    s.conn.execute('DELETE FROM "Case"')
    a, b = s.sql('SELECT COUNT(*) FROM "Case"').text, s2.sql('SELECT COUNT(*) FROM "Case"').text
    check("sessions are independent", a == "[{'COUNT(*)': 0}]" and b == "[{'COUNT(*)': 153}]", f"{a} vs {b}")
    results = {}
    threads = [threading.Thread(target=lambda k=k: results.__setitem__(
        k, env.open_task().sql('SELECT COUNT(*) FROM "Case"').text)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    check("4 sessions in threads", all(v == "[{'COUNT(*)': 153}]" for v in results.values()), str(results))
    o = s2.sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 1000000000) "
               "SELECT COUNT(*) FROM c")
    check("pathological query times out", not o.ok and o.text == "Error: Query timed out after 5.0 s"
          and 4900 < o.elapsed_ms < 7000, f"{o.text} ({o.elapsed_ms:.0f} ms)")
    o = s2.sql('SELECT COUNT(*) FROM "Case"')
    check("session usable after timeout", o.ok, o.text)
    s.close()
    s2.close()
    objs = schema_objects("b2c")
    acc = next(x for x in objs if x["object"] == "Contact")
    check("schema_objects drops null fields", all(v is not None for x in objs for v in x["fields"].values()),
          f"b2c Contact fields: {list(acc['fields'])}")
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CRM agent environment")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(_selftest())
    ap.print_help()
