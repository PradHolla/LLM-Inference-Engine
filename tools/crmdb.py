#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Phase 11: build and check the derived CRMArena-Pro databases (fixed timestamps + FTS5 search).

  uv run tools/crmdb.py build --org all
  uv run tools/crmdb.py check --org all
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crmarena"
CANONICAL_SHA = {
    "b2b": "cb537bf8670053702ec35ba70aa8151a6dfe6261a382e4c3b7fa23c6677ac2e5",
    "b2c": "b0be46eff7c7bdb193b93e868d78b39d9117c90e1f6cc0138e9b57a6e2b9a34c",
}
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+0000$", re.ASCII)

# (object, field label, source columns); several columns are joined with a space.
FTS_SPEC = [
    ("Account", "Name", ["Name"]),
    ("Contact", "Name", ["FirstName", "LastName"]),
    ("Lead", "Name", ["FirstName", "LastName"]),
    ("Lead", "Company", ["Company"]),
    ("User", "Name", ["FirstName", "LastName"]),
    ("Product2", "Name", ["Name"]),
    ("Product2", "Description", ["Description"]),
    ("Opportunity", "Name", ["Name"]),
    ("Opportunity", "Description", ["Description"]),
    ("Case", "Subject", ["Subject"]),
    ("Case", "Description", ["Description"]),
    ("Issue__c", "Name", ["Name"]),
    ("Knowledge__kav", "Title", ["Title"]),
    ("Knowledge__kav", "Summary", ["Summary"]),
    ("Knowledge__kav", "FAQ_Answer__c", ["FAQ_Answer__c"]),
    ("EmailMessage", "Subject", ["Subject"]),
    ("EmailMessage", "TextBody", ["TextBody"]),
    ("VoiceCallTranscript__c", "Body__c", ["Body__c"]),
    ("LiveChatTranscript", "Body", ["Body"]),
    ("Quote", "Name", ["Name"]),
    ("Quote", "Description", ["Description"]),
    ("Contract", "Description", ["Description"]),
]
# Objects where every column whose name contains "description" is also indexed.
FTS_DESCRIPTION_LIKE = ["Issue__c"]


def canonical_path(org: str) -> Path:
    return DATA / "canonical" / f"crmarenapro_{org}_data.db"


def derived_path(org: str) -> Path:
    return DATA / "derived" / f"{org}.db"


def q(name: str) -> str:
    """Double-quote an SQL identifier (Case and Order are keywords)."""
    return '"' + name.replace('"', '""') + '"'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE '\\_%' ESCAPE '\\' ORDER BY name").fetchall()
    return [r[0] for r in rows]


def columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    return [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({q(table)})")]


def open_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(f"missing database: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def rewrite_timestamps(conn: sqlite3.Connection) -> dict[str, int]:
    """Replace a trailing +0000 with +00:00 on exact timestamp matches; count per table.column."""
    counts: dict[str, int] = {}
    for t in user_tables(conn):
        cols = [c for c, typ in columns(conn, t) if typ.upper() == "TEXT"]
        for c in cols:
            rows = conn.execute(
                f"SELECT rowid, {q(c)} FROM {q(t)} WHERE typeof({q(c)}) = 'text' "
                f"AND {q(c)} LIKE '%+0000'").fetchall()
            fixed = [(v[:-5] + "+00:00", rid) for rid, v in rows if TS_RE.match(v)]
            if fixed:
                conn.executemany(f"UPDATE {q(t)} SET {q(c)} = ? WHERE rowid = ?", fixed)
                counts[f"{t}.{c}"] = len(fixed)
    return counts


def build_fts(conn: sqlite3.Connection) -> tuple[int, list[str], dict[str, int]]:
    """Create and fill _fts_search; return (rows, missing 'Object.Column' names, rows per object.field)."""
    conn.execute("CREATE VIRTUAL TABLE _fts_search USING fts5("
                 "object UNINDEXED, record_id UNINDEXED, field UNINDEXED, text, "
                 "tokenize='porter unicode61')")
    tables = set(user_tables(conn))
    spec = list(FTS_SPEC)
    for obj in FTS_DESCRIPTION_LIKE:
        if obj in tables:
            spec += [(obj, c, [c]) for c, _ in columns(conn, obj) if "description" in c.lower()]
    missing: list[str] = []
    per_field: dict[str, int] = {}
    for obj, label, cols in spec:
        if obj not in tables:
            missing.append(f"{obj} (table)")
            continue
        have = {c for c, _ in columns(conn, obj)}
        absent = [c for c in cols if c not in have]
        missing += [f"{obj}.{c}" for c in absent]
        present = [c for c in cols if c in have]
        if not present:
            continue
        sel = ", ".join(q(c) for c in present)
        rows = []
        for rec in conn.execute(f"SELECT Id, {sel} FROM {q(obj)}"):
            parts = [str(v).strip() for v in rec[1:] if v is not None and str(v).strip()]
            if parts:
                rows.append((obj, rec[0], label, " ".join(parts)))
        conn.executemany("INSERT INTO _fts_search(object, record_id, field, text) VALUES (?,?,?,?)", rows)
        per_field[f"{obj}.{label}"] = len(rows)
    n = conn.execute("SELECT COUNT(*) FROM _fts_search").fetchone()[0]
    return n, missing, per_field


def build(org: str) -> None:
    src_path, out = canonical_path(org), derived_path(org)
    digest = sha256(src_path)
    assert digest == CANONICAL_SHA[org], f"{org}: canonical sha256 {digest} != expected {CANONICAL_SHA[org]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    src = open_ro(src_path)
    dst = sqlite3.connect(tmp)
    src.backup(dst)
    src.close()
    with dst:
        rewrites = rewrite_timestamps(dst)
        fts_rows, missing, per_field = build_fts(dst)
        counts = {t: dst.execute(f"SELECT COUNT(*) FROM {q(t)}").fetchone()[0] for t in user_tables(dst)}
        dst.execute("CREATE TABLE _derived_manifest(key TEXT PRIMARY KEY, value TEXT)")
        manifest = {
            "canonical_path": str(src_path.relative_to(ROOT)),
            "canonical_sha256": digest,
            "build_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "row_counts": json.dumps(counts, sort_keys=True),
            "rewrite_counts": json.dumps(rewrites, sort_keys=True),
            "fts_rows": str(fts_rows),
            "fts_rows_per_field": json.dumps(per_field, sort_keys=True),
            "fts_missing_columns": json.dumps(missing),
        }
        dst.executemany("INSERT INTO _derived_manifest VALUES (?, ?)", manifest.items())
    dst.execute("VACUUM")
    dst.close()
    os.replace(tmp, out)
    print(f"[{org}] wrote {out.relative_to(ROOT)} (canonical sha256 ok)")
    print(f"[{org}] tables: {len(counts)}  rows: {sum(counts.values())}")
    print(f"[{org}] timestamp rewrites ({sum(rewrites.values())} total):")
    for k, v in sorted(rewrites.items()):
        print(f"    {k:<45} {v}")
    print(f"[{org}] FTS rows: {fts_rows}")
    for k, v in sorted(per_field.items()):
        print(f"    {k:<45} {v}")
    print(f"[{org}] expected columns not found: {missing or 'none'}")


def check(org: str) -> list[tuple[str, bool, str]]:
    """Return (check name, passed, detail) rows for one org."""
    res: list[tuple[str, bool, str]] = []
    path = derived_path(org)
    if not path.exists():
        return [(f"{org}: derived db exists", False, f"missing {path}")]
    can, der = open_ro(canonical_path(org)), open_ro(path)
    ctabs, dtabs = user_tables(can), user_tables(der)
    res.append((f"{org}: same 27 tables", ctabs == dtabs and len(ctabs) == 27, f"{len(ctabs)} canonical, {len(dtabs)} derived"))
    schema_c = dict(can.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall())
    bad_schema = [t for t in ctabs if schema_c.get(t) != der.execute(
        "SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()[0]]
    res.append((f"{org}: CREATE TABLE statements identical", not bad_schema, ", ".join(bad_schema) or "all equal"))
    bad_rows = []
    for t in ctabs:
        a = can.execute(f"SELECT COUNT(*) FROM {q(t)}").fetchone()[0]
        b = der.execute(f"SELECT COUNT(*) FROM {q(t)}").fetchone()[0] if t in dtabs else None
        if a != b:
            bad_rows.append(f"{t} {a}!={b}")
    res.append((f"{org}: row counts equal canonical", not bad_rows, ", ".join(bad_rows) or f"{len(ctabs)} tables"))
    left = 0
    for t in dtabs:
        for c, _ in columns(der, t):
            left += der.execute(f"SELECT COUNT(*) FROM {q(t)} WHERE typeof({q(c)})='text' "
                                f"AND {q(c)} LIKE '%+0000'").fetchone()[0]
    res.append((f"{org}: no +0000-suffixed values left", left == 0, f"{left} remaining"))
    # Every changed cell must be exactly the +0000 -> +00:00 rewrite, and their count must match the manifest.
    der.execute("ATTACH DATABASE ? AS c", (f"file:{canonical_path(org)}?mode=ro",))
    changed, other = 0, 0
    for t in ctabs:
        for col, _ in columns(der, t):
            n_all, n_ts = der.execute(
                f"SELECT COUNT(*), SUM(d.{q(col)} = substr(o.{q(col)}, 1, length(o.{q(col)}) - 5) || '+00:00' "
                f"AND o.{q(col)} LIKE '%+0000') FROM main.{q(t)} d JOIN c.{q(t)} o ON d.rowid = o.rowid "
                f"WHERE d.{q(col)} IS NOT o.{q(col)}").fetchone()
            changed += n_ts or 0
            other += n_all - (n_ts or 0)
    der.execute("DETACH DATABASE c")
    man_rw = sum(json.loads(der.execute(
        "SELECT value FROM _derived_manifest WHERE key='rewrite_counts'").fetchone()[0]).values())
    res.append((f"{org}: only timestamp cells differ from canonical", other == 0 and changed == man_rw,
                f"{changed} rewritten (manifest {man_rw}), {other} other diffs"))
    for t in ("Case", "Opportunity", "Lead"):
        n, bad = der.execute(f"SELECT COUNT(*), SUM(julianday(CreatedDate) IS NULL) FROM {q(t)}").fetchone()
        res.append((f"{org}: julianday({t}.CreatedDate) non-null", n > 0 and not bad, f"{bad or 0} null of {n}"))
    n, bad = der.execute('SELECT COUNT(*), SUM(julianday(ClosedDate) IS NULL) FROM "Case" '
                         "WHERE ClosedDate IS NOT NULL").fetchone()
    res.append((f"{org}: julianday(Case.ClosedDate) non-null", n > 0 and not bad, f"{bad or 0} null of {n} non-null"))
    fts_n = der.execute("SELECT COUNT(*) FROM _fts_search").fetchone()[0]
    man = dict(der.execute("SELECT key, value FROM _derived_manifest").fetchall())
    res.append((f"{org}: manifest sha256 + FTS count", man.get("canonical_sha256") == CANONICAL_SHA[org]
                and man.get("fts_rows") == str(fts_n), f"fts_rows={fts_n}"))
    title = der.execute("SELECT Title FROM Knowledge__kav WHERE Title IS NOT NULL ORDER BY Id LIMIT 1").fetchone()[0]
    word = max(re.findall(r"\w+", title), key=len)
    probes = [("Knowledge__kav", word),
              ("Case", der.execute('SELECT Subject FROM "Case" ORDER BY Id LIMIT 1').fetchone()[0])]
    if "Name" in {c for c, _ in columns(der, "Account")}:  # b2c Account has no Name column
        probes.append(("Account", der.execute("SELECT Name FROM Account ORDER BY Id LIMIT 1").fetchone()[0]))
    for obj, text in probes:
        terms = " ".join('"' + w + '"' for w in re.findall(r"\w+", text))
        hits = der.execute("SELECT COUNT(*) FROM _fts_search WHERE _fts_search MATCH ? AND object = ?",
                           (terms, obj)).fetchone()[0]
        res.append((f"{org}: FTS probe {obj} {text[:30]!r}", hits > 0, f"{hits} hits"))
    can.close()
    der.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--org", choices=["b2b", "b2c", "all"], default="all")
    a = ap.parse_args()
    orgs = ["b2b", "b2c"] if a.org == "all" else [a.org]
    if a.cmd == "build":
        for org in orgs:
            build(org)
        return
    rows = [r for org in orgs for r in check(org)]
    width = max(len(r[0]) for r in rows)
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(not ok for _, ok, _ in rows)
    print(f"{len(rows) - failed}/{len(rows)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
