"""SQLite persistence for chat branches.

  uv run python -m app.db --selftest
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import json
from contextlib import contextmanager

from .config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
  id               INTEGER PRIMARY KEY,
  title            TEXT    NOT NULL,
  created_at       REAL    NOT NULL,
  updated_at       REAL    NOT NULL,
  thinking_default TEXT    NOT NULL,
  head_message_id  INTEGER
);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY,
  chat_id    INTEGER NOT NULL,
  parent_id  INTEGER,
  role       TEXT    NOT NULL,
  content    TEXT    NOT NULL,
  thinking   TEXT,
  tokens     INTEGER,
  created_at REAL    NOT NULL,
  sources_json TEXT,
  stats_json TEXT,
  stopped INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS searches (
  id          INTEGER PRIMARY KEY,
  query       TEXT NOT NULL,
  results_json TEXT NOT NULL,
  fetched_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
  request_id  TEXT PRIMARY KEY,
  chat_id     INTEGER,
  turn_index  INTEGER,
  app_rtt_ms  REAL,
  trace_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
  covered_through_id INTEGER PRIMARY KEY,
  chat_id            INTEGER NOT NULL,
  text               TEXT    NOT NULL,
  created_at         REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the SQLite schema if it does not already exist."""
    with _connection() as conn:
        conn.executescript(SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
        for name, declaration in (("sources_json", "TEXT"), ("stats_json", "TEXT"),
                                  ("stopped", "INTEGER NOT NULL DEFAULT 0")):
            if name not in columns:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {declaration}")


def create_chat(title: str, thinking_default: str) -> int:
    """Create an empty chat and return its primary key."""
    now = time.time()
    with _connection() as conn:
        cur = conn.execute(
            "INSERT INTO chats (title, created_at, updated_at, thinking_default) VALUES (?, ?, ?, ?)",
            (title, now, now, thinking_default),
        )
        return int(cur.lastrowid)


def list_chats() -> list[sqlite3.Row]:
    """Return chats with the most recently updated first."""
    with _connection() as conn:
        return conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()


def get_chat(chat_id: int) -> sqlite3.Row | None:
    """Return one chat, or None when it does not exist."""
    with _connection() as conn:
        return conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()


def rename_chat(chat_id: int, title: str) -> None:
    """Rename a chat and mark it as recently updated."""
    with _connection() as conn:
        conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                     (title, time.time(), chat_id))


def delete_chat(chat_id: int) -> None:
    """Delete a chat and every message belonging to it."""
    with _connection() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM summaries WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(chat_id: int, parent_id: int | None, role: str, content: str,
                thinking: str | None, tokens: int | None,
                sources: list[dict] | None = None, stats: dict | None = None,
                stopped: bool = False) -> int:
    """Append a message to a branch and make it the active leaf."""
    now = time.time()
    with _connection() as conn:
        cur = conn.execute(
            """INSERT INTO messages
               (chat_id, parent_id, role, content, thinking, tokens, created_at,
                sources_json, stats_json, stopped)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, parent_id, role, content, thinking, tokens, now,
             json.dumps(sources) if sources is not None else None,
             json.dumps(stats) if stats is not None else None, int(stopped)),
        )
        message_id = int(cur.lastrowid)
        conn.execute("UPDATE chats SET head_message_id = ?, updated_at = ? WHERE id = ?",
                     (message_id, now, chat_id))
        return message_id


def get_branch(chat_id: int) -> list[sqlite3.Row]:
    """Return the active message branch from root to leaf."""
    chat = get_chat(chat_id)
    if chat is None or chat["head_message_id"] is None:
        return []
    return get_path(chat_id, int(chat["head_message_id"]))


def get_path(chat_id: int, leaf_id: int) -> list[sqlite3.Row]:
    """Return one root-to-leaf path, or an empty list for an unrelated message."""
    branch: list[sqlite3.Row] = []
    message_id: int | None = leaf_id
    with _connection() as conn:
        while message_id is not None:
            message = conn.execute("SELECT * FROM messages WHERE id = ? AND chat_id = ?",
                                   (message_id, chat_id)).fetchone()
            if message is None:
                break
            branch.append(message)
            message_id = message["parent_id"]
    branch.reverse()
    return branch


def get_message(chat_id: int, message_id: int) -> sqlite3.Row | None:
    """Return a message only when it belongs to the given chat."""
    with _connection() as conn:
        return conn.execute("SELECT * FROM messages WHERE id = ? AND chat_id = ?",
                            (message_id, chat_id)).fetchone()


def sibling_ids(chat_id: int, message_id: int) -> list[int]:
    """Return messages with the same parent, in creation order."""
    message = get_message(chat_id, message_id)
    if message is None:
        return []
    with _connection() as conn:
        rows = conn.execute(
            "SELECT id FROM messages WHERE chat_id = ? AND parent_id IS ? ORDER BY id",
            (chat_id, message["parent_id"]),
        ).fetchall()
    return [int(row["id"]) for row in rows]


def newest_leaf(chat_id: int, message_id: int) -> int | None:
    """Return the newest leaf in the subtree rooted at a message."""
    rows = get_messages(chat_id)
    by_parent: dict[int | None, list[sqlite3.Row]] = {}
    by_id = {int(row["id"]): row for row in rows}
    if message_id not in by_id:
        return None
    for row in rows:
        by_parent.setdefault(row["parent_id"], []).append(row)
    pending = [message_id]
    leaves: list[sqlite3.Row] = []
    while pending:
        current = pending.pop()
        children = by_parent.get(current, [])
        if children:
            pending.extend(int(child["id"]) for child in children)
        else:
            leaves.append(by_id[current])
    return int(max(leaves, key=lambda row: (row["created_at"], row["id"]))["id"])


def get_messages(chat_id: int) -> list[sqlite3.Row]:
    """Return every message in a chat in insertion order."""
    with _connection() as conn:
        return conn.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id",
                            (chat_id,)).fetchall()


def delete_message(message_id: int) -> None:
    """Remove a message whose upstream request never opened."""
    with _connection() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))


def rollback_user_message(chat_id: int, message_id: int,
                          previous_head_id: int | None) -> None:
    """Atomically remove a user turn whose gateway stream never opened."""
    with _connection() as conn:
        conn.execute("DELETE FROM messages WHERE id = ? AND chat_id = ?",
                     (message_id, chat_id))
        conn.execute("UPDATE chats SET head_message_id = ?, updated_at = ? WHERE id = ?",
                     (previous_head_id, time.time(), chat_id))


def set_head(chat_id: int, message_id: int | None) -> None:
    """Move a chat's active branch pointer to an existing message."""
    with _connection() as conn:
        conn.execute("UPDATE chats SET head_message_id = ?, updated_at = ? WHERE id = ?",
                     (message_id, time.time(), chat_id))


def save_trace(request_id: str, chat_id: int, turn_index: int, app_rtt_ms: float,
               trace_json: str) -> None:
    """Store one gateway trace joined to the app's chat turn."""
    with _connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO traces
               (request_id, chat_id, turn_index, app_rtt_ms, trace_json)
               VALUES (?, ?, ?, ?, ?)""",
            (request_id, chat_id, turn_index, app_rtt_ms, trace_json),
        )


def save_summary(covered_through_id: int, chat_id: int, text: str) -> None:
    """Store a summary once; the first text wins so the prompt prefix stays byte-identical."""
    with _connection() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO summaries (covered_through_id, chat_id, text, created_at)
               VALUES (?, ?, ?, ?)""", (covered_through_id, chat_id, text, time.time()))


def get_summary(covered_through_id: int) -> str | None:
    """Return the summary covering history through this message, if one is stored."""
    with _connection() as conn:
        row = conn.execute("SELECT text FROM summaries WHERE covered_through_id = ?",
                           (covered_through_id,)).fetchone()
    return row["text"] if row is not None else None


def latest_summary(message_ids: list[int]) -> tuple[int, str] | None:
    """The deepest stored summary among these path ids, as (index in the list, text)."""
    if not message_ids:
        return None
    marks = ",".join("?" for _ in message_ids)
    with _connection() as conn:
        rows = conn.execute(f"SELECT covered_through_id, text FROM summaries "
                            f"WHERE covered_through_id IN ({marks})", message_ids).fetchall()
    found = {int(row["covered_through_id"]): row["text"] for row in rows}
    for index in range(len(message_ids) - 1, -1, -1):
        if message_ids[index] in found:
            return index, found[message_ids[index]]
    return None


def selftest() -> int:
    """Exercise active-branch storage against a temporary database."""
    global DB_PATH
    fails: list[str] = []
    old_path = DB_PATH

    def check(name: str, condition: bool) -> None:
        if not condition:
            fails.append(f"  FAIL {name}")

    with tempfile.TemporaryDirectory() as tmp:
        DB_PATH = os.path.join(tmp, "chats.db")
        try:
            init_db()
            chat_id = create_chat("test", "brief")
            first = add_message(chat_id, None, "user", "one", None, None)
            second = add_message(chat_id, first, "assistant", "two", None, 2)
            third = add_message(chat_id, second, "user", "three", None, None)
            check("linear branch order", [row["content"] for row in get_branch(chat_id)] ==
                  ["one", "two", "three"])

            branch = add_message(chat_id, first, "assistant", "branch", None, 1)
            check("branched order", [row["content"] for row in get_branch(chat_id)] ==
                  ["one", "branch"])
            with _connection() as conn:
                stored_ids = {row[0] for row in conn.execute(
                    "SELECT id FROM messages WHERE chat_id = ?", (chat_id,))}
            check("original messages retained", {first, second, third} <= stored_ids)
            chat = get_chat(chat_id)
            check("head is newest message", chat is not None and chat["head_message_id"] == branch)
            second_branch = add_message(chat_id, first, "assistant", "sibling", None, 1)
            check("sibling order", sibling_ids(chat_id, branch) ==
                  [second, branch, second_branch])
            check("path lookup", [r["content"] for r in get_path(chat_id, third)] ==
                  ["one", "two", "three"])
            check("newest leaf", newest_leaf(chat_id, first) == second_branch)

            migration_path = os.path.join(tmp, "old.db")
            DB_PATH = migration_path
            with _connection() as conn:
                conn.executescript("""
                    CREATE TABLE chats (id INTEGER PRIMARY KEY, title TEXT NOT NULL,
                      created_at REAL NOT NULL, updated_at REAL NOT NULL,
                      thinking_default TEXT NOT NULL, head_message_id INTEGER);
                    CREATE TABLE messages (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL,
                      parent_id INTEGER, role TEXT NOT NULL, content TEXT NOT NULL,
                      thinking TEXT, tokens INTEGER, created_at REAL NOT NULL);
                    INSERT INTO chats VALUES (9, 'old', 1, 1, 'brief', NULL);
                    INSERT INTO messages VALUES (7, 9, NULL, 'user', 'kept', NULL, NULL, 1);
                """)
            init_db()
            migrated = get_message(9, 7)
            check("old schema data preserved", migrated is not None and migrated["content"] == "kept")
            check("old schema gains metadata columns", migrated is not None and
                  {"sources_json", "stats_json", "stopped"} <= set(migrated.keys()))
            with _connection() as conn:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'")}
                old_chat = conn.execute("SELECT * FROM chats WHERE id = 9").fetchone()
            check("old schema gains the summaries table", "summaries" in tables)
            check("old chat row untouched by the migration", old_chat is not None and
                  tuple(old_chat) == (9, "old", 1, 1, "brief", None))
            init_db()
            check("migration is idempotent", get_message(9, 7)["content"] == "kept")
            save_summary(7, 9, "first text")
            save_summary(7, 9, "second text")
            check("first summary text wins", get_summary(7) == "first text")
            check("missing summary is None", get_summary(8) is None)
            check("latest summary on a path", latest_summary([1, 7, 3]) == (1, "first text")
                  and latest_summary([1, 3]) is None and latest_summary([]) is None)
            delete_chat(9)
            check("deleting a chat removes its summaries", get_summary(7) is None)
        finally:
            DB_PATH = old_path

    print("\n".join(fails) if fails else "selftest: PASS")
    if fails:
        print(f"selftest: {len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
