"""SQLite persistence for chat branches.

  uv run python -m app.db --selftest
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
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
  created_at REAL    NOT NULL
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
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(chat_id: int, parent_id: int | None, role: str, content: str,
                thinking: str | None, tokens: int | None) -> int:
    """Append a message to a branch and make it the active leaf."""
    now = time.time()
    with _connection() as conn:
        cur = conn.execute(
            """INSERT INTO messages (chat_id, parent_id, role, content, thinking, tokens, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, parent_id, role, content, thinking, tokens, now),
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
    branch: list[sqlite3.Row] = []
    message_id = chat["head_message_id"]
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


def delete_message(message_id: int) -> None:
    """Remove a message whose upstream request never opened."""
    with _connection() as conn:
        conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))


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
