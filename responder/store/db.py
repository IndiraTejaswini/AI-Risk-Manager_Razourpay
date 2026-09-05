"""SQLite substrate for the responder transactional outbox."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a responder database, enable WAL, and apply the idempotent schema."""
    connection = sqlite3.connect(path, check_same_thread=False)
    journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(journal_mode).lower() != "wal":
        connection.close()
        raise RuntimeError(f"SQLite WAL mode was not enabled: {journal_mode!r}")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return connection
