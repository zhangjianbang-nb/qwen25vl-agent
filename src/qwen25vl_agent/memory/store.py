"""SQLite storage for the five memories.

Everything the agent remembers lives in one SQLite file, in WAL mode.  SQLite is the right
choice here for unglamorous reasons:

* it is in the standard library, so the memory layer adds no dependency;
* a 24/7 writer plus a reader (the CLI inspecting memory while the agent runs) is exactly what
  WAL mode is for -- readers never block the writer;
* the archive step is a plain SQL ``DELETE`` plus a JSONL append, not a compaction job.

Embeddings are stored as float32 blobs and matched with numpy (see ``vectors``), because a
household-scale gallery is tens of rows.  A vector database would be more machinery than the
problem needs.

The schema is created on first open and is additive: ``CREATE TABLE IF NOT EXISTS`` for every
table, so an existing database keeps working when a new memory type is added.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Iterable, Optional

__all__ = ["MemoryStore", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Person memory: one row per identity, with face and voice embeddings hanging off it.
CREATE TABLE IF NOT EXISTS persons (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    first_seen REAL,
    last_seen  REAL,
    seen_count INTEGER NOT NULL DEFAULT 0,
    notes      TEXT NOT NULL DEFAULT '',
    meta       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    ts        REAL NOT NULL,
    vec       BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    quality   REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS voice_embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    ts        REAL NOT NULL,
    vec       BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    quality   REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS interactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    kind      TEXT NOT NULL,
    text      TEXT NOT NULL DEFAULT ''
);

-- Episodic memory: what happened, when, and where the frame is.
CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    t_end      REAL,
    kind       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '{}',
    person_ids TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.0,
    frame_path TEXT
);

-- Raw perception log: the unsummarized backbone the archive keeps (and prunes).
CREATE TABLE IF NOT EXISTS observations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL DEFAULT '',
    person_ids TEXT NOT NULL DEFAULT '[]',
    payload    TEXT NOT NULL DEFAULT '{}',
    frame_path TEXT
);

-- Semantic memory: durable facts, keyed so re-learning a fact updates rather than duplicates.
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    updated_at REAL NOT NULL,
    subject    TEXT NOT NULL,
    predicate  TEXT NOT NULL,
    object     TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source     TEXT NOT NULL DEFAULT '',
    tags       TEXT NOT NULL DEFAULT '',
    UNIQUE (subject, predicate, object)
);

-- Procedural memory: monitor rules (trigger -> action), stored as JSON.
CREATE TABLE IF NOT EXISTS procedures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    trigger     TEXT NOT NULL,
    action      TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    cooldown_s  REAL NOT NULL DEFAULT 60.0,
    updated_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_ts    ON episodes(ts DESC);
CREATE INDEX IF NOT EXISTS idx_obs_ts         ON observations(ts DESC);
CREATE INDEX IF NOT EXISTS idx_inter_ts       ON interactions(ts DESC);
CREATE INDEX IF NOT EXISTS idx_facts_subject  ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_face_person    ON face_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_voice_person   ON voice_embeddings(person_id);
"""


class MemoryStore:
    """The SQLite connection plus the schema, shared by every memory type."""

    def __init__(self, path: str, archive_dir: Optional[str] = None) -> None:
        self.path = os.path.expanduser(path)
        self.archive_dir = os.path.expanduser(archive_dir) if archive_dir else None
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if self.archive_dir:
            os.makedirs(self.archive_dir, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection ---------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        # ``check_same_thread=False`` because the connection is *handed off*, not shared: the
        # monitor is built on the main thread but runs on its own thread (that is how
        # ``qwen25vl-agent web`` streams events while serving HTTP), so the connection is created
        # in one thread and used in another.  Ownership is still single-threaded -- the web
        # frontend opens its own short-lived connections for reads -- so nothing uses this one
        # concurrently.  Without the flag SQLite raises on the very first tick.
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL so the CLI can read memory while the agent writes; NORMAL sync because losing the
        # last second of a monitoring log on a power cut is acceptable, and FULL would fsync on
        # every frame.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('created_at', ?)",
            (str(time.time()),),
        )
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MemoryStore":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- helpers ------------------------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list:
        return list(self.connect().execute(sql, tuple(params)))

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        cur = self.connect().execute(sql, tuple(params))
        return cur.fetchone()

    def now(self) -> float:
        return time.time()

    # -- maintenance ---------------------------------------------------------------
    def delete_older_than(self, table: str, column: str, cutoff: float) -> int:
        """Delete rows older than ``cutoff``; used by the archive/prune step."""
        cur = self.execute(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))
        return cur.rowcount or 0

    def size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

    def stats(self) -> dict:
        """Row counts per table -- what ``qwen25vl-agent mem stats`` prints."""
        out: dict = {"db_bytes": self.size_bytes(), "schema_version": SCHEMA_VERSION}
        for table in ("persons", "face_embeddings", "voice_embeddings", "interactions",
                      "episodes", "observations", "facts", "procedures"):
            row = self.query_one(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(row["n"]) if row else 0
        return out

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def loads(text: Optional[str], default: Any = None) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return default
