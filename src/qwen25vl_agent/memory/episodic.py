"""Episodic memory: what happened, when.

An episode is a *summarized* event -- "Ming sat down at the desk", "an unknown person appeared
at the door" -- as opposed to the raw per-tick observations, which are kept separately and
pruned much sooner.  The distinction is what makes the memory usable months in: episodes are
small, human-readable, and worth keeping for the configured 30 days, while the raw log is only
useful for the last few days of debugging.

Every episode can point at the frame that produced it (``frame_path``), so a summary in the
database can be checked against what the camera actually saw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from .store import MemoryStore

__all__ = ["Episode", "EpisodicMemory"]


@dataclass
class Episode:
    id: int
    ts: float
    t_end: Optional[float]
    kind: str
    summary: str
    detail: dict
    person_ids: List[int]
    importance: float
    frame_path: Optional[str]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "t_end": self.t_end,
            "kind": self.kind,
            "summary": self.summary,
            "person_ids": self.person_ids,
            "importance": round(self.importance, 3),
            "frame_path": self.frame_path,
        }


def _row_to_episode(row) -> Episode:
    return Episode(
        id=int(row["id"]),
        ts=float(row["ts"]),
        t_end=row["t_end"],
        kind=row["kind"],
        summary=row["summary"],
        detail=MemoryStore.loads(row["detail"], {}),
        person_ids=MemoryStore.loads(row["person_ids"], []),
        importance=float(row["importance"] or 0.0),
        frame_path=row["frame_path"],
    )


class EpisodicMemory:
    """The episode log, plus the raw observation log it is distilled from."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # -- writes -------------------------------------------------------------------
    def record(self, kind: str, summary: str, *, detail: Optional[dict] = None,
               person_ids: Optional[List[int]] = None, importance: float = 0.0,
               frame_path: Optional[str] = None, ts: Optional[float] = None,
               t_end: Optional[float] = None) -> int:
        ts = ts if ts is not None else time.time()
        cur = self.store.execute(
            "INSERT INTO episodes(ts, t_end, kind, summary, detail, person_ids, importance, frame_path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, t_end, kind, summary, MemoryStore.dumps(detail or {}),
             MemoryStore.dumps(person_ids or []), float(importance), frame_path),
        )
        return int(cur.lastrowid)

    def observe(self, kind: str, text: str = "", *, payload: Optional[dict] = None,
                person_ids: Optional[List[int]] = None, frame_path: Optional[str] = None,
                ts: Optional[float] = None) -> int:
        """Append one raw perception row (the archive's backbone)."""
        ts = ts if ts is not None else time.time()
        cur = self.store.execute(
            "INSERT INTO observations(ts, kind, text, person_ids, payload, frame_path)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (ts, kind, text, MemoryStore.dumps(person_ids or []),
             MemoryStore.dumps(payload or {}), frame_path),
        )
        return int(cur.lastrowid)

    # -- reads ---------------------------------------------------------------------
    def recent(self, limit: int = 20, *, since: Optional[float] = None,
               kind: Optional[str] = None, min_importance: float = 0.0) -> List[Episode]:
        sql = "SELECT * FROM episodes WHERE importance >= ?"
        params: list = [min_importance]
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        return [_row_to_episode(r) for r in self.store.query(sql, params)]

    def search(self, text: str, limit: int = 20) -> List[Episode]:
        like = f"%{text}%"
        rows = self.store.query(
            "SELECT * FROM episodes WHERE summary LIKE ? OR detail LIKE ? ORDER BY ts DESC LIMIT ?",
            (like, like, int(limit)),
        )
        return [_row_to_episode(r) for r in rows]

    def for_person(self, person_id: int, limit: int = 20) -> List[Episode]:
        # person_ids is a JSON list; a substring match on the id is sufficient at this scale and
        # avoids a join table for a field that is written once per episode.
        like = f'%{int(person_id)}%'
        rows = self.store.query(
            "SELECT * FROM episodes WHERE person_ids LIKE ? ORDER BY ts DESC LIMIT ?",
            (like, int(limit)),
        )
        return [_row_to_episode(r) for r in rows]

    def count(self) -> int:
        row = self.store.query_one("SELECT COUNT(*) AS n FROM episodes")
        return int(row["n"]) if row else 0
