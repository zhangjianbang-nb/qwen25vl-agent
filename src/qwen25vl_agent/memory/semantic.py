"""Semantic memory: durable facts about the world.

This is where the agent keeps what it has *learned* rather than what it *saw* -- "the door is on
the left", "the lamp is usually on in the evening", "Ming works at this desk".  Facts are keyed
by (subject, predicate, object) and upserted, so hearing the same thing twice bumps confidence
and recency instead of appending a duplicate.  That keeps the table small enough to hand the
whole thing to the model as context.

Confidence is stored rather than inferred: a fact the user stated is 0.9, one the VL model
guessed from one frame is 0.4, and the agent's prompt can weight them accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .store import MemoryStore

__all__ = ["Fact", "SemanticMemory"]


@dataclass
class Fact:
    id: int
    ts: float
    updated_at: float
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    tags: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }

    def sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


def _row_to_fact(row) -> Fact:
    return Fact(
        id=int(row["id"]), ts=float(row["ts"]), updated_at=float(row["updated_at"]),
        subject=row["subject"], predicate=row["predicate"], object=row["object"],
        confidence=float(row["confidence"]), source=row["source"] or "", tags=row["tags"] or "",
    )


class SemanticMemory:
    """Keyed facts with confidence and provenance."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def remember(self, subject: str, predicate: str, object: str, *, confidence: float = 0.5,
                 source: str = "", tags: str = "", ts: Optional[float] = None) -> int:
        """Upsert a fact.  Re-learning a fact raises confidence to the max of old and new."""
        now = ts if ts is not None else time.time()
        subject, predicate, object = subject.strip(), predicate.strip(), object.strip()
        if not subject or not predicate or not object:
            raise ValueError("a fact needs a non-empty subject, predicate and object")

        existing = self.store.query_one(
            "SELECT id, confidence FROM facts WHERE subject = ? AND predicate = ? AND object = ?",
            (subject, predicate, object),
        )
        if existing is not None:
            self.store.execute(
                "UPDATE facts SET updated_at = ?, confidence = ?, source = ?, tags = ? WHERE id = ?",
                (now, max(float(existing["confidence"]), float(confidence)), source, tags,
                 int(existing["id"])),
            )
            return int(existing["id"])
        cur = self.store.execute(
            "INSERT INTO facts(ts, updated_at, subject, predicate, object, confidence, source, tags)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, now, subject, predicate, object, float(confidence), source, tags),
        )
        return int(cur.lastrowid)

    def get(self, subject: Optional[str] = None, predicate: Optional[str] = None,
            limit: int = 200) -> List[Fact]:
        sql, params = "SELECT * FROM facts WHERE 1=1", []
        if subject is not None:
            sql += " AND subject LIKE ?"
            params.append(f"%{subject}%")
        if predicate is not None:
            sql += " AND predicate = ?"
            params.append(predicate)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        return [_row_to_fact(r) for r in self.store.query(sql, params)]

    def about(self, subject: str, limit: int = 50) -> List[Fact]:
        return self.get(subject=subject, limit=limit)

    def search(self, text: str, limit: int = 50) -> List[Fact]:
        like = f"%{text}%"
        rows = self.store.query(
            "SELECT * FROM facts WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ?"
            " ORDER BY updated_at DESC LIMIT ?",
            (like, like, like, int(limit)),
        )
        return [_row_to_fact(r) for r in rows]

    def forget(self, fact_id: int) -> bool:
        cur = self.store.execute("DELETE FROM facts WHERE id = ?", (int(fact_id),))
        return bool(cur.rowcount)

    def count(self) -> int:
        row = self.store.query_one("SELECT COUNT(*) AS n FROM facts")
        return int(row["n"]) if row else 0
