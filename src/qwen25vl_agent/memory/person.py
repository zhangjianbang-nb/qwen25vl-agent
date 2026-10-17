"""Person memory: identity from face *and* voiceprint.

A person record holds two independent biometric galleries -- SFace 128-d face embeddings and
CAM++ 192-d voiceprints -- and identity is decided by fusing them.  The reason to carry both is
that neither modality is always available: a face needs the person visible and facing the
camera, a voice needs them to speak.  Over a full day, requiring only one of the two keeps
identity working far more of the time than either alone.

``identity.mode`` controls the fusion:

* ``prefer_face`` -- use the face when a face embedding exists, else fall back to the voice;
* ``any``         -- accept a match from either modality;
* ``both``        -- require both to agree (strictest; best when both are reliably present).

The gallery is cached as a stacked matrix so a match is one numpy dot product per modality,
and the cache is invalidated whenever an embedding is added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..vectors import best_match, pack, unpack
from .store import MemoryStore

__all__ = ["Person", "IdentityMatch", "PersonMemory"]


@dataclass
class Person:
    id: int
    name: str
    created_at: float
    first_seen: Optional[float]
    last_seen: Optional[float]
    seen_count: int
    notes: str
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "seen_count": self.seen_count, "notes": self.notes,
        }


@dataclass
class IdentityMatch:
    """The result of asking "who is this?"."""

    known: bool
    name: str = "unknown"
    person_id: Optional[int] = None
    score: float = 0.0
    modality: str = "none"          # 'face' | 'voice' | 'both' | 'none'
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "known": self.known, "name": self.name, "person_id": self.person_id,
            "score": round(self.score, 4), "modality": self.modality,
            **({"detail": self.detail} if self.detail else {}),
        }


def _row_to_person(row) -> Person:
    return Person(
        id=int(row["id"]), name=row["name"], created_at=float(row["created_at"]),
        first_seen=row["first_seen"], last_seen=row["last_seen"],
        seen_count=int(row["seen_count"] or 0), notes=row["notes"] or "",
        meta=MemoryStore.loads(row["meta"], {}),
    )


class PersonMemory:
    """Person records plus the face/voice galleries used to match them."""

    def __init__(self, store: MemoryStore, *, face_threshold: float = 0.363,
                 voice_threshold: float = 0.65, identity_mode: str = "prefer_face") -> None:
        self.store = store
        self.face_threshold = float(face_threshold)
        self.voice_threshold = float(voice_threshold)
        self.identity_mode = identity_mode
        self._face_cache: Optional[Tuple[Tuple[int, ...], np.ndarray]] = None
        self._voice_cache: Optional[Tuple[Tuple[int, ...], np.ndarray]] = None

    # -- records ------------------------------------------------------------------
    def ensure(self, name: str, *, notes: str = "", ts: Optional[float] = None) -> Person:
        """Return the person with this name, creating the record if needed."""
        name = name.strip()
        if not name:
            raise ValueError("a person needs a name")
        row = self.store.query_one("SELECT * FROM persons WHERE name = ?", (name,))
        if row is not None:
            return _row_to_person(row)
        now = ts if ts is not None else time.time()
        cur = self.store.execute(
            "INSERT INTO persons(name, created_at, notes) VALUES (?, ?, ?)", (name, now, notes),
        )
        return self.get(int(cur.lastrowid))  # type: ignore[return-value]

    def get(self, key) -> Optional[Person]:
        if isinstance(key, int):
            row = self.store.query_one("SELECT * FROM persons WHERE id = ?", (key,))
        else:
            row = self.store.query_one("SELECT * FROM persons WHERE name = ?", (str(key),))
        return _row_to_person(row) if row is not None else None

    def all(self) -> List[Person]:
        return [_row_to_person(r) for r in self.store.query("SELECT * FROM persons ORDER BY name")]

    def rename(self, person_id: int, name: str) -> None:
        self.store.execute("UPDATE persons SET name = ? WHERE id = ?", (name.strip(), int(person_id)))

    def set_notes(self, person_id: int, notes: str) -> None:
        self.store.execute("UPDATE persons SET notes = ? WHERE id = ?", (notes, int(person_id)))

    def set_meta(self, person_id: int, meta: dict) -> None:
        self.store.execute("UPDATE persons SET meta = ? WHERE id = ?",
                           (MemoryStore.dumps(meta), int(person_id)))

    def delete(self, person_id: int) -> None:
        self.store.execute("DELETE FROM persons WHERE id = ?", (int(person_id),))
        self._invalidate()

    # -- embeddings -----------------------------------------------------------------
    def add_face(self, person: Person, vec: np.ndarray, *, quality: float = 0.0,
                 ts: Optional[float] = None) -> int:
        return self._add_embedding("face_embeddings", person.id, vec, quality, ts)

    def add_voice(self, person: Person, vec: np.ndarray, *, quality: float = 0.0,
                  ts: Optional[float] = None) -> int:
        return self._add_embedding("voice_embeddings", person.id, vec, quality, ts)

    def _add_embedding(self, table: str, person_id: int, vec: np.ndarray,
                       quality: float, ts: Optional[float]) -> int:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        cur = self.store.execute(
            f"INSERT INTO {table}(person_id, ts, vec, dim, quality) VALUES (?, ?, ?, ?, ?)",
            (int(person_id), ts if ts is not None else time.time(),
             pack(vec), int(vec.size), float(quality)),
        )
        self._invalidate()
        return int(cur.lastrowid)

    def _load_gallery(self, table: str) -> Tuple[Tuple[int, ...], np.ndarray]:
        rows = self.store.query(f"SELECT person_id, vec FROM {table} ORDER BY id")
        if not rows:
            return (), np.zeros((0, 0), dtype=np.float32)
        ids = tuple(int(r["person_id"]) for r in rows)
        vecs = [unpack(r["vec"]) for r in rows]
        dim = max(v.size for v in vecs)
        matrix = np.zeros((len(vecs), dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            matrix[i, : v.size] = v
        return ids, matrix

    def _gallery(self, modality: str) -> Tuple[Tuple[int, ...], np.ndarray]:
        cache = self._face_cache if modality == "face" else self._voice_cache
        if cache is None:
            cache = self._load_gallery("face_embeddings" if modality == "face" else "voice_embeddings")
            if modality == "face":
                self._face_cache = cache
            else:
                self._voice_cache = cache
        return cache

    def _invalidate(self) -> None:
        self._face_cache = None
        self._voice_cache = None

    # -- matching -------------------------------------------------------------------
    def match_face(self, vec: np.ndarray) -> IdentityMatch:
        ids, matrix = self._gallery("face")
        hit = best_match(vec, matrix, ids, self.face_threshold)
        if hit is None:
            return IdentityMatch(known=False, modality="face")
        person_id, score = hit
        person = self.get(person_id)
        if person is None:
            return IdentityMatch(known=False, modality="face")
        return IdentityMatch(known=True, name=person.name, person_id=person.id,
                             score=score, modality="face")

    def match_voice(self, vec: np.ndarray) -> IdentityMatch:
        ids, matrix = self._gallery("voice")
        hit = best_match(vec, matrix, ids, self.voice_threshold)
        if hit is None:
            return IdentityMatch(known=False, modality="voice")
        person_id, score = hit
        person = self.get(person_id)
        if person is None:
            return IdentityMatch(known=False, modality="voice")
        return IdentityMatch(known=True, name=person.name, person_id=person.id,
                             score=score, modality="voice")

    def fuse(self, face_vec: Optional[np.ndarray] = None,
             voice_vec: Optional[np.ndarray] = None) -> IdentityMatch:
        """Combine the two modalities according to ``identity.mode``."""
        face = self.match_face(face_vec) if face_vec is not None else None
        voice = self.match_voice(voice_vec) if voice_vec is not None else None
        detail = {}
        if face is not None:
            detail["face_score"] = round(face.score, 4)
            detail["face_known"] = face.known
        if voice is not None:
            detail["voice_score"] = round(voice.score, 4)
            detail["voice_known"] = voice.known

        mode = self.identity_mode
        if mode == "both":
            if face is not None and voice is not None and face.known and voice.known \
                    and face.person_id == voice.person_id:
                return IdentityMatch(True, face.name, face.person_id,
                                     (face.score + voice.score) / 2, "both", detail)
            return IdentityMatch(False, modality="both", detail=detail)

        if mode == "any":
            candidates = [m for m in (face, voice) if m is not None and m.known]
            if candidates:
                best = max(candidates, key=lambda m: m.score)
                return IdentityMatch(True, best.name, best.person_id, best.score,
                                     "face" if best is face else "voice", detail)
            return IdentityMatch(False, modality="none", detail=detail)

        # prefer_face (default)
        if face is not None and face.known:
            return IdentityMatch(True, face.name, face.person_id, face.score, "face", detail)
        if voice is not None and voice.known:
            return IdentityMatch(True, voice.name, voice.person_id, voice.score, "voice", detail)
        return IdentityMatch(False, modality="none", detail=detail)

    # -- activity -------------------------------------------------------------------
    def touch(self, person_id: int, *, ts: Optional[float] = None) -> None:
        """Record a sighting: bump last_seen/seen_count."""
        now = ts if ts is not None else time.time()
        self.store.execute(
            "UPDATE persons SET last_seen = ?, seen_count = seen_count + 1,"
            " first_seen = COALESCE(first_seen, ?) WHERE id = ?",
            (now, now, int(person_id)),
        )

    def note(self, person_id: int, text: str, *, kind: str = "note",
             ts: Optional[float] = None) -> int:
        cur = self.store.execute(
            "INSERT INTO interactions(ts, person_id, kind, text) VALUES (?, ?, ?, ?)",
            (ts if ts is not None else time.time(), int(person_id), kind, text),
        )
        return int(cur.lastrowid)

    def history(self, person_id: int, limit: int = 50) -> List[dict]:
        rows = self.store.query(
            "SELECT ts, kind, text FROM interactions WHERE person_id = ? ORDER BY ts DESC LIMIT ?",
            (int(person_id), int(limit)),
        )
        return [{"ts": float(r["ts"]), "kind": r["kind"], "text": r["text"]} for r in rows]

    def embedding_counts(self, person_id: int) -> Dict[str, int]:
        out = {}
        for table, key in (("face_embeddings", "faces"), ("voice_embeddings", "voices")):
            row = self.store.query_one(
                f"SELECT COUNT(*) AS n FROM {table} WHERE person_id = ?", (int(person_id),))
            out[key] = int(row["n"]) if row else 0
        return out
