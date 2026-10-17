"""Small vector helpers.

Embeddings (128-d SFace face features, 192-d CAM++ voiceprints) live in SQLite as float32
blobs, and matching is a cosine similarity over a small gallery.  That is deliberately not a
vector database: a household-scale gallery is tens of rows, so a numpy dot product over a
stacked matrix is exact, dependency-free and fast enough to run every tick.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

__all__ = ["pack", "unpack", "normalize", "cosine", "cosine_matrix", "best_match"]


def pack(vec: np.ndarray) -> bytes:
    """Serialize a vector as float32 bytes for storage."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes, dim: Optional[int] = None) -> np.ndarray:
    """Inverse of :func:`pack`."""
    vec = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and vec.size != dim:
        raise ValueError(f"expected {dim} values, blob holds {vec.size}")
    return vec.copy()


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize so a dot product *is* the cosine similarity."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = normalize(a), normalize(b)
    if a.size != b.size:
        raise ValueError(f"dimension mismatch: {a.size} vs {b.size}")
    return float(np.dot(a, b))


def cosine_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine of one query against every row of ``gallery`` (shape ``(n, dim)``)."""
    if gallery.size == 0:
        return np.zeros(0, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    q = normalize(query)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return g @ q


def best_match(query: np.ndarray, gallery: np.ndarray, ids: Iterable[int],
               threshold: float) -> Optional[Tuple[int, float]]:
    """Highest-scoring gallery row above ``threshold``, or ``None``.

    ``None`` means "unknown person", which is a first-class outcome here: an unknown face is
    exactly what the procedural rules are usually watching for.
    """
    ids = list(ids)
    scores = cosine_matrix(query, gallery)
    if scores.size == 0 or scores.size != len(ids):
        return None
    idx = int(np.argmax(scores))
    if float(scores[idx]) < threshold:
        return None
    return ids[idx], float(scores[idx])
