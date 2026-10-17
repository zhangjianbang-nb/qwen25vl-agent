"""Vector helpers: packing, normalization, cosine, and gallery matching."""

from __future__ import annotations

import numpy as np
import pytest

from qwen25vl_agent.vectors import best_match, cosine, cosine_matrix, normalize, pack, unpack


def test_pack_unpack_roundtrip():
    vec = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    blob = pack(vec)
    assert isinstance(blob, bytes)
    assert blob == vec.tobytes()
    np.testing.assert_allclose(unpack(blob), vec)


def test_unpack_dim_check():
    blob = pack(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        unpack(blob, dim=4)
    assert unpack(blob, dim=3).size == 3


def test_normalize_unit_norm():
    vec = normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


def test_normalize_zero_vector_is_left_alone():
    vec = normalize(np.zeros(4, dtype=np.float32))
    assert float(np.linalg.norm(vec)) == 0.0


def test_cosine_identical_and_orthogonal():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)


def test_cosine_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_cosine_matrix_scores_every_row():
    query = np.array([1.0, 0.0], dtype=np.float32)
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = cosine_matrix(query, gallery)
    assert scores.size == 2
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_cosine_matrix_empty_gallery():
    assert cosine_matrix(np.ones(2, dtype=np.float32), np.zeros((0, 0), dtype=np.float32)).size == 0


def test_best_match_above_and_below_threshold():
    query = np.array([1.0, 0.0], dtype=np.float32)
    gallery = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert best_match(query, gallery, [10, 20], threshold=0.5) == (20, pytest.approx(1.0))
    # Nothing clears a threshold above the best score.
    assert best_match(query, gallery, [10, 20], threshold=0.99) == (20, pytest.approx(1.0))
    assert best_match(query, gallery, [10, 20], threshold=1.5) is None


def test_best_match_empty_gallery_is_none():
    assert best_match(np.ones(2, dtype=np.float32), np.zeros((0, 0), dtype=np.float32),
                      [], threshold=0.3) is None
