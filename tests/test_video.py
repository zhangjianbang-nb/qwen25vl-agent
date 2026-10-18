"""Frame rotation.

A camera mounted upside-down opens and streams normally -- it just returns inverted frames, and
an inverted face is one YuNet does not detect.  That makes rotation a correctness fix rather than
a cosmetic one, so the geometry is pinned here.
"""

from __future__ import annotations

import numpy as np

from qwen25vl_agent.perception.video import rotate_frame


def test_zero_degrees_is_a_no_op():
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    assert rotate_frame(img, 0) is img


def test_180_degrees_puts_the_bottom_right_corner_top_left():
    img = np.array([[[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]]], dtype=np.uint8)
    rotated = rotate_frame(img, 180)
    assert rotated.shape == img.shape
    assert rotated[0, 0].tolist() == [4, 4, 4]
    assert rotated[1, 1].tolist() == [1, 1, 1]


def test_90_degrees_is_clockwise_and_transposes_the_shape():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rotated = rotate_frame(img, 90)
    assert rotated.shape == (3, 2, 3)
    # A clockwise quarter turn moves the original bottom-left corner to the top-left.
    assert rotated[0, 0].tolist() == img[1, 0].tolist()


def test_270_degrees_is_the_inverse_of_90():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(rotate_frame(rotate_frame(img, 90), 270), img)


def test_result_is_contiguous():
    """OpenCV and YuNet are handed this array directly and dislike non-contiguous views."""
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert rotate_frame(img, 90).flags["C_CONTIGUOUS"]


def test_degrees_are_normalised_modulo_360():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(rotate_frame(img, 450), rotate_frame(img, 90))
