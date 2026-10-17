"""Image helpers: JPEG encoding, downscaling and cropping.

These need only PIL and numpy, so they run without a camera or the face models.
"""

from __future__ import annotations

import numpy as np

from qwen25vl_agent.imaging import (
    crop_rgb,
    decode_jpeg,
    downscale_rgb,
    jpeg_data_url,
    rgb_to_data_url,
    rgb_to_jpeg,
)


def test_jpeg_roundtrip_shape_and_magic():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:20, 5:15] = 255
    jpeg = rgb_to_jpeg(rgb, 85)
    assert jpeg[:2] == b"\xff\xd8"          # JPEG SOI marker
    assert decode_jpeg(jpeg).shape == (48, 64, 3)


def test_data_url_prefix():
    url = jpeg_data_url(rgb_to_jpeg(np.zeros((8, 8, 3), dtype=np.uint8)))
    assert url.startswith("data:image/jpeg;base64,")


def test_rgb_to_data_url_combines_downscale_and_encode():
    url = rgb_to_data_url(np.zeros((48, 64, 3), dtype=np.uint8), width=32)
    assert url.startswith("data:image/jpeg;base64,")
    # 64 -> 32 columns means the decoded image is half size in both axes.
    assert decode_jpeg(rgb_to_jpeg(downscale_rgb(np.zeros((48, 64, 3), dtype=np.uint8), 32))).shape \
        == (24, 32, 3)


def test_downscale_preserves_aspect_and_ignores_bad_widths():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    assert downscale_rgb(rgb, 32).shape == (24, 32, 3)
    # width <= 0 keeps the original, and a width no smaller than the frame is a no-op too.
    assert downscale_rgb(rgb, 0).shape == (48, 64, 3)
    assert downscale_rgb(rgb, 999).shape == (48, 64, 3)


def test_crop_inside_the_frame():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    out = crop_rgb(rgb, (10, 20, 30, 40))
    assert out is not None
    assert out.shape == (40, 30, 3)


def test_crop_is_clamped_at_the_edge():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    out = crop_rgb(rgb, (90, 90, 30, 30))
    assert out is not None
    assert out.shape == (10, 10, 3)


def test_degenerate_crop_is_none():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_rgb(rgb, (10, 20, 0, 0)) is None
    assert crop_rgb(rgb, (10, 20, 1, 1)) is None
