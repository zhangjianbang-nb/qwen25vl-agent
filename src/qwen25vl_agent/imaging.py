"""Image helpers shared by perception and the LLM client.

Kept separate from ``perception.video`` so that anything holding an RGB array (a camera frame,
a cropped face, a test fixture) can be encoded or downscaled without importing OpenCV.

Two decisions are baked in here and are worth stating once:

* Frames are JPEG, not PNG.  A 640x480 camera frame is photographic: ~44 KB as JPEG against
  ~600 KB as PNG, and the vision tower downsamples it anyway.
* Downscaling is done client-side because vision tokens scale with *area*.  Halving the width
  cuts the token count roughly fourfold, and unlike the server's ``--vision-tier`` flag it
  needs no restart.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np

__all__ = ["downscale_rgb", "rgb_to_jpeg", "jpeg_data_url", "decode_jpeg", "rgb_to_data_url"]


def downscale_rgb(rgb: np.ndarray, width: int) -> np.ndarray:
    """Shrink an RGB array to ``width`` columns, preserving aspect ratio.

    ``width <= 0`` (or a width no smaller than the current one) returns the input unchanged.
    """
    if width <= 0:
        return rgb
    from PIL import Image

    height, cols = rgb.shape[:2]
    if width >= cols:
        return rgb
    return np.asarray(
        Image.fromarray(rgb).resize((width, max(1, round(height * width / cols))), Image.BILINEAR)
    )


def rgb_to_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    """JPEG-encode an RGB array."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_data_url(jpeg: bytes) -> str:
    """Wrap JPEG bytes in the ``data:`` URL the server's image parser accepts."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def rgb_to_data_url(rgb: np.ndarray, quality: int = 85, width: int = 0) -> str:
    return jpeg_data_url(rgb_to_jpeg(downscale_rgb(rgb, width), quality))


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB array (used by tests and by saved-frame inspection)."""
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))


def crop_rgb(rgb: np.ndarray, box, pad: float = 0.0) -> Optional[np.ndarray]:
    """Crop ``box`` = (x, y, w, h) out of an RGB array, clamped to the image.

    Returns ``None`` for a degenerate crop so callers can skip it rather than feed a 0-pixel
    image to an embedder.
    """
    x, y, w, h = (float(v) for v in box)
    if pad:
        x, y, w, h = x - pad * w, y - pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad)
    h_img, w_img = rgb.shape[:2]
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(w_img, int(round(x + w)))
    y1 = min(h_img, int(round(y + h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.ascontiguousarray(rgb[y0:y1, x0:x1])
