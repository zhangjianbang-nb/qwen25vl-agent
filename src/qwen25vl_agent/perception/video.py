"""Video sampling from a V4L2 camera.

The one thing that makes this more than a ``cv2.VideoCapture`` wrapper is buffer draining.  A
30 fps camera behind a ~0.5 s loop will otherwise hand back a frame from hundreds of
milliseconds ago, which turns "live monitoring" into "describing the past".  So the buffer is
drained (``grab`` twice) before every ``retrieve``, and ``CAP_PROP_BUFFERSIZE`` is set to 1.

The source also survives a camera that drops out: a 24/7 agent cannot exit because USB
hiccuped once, so ``grab`` re-opens with a backoff and returns ``None`` until it recovers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Frame", "CameraSource", "CameraUnavailable"]


class CameraUnavailable(RuntimeError):
    """OpenCV is missing, or the device cannot be opened."""


@dataclass
class Frame:
    """One sampled frame plus the bookkeeping the agent needs downstream."""

    rgb: np.ndarray
    ts: float                      # wall-clock seconds (time.time()), so it aligns with logs
    index: int                     # monotonic per-source counter
    camera_ts: float = 0.0         # monotonic capture time, for latency accounting


@dataclass
class CameraSource:
    """A reconnecting V4L2 camera."""

    device: int = 0
    width: int = 640
    height: int = 480
    reopen_backoff_s: float = 1.0
    max_backoff_s: float = 30.0

    _cap: object = field(default=None, repr=False)
    _index: int = 0
    _failures: int = 0
    _next_retry: float = 0.0

    # -- lifecycle ----------------------------------------------------------------
    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise CameraUnavailable(
                "the camera source needs OpenCV: pip install 'qwen25vl-agent' (opencv-python-headless)"
            ) from exc

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailable(f"could not open /dev/video{self.device}")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Without this the driver queues frames and `retrieve` returns stale ones.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._failures = 0
        self._next_retry = 0.0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sampling -----------------------------------------------------------------
    def _reopen_if_due(self) -> None:
        """Re-open after a failure, but not more often than the backoff allows."""
        if self._cap is not None:
            return
        now = time.monotonic()
        if now < self._next_retry:
            return
        try:
            self.open()
        except CameraUnavailable:
            self._failures += 1
            backoff = min(self.max_backoff_s, self.reopen_backoff_s * (2 ** min(self._failures, 5)))
            self._next_retry = now + backoff

    def grab(self) -> Optional[Frame]:
        """Newest available frame, or ``None`` if the camera is momentarily unusable."""
        import cv2

        self._reopen_if_due()
        if self._cap is None:
            return None

        # Drain whatever is already queued, then take the fresh frame.
        for _ in range(2):
            if not self._cap.grab():
                self._mark_failed()
                return None
        ok, frame = self._cap.retrieve()
        if not ok or frame is None:
            self._mark_failed()
            return None

        self._failures = 0
        self._index += 1
        return Frame(
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            ts=time.time(),
            index=self._index,
            camera_ts=time.monotonic(),
        )

    def _mark_failed(self) -> None:
        """A read failure usually means the device went away; drop it and let backoff re-open."""
        self._failures += 1
        self.close()
        self._next_retry = time.monotonic() + min(
            self.max_backoff_s, self.reopen_backoff_s * (2 ** min(self._failures, 5))
        )
