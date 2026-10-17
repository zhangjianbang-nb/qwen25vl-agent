"""Face detection (YuNet) and recognition (SFace).

Both models are OpenCV Zoo ONNX files run through ``cv2.dnn``, which means face recognition
needs *no extra Python dependency* -- no onnxruntime, no insightface.  On this machine that
matters: the CUDA-visible VRAM is fully committed to the served Qwen2.5-VL, so anything that
wanted its own GPU runtime would have to fight for it.  These run on CPU, and a 640x480 frame
costs a few tens of milliseconds.

The two models are split on purpose, because they answer different questions:

* YuNet answers *"is there a face here?"* -- cheap, runs every tick.
* SFace answers *"whose face is it?"* -- a 128-d embedding compared against the person gallery.

``scripts/fetch_models.sh`` downloads both.  ``available()`` lets the rest of the agent run
without them (the VL model can still describe people) and ``doctor`` explain the gap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..vectors import cosine

__all__ = ["FaceDetection", "FaceEngine", "FaceUnavailable"]


class FaceUnavailable(RuntimeError):
    """OpenCV is missing, or one of the two ONNX models is not present/loadable."""


@dataclass
class FaceDetection:
    """One detected face.

    ``box`` is ``(x, y, w, h)`` in pixels of the frame it came from; ``embedding`` is filled in
    only when the recognizer ran (it is skipped for faces smaller than ``min_face_px``, where a
    128-d embedding is not trustworthy).
    """

    box: Tuple[int, int, int, int]
    score: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        return int(self.box[2]) * int(self.box[3])

    def as_dict(self, with_embedding: bool = False) -> dict:
        out = {"box": [int(v) for v in self.box], "score": round(float(self.score), 4)}
        if with_embedding and self.embedding is not None:
            out["embedding_dim"] = int(self.embedding.size)
        return out


@dataclass
class FaceEngine:
    """YuNet detector + SFace recognizer over one shared ``cv2.dnn`` pair."""

    detector_model: str
    recognizer_model: str
    detect_score: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 50
    min_face_px: int = 48

    _detector: object = field(default=None, repr=False)
    _recognizer: object = field(default=None, repr=False)
    _input_size: Tuple[int, int] = field(default=(0, 0), repr=False)

    # -- availability -------------------------------------------------------------
    @staticmethod
    def models_present(detector_model: str, recognizer_model: str) -> bool:
        return os.path.isfile(detector_model) and os.path.isfile(recognizer_model)

    def available(self) -> bool:
        """True when OpenCV and both model files are usable."""
        try:
            self._ensure()
            return True
        except FaceUnavailable:
            return False

    def _ensure(self) -> None:
        if self._detector is not None and self._recognizer is not None:
            return
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise FaceUnavailable("face recognition needs OpenCV (opencv-python-headless)") from exc

        for path, kind in ((self.detector_model, "detector"), (self.recognizer_model, "recognizer")):
            if not os.path.isfile(path):
                raise FaceUnavailable(
                    f"{kind} model not found at {path}; run scripts/fetch_models.sh"
                )

        try:
            # YuNet's input size is set per-frame in `detect`; the placeholder here is only so
            # create() has something valid to bind.
            self._detector = cv2.FaceDetectorYN.create(
                self.detector_model, "", (320, 320),
                self.detect_score, self.nms_threshold, self.top_k,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(self.recognizer_model, "")
        except Exception as exc:  # cv2 raises a bare Exception on a corrupt model
            raise FaceUnavailable(f"could not load face models: {exc}") from exc

    # -- detection -----------------------------------------------------------------
    def detect(self, rgb: np.ndarray) -> List[FaceDetection]:
        """Detect faces in an RGB frame (BGR conversion happens here, once)."""
        import cv2

        self._ensure()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if (w, h) != self._input_size:
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, raw = self._detector.detect(bgr)
        if raw is None:
            return []

        faces: List[FaceDetection] = []
        for row in raw:
            x, y, fw, fh = (float(v) for v in row[:4])
            score = float(row[-1])
            if fw < self.min_face_px or fh < self.min_face_px:
                continue
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            faces.append(FaceDetection(
                box=(int(round(x)), int(round(y)), int(round(fw)), int(round(fh))),
                score=score,
                landmarks=landmarks,
            ))
        return faces

    # -- recognition ---------------------------------------------------------------
    def embed(self, rgb: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """128-d SFace embedding for one detection, aligned by the recognizer's own crop.

        ``alignCrop`` needs the raw 15-column YuNet row (it uses the 5 landmarks to warp the
        face upright), so the row is rebuilt from the detection rather than re-detected.
        """
        import cv2

        self._ensure()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        row = np.zeros(15, dtype=np.float32)
        row[:4] = detection.box
        if detection.landmarks is not None:
            row[4:14] = detection.landmarks.reshape(-1)
        row[14] = detection.score

        try:
            aligned = self._recognizer.alignCrop(bgr, row)
            feature = self._recognizer.feature(aligned)
        except Exception:  # a degenerate box (face at the frame edge) can make alignCrop fail
            return None
        if feature is None:
            return None
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    def analyze(self, rgb: np.ndarray) -> List[FaceDetection]:
        """Detect every face and embed the ones large enough to trust."""
        faces = self.detect(rgb)
        for face in faces:
            if min(face.box[2], face.box[3]) >= self.min_face_px:
                face.embedding = self.embed(rgb, face)
        return faces

    # -- matching -------------------------------------------------------------------
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return cosine(a, b)

    @staticmethod
    def match_score(query: np.ndarray, gallery: Sequence[np.ndarray]) -> Optional[Tuple[int, float]]:
        """Index of the closest gallery embedding, or ``None`` when the gallery is empty."""
        if not gallery:
            return None
        best_i, best_s = 0, -1.0
        for i, vec in enumerate(gallery):
            s = cosine(query, vec)
            if s > best_s:
                best_i, best_s = i, s
        return best_i, best_s
