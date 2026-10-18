"""CLI helpers worth testing without a camera, a microphone, or a running model server.

``enroll`` is the one command whose failure is easy to misread: a gallery built from samples that
are not the same person looks exactly like a broken detector at recognition time.  These tests
cover the consistency check and the ``--from-frames`` path, both with fakes.
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pytest

from qwen25vl_agent.cli import _face_consistency, cmd_enroll
from qwen25vl_agent.perception.face import FaceDetection


def _det(embedding) -> FaceDetection:
    return FaceDetection(box=(0, 0, 50, 50), score=0.9, embedding=np.asarray(embedding, np.float32))


# -- the enrolment quality check ---------------------------------------------------------------
def test_consistency_is_none_for_a_single_sample():
    """One sample cannot disagree with itself, so there is nothing to warn about."""
    assert _face_consistency([_det(np.ones(128))]) is None
    assert _face_consistency([]) is None


def test_consistency_is_one_for_identical_samples():
    dets = [_det(np.ones(128)) for _ in range(3)]
    assert _face_consistency(dets) == pytest.approx(1.0)


def test_consistency_flags_samples_that_are_not_the_same_face():
    """Two of three samples agreeing, one orthogonal: the median similarity must fall."""
    a = np.zeros(128, dtype=np.float32)
    a[0] = 1.0
    b = a.copy()
    c = np.zeros(128, dtype=np.float32)
    c[1] = 1.0
    # pairwise: (a,b)=1, (a,c)=0, (b,c)=0 -> median 0
    assert _face_consistency([_det(a), _det(b), _det(c)]) == pytest.approx(0.0)


# -- enroll --from-frames ----------------------------------------------------------------------
class FakeFaceEngine:
    """Stands in for the ONNX detector/recognizer, returning one face per frame."""

    def __init__(self, *args, **kwargs):
        self._vec = np.ones(128, dtype=np.float32)
        self.seen = 0

    def available(self) -> bool:
        return True

    def analyze(self, rgb):
        self.seen += 1
        return [FaceDetection(box=(10, 10, 60, 60), score=0.95, embedding=self._vec.copy())]


def _write_frames(frames_dir: str, count: int) -> None:
    from PIL import Image

    os.makedirs(frames_dir, exist_ok=True)
    for i in range(count):
        path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (i * 7 % 255, 20, 30)).save(buffer, format="JPEG")
        with open(path, "wb") as fh:
            fh.write(buffer.getvalue())


def _write_config(tmp_path) -> str:
    """A config file rooted in ``tmp_path``, so ``enroll`` never touches the real state dir."""
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "memory": {"path": str(tmp_path / "memory.db"),
                   "archive_dir": str(tmp_path / "archive")},
        "alerts": {"log_path": str(tmp_path / "alerts.log")},
        "audio": {"enabled": False},
    }), encoding="utf-8")
    return str(path)


def _enroll_args(cfg_path, name, from_frames):
    return argparse.Namespace(
        config=cfg_path, name=name, from_frames=from_frames, image=None,
        device=None, frames=8, seconds=15.0, audio_seconds=0.0, audio_device=None,
        notes=None, json=True, verbose=False, quiet=False,
    )


def test_enroll_from_frames_needs_no_camera(tmp_path, monkeypatch):
    """Frames the monitor already saved are a valid gallery -- and work while it holds the camera."""
    from qwen25vl_agent.config import load_config
    from qwen25vl_agent.memory import Memory

    config_path = _write_config(tmp_path)
    cfg = load_config(config_path)
    _write_frames(cfg.frames_dir, 3)
    monkeypatch.setattr("qwen25vl_agent.perception.face.FaceEngine", FakeFaceEngine)

    code = cmd_enroll(_enroll_args(config_path, "小明", 3))

    assert code == 0
    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        person = mem.person.ensure("小明")
        assert mem.person.embedding_counts(person.id)["faces"] == 3


def test_enroll_from_frames_fails_clearly_when_there_are_none(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr("qwen25vl_agent.perception.face.FaceEngine", FakeFaceEngine)
    assert cmd_enroll(_enroll_args(config_path, "小明", 5)) == 3
