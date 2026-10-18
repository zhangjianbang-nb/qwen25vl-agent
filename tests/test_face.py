"""Face detection and recognition: the pure matching helpers, plus a real model load.

The engine tests touch the two ONNX files through ``cv2.dnn``; they skip when the models are not
present, so a machine that has not run ``scripts/fetch_models.sh`` still gets a green suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from qwen25vl_agent.perception.face import FaceDetection, FaceEngine


def test_match_score_on_an_empty_gallery_is_none():
    assert FaceEngine.match_score(np.ones(4, dtype=np.float32), []) is None


def test_similarity_and_best_gallery_index():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    assert FaceEngine.similarity(a, a) == pytest.approx(1.0)
    assert FaceEngine.similarity(a, b) == pytest.approx(0.0)

    hit = FaceEngine.match_score(a, [b, a])
    assert hit is not None
    assert hit[0] == 1                      # index of the closer gallery entry
    assert hit[1] == pytest.approx(1.0)


def test_detection_area_and_dict():
    detection = FaceDetection(box=(1, 2, 3, 4), score=0.9,
                              embedding=np.zeros(128, dtype=np.float32))
    assert detection.area == 12
    assert detection.as_dict()["box"] == [1, 2, 3, 4]
    assert detection.as_dict()["score"] == 0.9
    assert "embedding_dim" not in detection.as_dict()
    assert detection.as_dict(with_embedding=True)["embedding_dim"] == 128


def test_models_present_reports_both_files(tmp_path):
    # The real models are gitignored (`scripts/fetch_models.sh` downloads them), so this test
    # uses stand-in files: the helper only answers "are both paths files", and a suite that
    # needed the 37 MB downloads would be red on a fresh clone and on CI.
    detector = tmp_path / "detector.onnx"
    recognizer = tmp_path / "recognizer.onnx"
    detector.write_bytes(b"stand-in")
    recognizer.write_bytes(b"stand-in")

    assert FaceEngine.models_present(str(detector), str(recognizer)) is True
    assert FaceEngine.models_present("/definitely/not/here.onnx", str(recognizer)) is False
    assert FaceEngine.models_present(str(detector), "/definitely/not/here.onnx") is False


def test_engine_loads_and_finds_nothing_on_a_blank_frame(cfg):
    engine = FaceEngine(cfg.detector_model_path, cfg.recognizer_model_path)
    if not engine.available():
        pytest.skip("face models are not present; run scripts/fetch_models.sh")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert engine.detect(blank) == []
    assert engine.analyze(blank) == []


def test_missing_models_make_available_false(cfg):
    engine = FaceEngine("/definitely/not/here.onnx", "/nor/this.onnx")
    assert engine.available() is False
