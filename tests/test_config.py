"""Config loading: defaults, overrides, and the strict unknown-key check."""

from __future__ import annotations

import os

import pytest

from qwen25vl_agent.config import VADConfig, Config, load_config


def test_committed_defaults_load():
    cfg = load_config()
    assert cfg.llm.url == "http://127.0.0.1:8081"
    assert cfg.camera.width == 640
    assert cfg.audio.vad.start_db == -32.0
    assert cfg.identity.mode == "prefer_face"
    assert cfg.memory.working_size == 40
    assert cfg.monitor.describe_interval == 12.0


def test_dataclass_defaults_match_yaml():
    """The dataclass defaults are the fallback when a section is absent, so they must agree."""
    cfg = Config()
    assert cfg.llm.max_vision_tokens == 128
    assert cfg.face.match_threshold == 0.363
    assert cfg.speaker.threshold == 0.65


def test_unknown_key_is_rejected(tmp_path):
    """A misspelled key must be an error, not a silent fallback to the default."""
    path = tmp_path / "bad.yaml"
    path.write_text("camera:\n  devise: 0\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(str(path))
    assert "devise" in str(excinfo.value)


def test_override_merges_and_preserves_siblings():
    cfg = load_config(overrides={"monitor": {"tick_interval": 1.25}})
    assert cfg.monitor.tick_interval == 1.25
    # The other fields in the same section are untouched.
    assert cfg.monitor.describe_interval == 12.0
    assert cfg.monitor.save_frames is True


def test_partial_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("audio:\n  vad:\n    start_db: -20.0\n", encoding="utf-8")
    cfg = load_config(str(path))
    assert isinstance(cfg.audio.vad, VADConfig)
    assert cfg.audio.vad.start_db == -20.0
    # A section absent from the file keeps its dataclass default.
    assert cfg.llm.url == "http://127.0.0.1:8081"


def test_paths_are_expanded():
    cfg = load_config()
    home = os.path.expanduser("~")
    assert cfg.db_path.startswith(home)
    assert cfg.archive_dir.startswith(home)
    assert cfg.frames_dir.endswith("frames")


def test_face_model_paths_resolve_to_absolute():
    cfg = load_config()
    assert os.path.isabs(cfg.detector_model_path)
    assert os.path.isabs(cfg.recognizer_model_path)
    assert cfg.detector_model_path.endswith("face_detection_yunet_2023mar.onnx")
    assert cfg.recognizer_model_path.endswith("face_recognition_sface_2021dec.onnx")
