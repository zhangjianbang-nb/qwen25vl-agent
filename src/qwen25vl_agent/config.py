"""Typed configuration.

The committed defaults live in ``configs/default.yaml``.  This module turns that YAML (or any
file with the same shape) into dataclasses, so every module reads attributes rather than
stringly-typed dict keys, and a typo in the config fails at load time instead of silently
falling back to a default.

Unknown keys are rejected rather than ignored: a misspelled ``describe_intervall`` should be
an error, because the alternative is an agent that quietly runs with the wrong cadence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional, get_type_hints

__all__ = [
    "Config", "LLMConfig", "CameraConfig", "AudioConfig", "VADConfig", "ASRConfig",
    "SpeakerConfig", "FaceConfig", "IdentityConfig", "MonitorConfig", "MemoryConfig",
    "AlertConfig", "load_config",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "default.yaml")


def _expand(path: Optional[str]) -> Optional[str]:
    """Expand ``~`` so config paths are portable across shells that do not set HOME the same."""
    return None if path is None else os.path.expanduser(os.path.expandvars(path))


def _build(cls, data: Dict[str, Any]):
    """Instantiate a dataclass from a mapping, rejecting unknown keys and coercing types."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    # `from __future__ import annotations` turns every annotation into a string, so field types
    # must be resolved through get_type_hints before they can be tested with is_dataclass.
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {', '.join(unknown)} "
            f"(known: {', '.join(sorted(known))})"
        )
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        # A nested dataclass field arrives as a mapping.
        if ftype is not None and is_dataclass(ftype) and isinstance(value, dict):
            value = _build(ftype, value)
        kwargs[name] = value
    return cls(**kwargs)


@dataclass
class LLMConfig:
    """Where the reasoning engine lives.

    ``url`` is a running ``qwen25vl-lite serve``.  ``max_vision_tokens`` and ``width`` are the
    two levers that actually change a frame's cost: the server reads the image budget from the
    *request body*, and its own ``--vision-tier`` flag only drives FastV pruning.  Dropping to
    ~128 vision tokens holds prefill near its ~130 ms floor while still describing a scene.
    """

    url: str = "http://127.0.0.1:8081"
    model: str = "Qwen2.5-VL-3B-Instruct"
    max_tokens: int = 256
    temperature: float = 0.0
    timeout: float = 120.0
    max_vision_tokens: Optional[int] = 128
    width: int = 0                      # client-side downscale; 0 keeps the captured width
    text_url: Optional[str] = None      # optional text-only endpoint for consolidation
    text_model: Optional[str] = None


@dataclass
class CameraConfig:
    device: int = 0
    width: int = 640
    jpeg_quality: int = 85


@dataclass
class VADConfig:
    start_db: float = -32.0
    hangover_ms: int = 400
    min_speech_ms: int = 300
    max_segment_s: float = 20.0


@dataclass
class AudioConfig:
    enabled: bool = True
    device: str = "plughw:1,0"
    rate: int = 16000
    channels: int = 1
    chunk_ms: int = 32
    vad: VADConfig = field(default_factory=VADConfig)


@dataclass
class ASRConfig:
    backend: str = "funasr"             # funasr | none
    model: str = "iic/SenseVoiceSmall"
    language: str = "auto"
    use_itn: bool = True


@dataclass
class SpeakerConfig:
    backend: str = "funasr"             # funasr | none
    model: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    threshold: float = 0.65


@dataclass
class FaceConfig:
    detector_model: str = "models/face_detection_yunet_2023mar.onnx"
    recognizer_model: str = "models/face_recognition_sface_2021dec.onnx"
    detect_score: float = 0.6
    match_threshold: float = 0.363
    min_face_px: int = 48


@dataclass
class IdentityConfig:
    """How face and voice are fused into one identity decision."""

    mode: str = "prefer_face"           # prefer_face | any | both


@dataclass
class MonitorConfig:
    tick_interval: float = 0.5
    describe_interval: float = 12.0
    describe_on_event: bool = True
    min_describe_gap: float = 2.0
    save_frames: bool = True
    frame_retention_days: int = 3


@dataclass
class MemoryConfig:
    path: str = "~/.qwen25vl-agent/memory.db"
    archive_dir: str = "~/.qwen25vl-agent/archive"
    working_size: int = 40
    working_ttl_s: float = 300.0
    episode_retention_days: int = 30
    observation_retention_days: int = 7
    archive_max_mb: int = 2048


@dataclass
class AlertConfig:
    webhook: Optional[str] = None
    log_path: str = "~/.qwen25vl-agent/alerts.log"
    min_importance: float = 0.5


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # -- derived paths ------------------------------------------------------------
    @property
    def db_path(self) -> str:
        return _expand(self.memory.path)

    @property
    def archive_dir(self) -> str:
        return _expand(self.memory.archive_dir)

    @property
    def alert_log(self) -> str:
        return _expand(self.alerts.log_path)

    @property
    def frames_dir(self) -> str:
        """Where episode frames are saved, next to the database rather than inside it."""
        return os.path.join(os.path.dirname(self.db_path), "frames")

    @property
    def detector_model_path(self) -> str:
        return self._resolve_model(self.face.detector_model)

    @property
    def recognizer_model_path(self) -> str:
        return self._resolve_model(self.face.recognizer_model)

    def _resolve_model(self, value: str) -> str:
        """Resolve a model path relative to the repo root when it is not absolute."""
        value = _expand(value)
        if os.path.isabs(value):
            return value
        return os.path.join(_REPO_ROOT, value)


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load the YAML config and apply CLI overrides.

    ``overrides`` is a shallow mapping of ``section -> {key: value}`` (or a whole-section
    replacement).  Keeping it shallow means a CLI flag like ``--describe-interval`` only has
    to know the one field it touches.
    """
    import yaml

    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if overrides:
        merged = dict(raw)
        for section, values in overrides.items():
            if values is None:
                continue
            if isinstance(values, dict):
                merged[section] = {**(merged.get(section) or {}), **values}
            else:
                merged[section] = values
        raw = merged

    return _build(Config, raw)
