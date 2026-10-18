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
    "AlertConfig", "SummaryConfig", "TTSConfig", "WebConfig", "load_config",
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
    # Degrees clockwise to rotate every frame before anything looks at it.  A camera mounted
    # upside-down still opens fine and still returns frames -- it just returns them inverted,
    # and an inverted face is one YuNet does not detect, so "no faces" looks like a detector
    # bug.  On this machine the mount is inverted, hence 180.
    rotate: int = 0


@dataclass
class VADConfig:
    # Measured on this machine with the DMIC open: room floor sits at -50 dBFS and speech from
    # the speakers lands between -18 and -44 dBFS, so -40 dBFS clears the floor by 10 dB while
    # still catching quiet speech.  The old -32 was above the quiet half of that range.
    start_db: float = -40.0
    hangover_ms: int = 400
    min_speech_ms: int = 300
    max_segment_s: float = 20.0


@dataclass
class AudioConfig:
    enabled: bool = True
    # `arecord -l` on this machine: card 1 (sof-hda-dsp) device 0 = analog jack, device 6 = DMIC.
    # The analog jack is empty here (`Mic Jack` reads off) and its input is just -87 dBFS of ADC
    # noise; the internal DMIC array is the one that actually hears the room.
    device: str = "plughw:1,6"
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
    # How long the same set of people may disappear and come back before it counts as a new
    # presence episode.  Without this, a detector that flickers on a blurry frame writes a fresh
    # "someone is here" row every few seconds, and those rows crowd every scene and alert out of
    # the agent's memory context.
    face_presence_merge_s: float = 300.0


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
class SummaryConfig:
    """The hourly spoken summary.

    ``interval_s`` is how often it fires; ``window_s`` is how much memory it reads.  They are
    separate so a summary can cover slightly more than an hour without drifting.  ``min_items``
    is the quiet-hour guard: with nothing in the window there is nothing worth saying, and
    announcing "nothing happened" every hour is worse than staying silent.  ``retry_s`` is how
    soon a *failed* broadcast is tried again -- a transient timeout should not buy an hour of
    silence.
    """

    enabled: bool = True
    interval_s: float = 3600.0
    window_s: float = 3600.0
    max_items: int = 200
    min_items: int = 1
    retry_s: float = 300.0             # retry a failed broadcast after this long
    speak: bool = True                 # read the summary aloud through TTS


@dataclass
class TTSConfig:
    """How the agent speaks.

    ``backend=auto`` picks the best available of piper -> edge -> espeak -> spd-say.  For piper,
    ``voice`` is the path to the voice ``.onnx``; for edge it is a voice name such as
    ``zh-CN-XiaoxiaoNeural``.  ``command`` is a shell template with ``{text}`` and ``{out}``.
    """

    enabled: bool = True
    backend: str = "auto"              # auto | piper | edge | espeak | spd-say | command | none
    voice: str = "zh-CN-XiaoxiaoNeural"
    player: str = "auto"               # auto | pw-play | paplay | ffplay | aplay
    rate: str = "+0%"
    espeak_voice: str = "cmn"
    max_chars: int = 400               # cap an utterance so one hour never becomes one hour of audio
    timeout: float = 60.0
    command: Optional[str] = None


@dataclass
class WebConfig:
    """Where the optional web frontend listens.

    The port defaults to 8090 so it never collides with the two model servers the agent already
    depends on (8081 vision, 8082 text).  ``host`` defaults to loopback: the UI can enroll people
    and trigger speech, so it is not something to expose on the LAN by accident.
    """

    host: str = "127.0.0.1"
    port: int = 8090


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
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    web: WebConfig = field(default_factory=WebConfig)

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
