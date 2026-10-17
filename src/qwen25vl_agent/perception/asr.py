"""Speech recognition (ASR) via funasr's SenseVoice.

SenseVoice is chosen for two reasons that fit this project:

* it is multilingual and *reports the language it heard* -- so "语言识别" is a by-product of
  transcription rather than a second model;
* it is small enough to run on CPU, which matters because every megabyte of VRAM here is
  already committed to the served Qwen2.5-VL.

Model weights come from modelscope (``iic/SenseVoiceSmall``), not huggingface.co: on this
network modelscope downloads at ~10 MB/s while huggingface.co is unreachable.  The first call
therefore downloads ~940 MB; ``doctor`` reports whether that has happened.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Transcript", "ASREngine", "ASRUnavailable"]

# SenseVoice emits leading control tags, e.g. "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好".
_TAG_RE = re.compile(r"<\|([^|]*)\|>")


class ASRUnavailable(RuntimeError):
    """funasr is missing or the model could not be loaded."""


@dataclass
class Transcript:
    """One utterance, with the language SenseVoice heard it in."""

    text: str
    language: str = "unknown"
    emotion: str = "unknown"
    event: str = "unknown"
    duration_s: float = 0.0
    raw: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "emotion": self.emotion,
            "event": self.event,
            "duration_s": round(self.duration_s, 3),
        }


def _parse_tags(raw: str) -> tuple:
    """Pull language/emotion/event out of SenseVoice's tag prefix."""
    tags = _TAG_RE.findall(raw or "")
    language = emotion = event = "unknown"
    # Tag order is fixed: language, emotion, event, itn-mode.
    if len(tags) >= 1 and tags[0]:
        language = tags[0]
    if len(tags) >= 2 and tags[1]:
        emotion = tags[1]
    if len(tags) >= 3 and tags[2]:
        event = tags[2]
    return language, emotion, event


class ASREngine:
    """Lazily-loaded SenseVoice wrapper.

    The model is loaded on first use, never at import: a 24/7 agent that restarts should not pay
    ~10 s of model load before it can start watching the camera, and a deployment with the ASR
    backend disabled should never touch it at all.
    """

    def __init__(self, model: str = "iic/SenseVoiceSmall", language: str = "auto",
                 use_itn: bool = True, device: str = "cpu") -> None:
        self.model_id = model
        self.language = language
        self.use_itn = use_itn
        self.device = device
        self._model = None
        self._load_error: Optional[str] = None

    @staticmethod
    def available() -> bool:
        try:
            import funasr  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ASRUnavailable(
                "speech recognition needs funasr: pip install 'qwen25vl-agent[asr]'"
            ) from exc
        try:
            self._model = AutoModel(
                model=self.model_id,
                device=self.device,
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as exc:
            raise ASRUnavailable(f"could not load ASR model {self.model_id}: {exc}") from exc

    def transcribe(self, audio: np.ndarray, rate: int = 16000) -> Optional[Transcript]:
        """Transcribe a float32 mono segment.  Returns ``None`` when nothing was recognised.

        ``audio`` is passed as an array rather than a file so the monitor never has to write
        speech to disk just to transcribe it -- the archive keeps text, not audio.
        """
        self.load()
        assert self._model is not None
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < rate // 10:      # under 100 ms is not worth a forward pass
            return None

        t0 = time.perf_counter()
        try:
            res = self._model.generate(
                input=audio,
                language=self.language,
                use_itn=self.use_itn,
                batch_size_s=60,
                disable_pbar=True,
            )
        except Exception as exc:
            raise ASRUnavailable(f"ASR inference failed: {exc}") from exc

        raw = ""
        if res:
            first = res[0]
            raw = first.get("text", "") if isinstance(first, dict) else str(first)
        language, emotion, event = _parse_tags(raw)
        text = _strip_tags(raw)
        if not text.strip():
            return None
        return Transcript(
            text=text.strip(),
            language=language,
            emotion=emotion,
            event=event,
            duration_s=audio.size / float(rate),
            raw=raw,
        )

    def warmup(self) -> bool:
        """Load the model and run one short pass, so the first real utterance is not slow."""
        try:
            self.load()
            self.transcribe(np.zeros(16000, dtype=np.float32))
            return True
        except ASRUnavailable:
            return False


def _strip_tags(raw: str) -> str:
    """Remove SenseVoice's control tags, preferring funasr's own postprocessor if present."""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(raw)
    except Exception:
        return _TAG_RE.sub("", raw or "")
