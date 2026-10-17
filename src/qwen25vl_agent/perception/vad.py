"""Energy-based voice activity detection.

A full VAD (WebRTC/Silero) would be more accurate, but this needs no model and no dependency,
and its job here is narrow: cut the continuous microphone stream into speech segments so the
ASR model is only invoked when someone is actually talking.  That is what keeps a 24/7 run
from burning CPU transcribing room tone.

The state machine is deliberately simple and observable:

* speech starts when a chunk's RMS rises above ``start_db`` (dBFS);
* it continues through short dips (hysteresis at ``start_db - hysteresis_db``) and ends after
  ``hangover_ms`` of true silence;
* segments shorter than ``min_speech_ms`` are dropped as noise, and a segment longer than
  ``max_segment_s`` is force-cut so transcription stays responsive.

``start_db`` is an absolute dBFS threshold, not relative to the noise floor: an absolute value
is predictable across rooms, whereas a relative one drifts with the floor estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["Segment", "EnergyVAD"]

_EPS = 1e-9


def _rms_db(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, _EPS)))


@dataclass
class Segment:
    """A completed speech segment."""

    audio: np.ndarray              # float32 mono in [-1, 1)
    start_ts: float
    end_ts: float
    peak_db: float = -120.0

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class EnergyVAD:
    start_db: float = -32.0
    hysteresis_db: float = 6.0
    hangover_ms: int = 400
    min_speech_ms: int = 300
    max_segment_s: float = 20.0
    rate: int = 16000

    _buf: List[np.ndarray] = field(default_factory=list, repr=False)
    _in_speech: bool = False
    _start_ts: float = 0.0
    _last_voice_ts: float = 0.0
    _peak_db: float = -120.0
    _voiced_ms: float = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._in_speech = False
        self._start_ts = 0.0
        self._last_voice_ts = 0.0
        self._peak_db = -120.0
        self._voiced_ms = 0.0

    def push(self, chunk: np.ndarray, ts: float) -> Optional[Segment]:
        """Feed one chunk; return a :class:`Segment` when one just ended, else ``None``."""
        chunk_ms = 1000.0 * chunk.size / self.rate
        db = _rms_db(chunk)
        loud = db >= self.start_db
        voice = db >= (self.start_db - self.hysteresis_db)

        if not self._in_speech:
            if not loud:
                return None
            self._in_speech = True
            self._start_ts = ts - chunk_ms / 1000.0
            self._buf = []
            self._peak_db = db
            self._voiced_ms = 0.0

        self._buf.append(np.asarray(chunk, dtype=np.float32))
        self._peak_db = max(self._peak_db, db)
        if voice:
            self._voiced_ms += chunk_ms
            self._last_voice_ts = ts

        elapsed_ms = 1000.0 * (ts - self._start_ts)
        silence_ms = 1000.0 * (ts - self._last_voice_ts)

        ended = silence_ms >= self.hangover_ms or elapsed_ms >= self.max_segment_s * 1000.0
        if not ended:
            return None

        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
        segment = Segment(
            audio=audio,
            start_ts=self._start_ts,
            end_ts=ts,
            peak_db=self._peak_db,
        )
        self.reset()
        if self._voiced_ms < self.min_speech_ms and segment.duration_s * 1000.0 < self.min_speech_ms:
            return None
        if segment.audio.size == 0:
            return None
        return segment

    def flush(self) -> Optional[Segment]:
        """Force-close any open segment (used on shutdown so nothing is lost)."""
        if not self._in_speech or not self._buf:
            return None
        audio = np.concatenate(self._buf)
        segment = Segment(audio=audio, start_ts=self._start_ts,
                          end_ts=self._last_voice_ts or self._start_ts, peak_db=self._peak_db)
        self.reset()
        return segment if segment.audio.size else None
