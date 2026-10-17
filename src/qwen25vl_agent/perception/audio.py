"""Audio capture from the microphone.

Capture is done with ``arecord`` (ALSA) rather than a Python audio library because the machine
already has it and does not have ``sounddevice``/``pyaudio``.  Spawning it once and reading raw
S16_LE chunks from its stdout gives the agent a continuous stream with no extra dependency, and
the process model is simple: if ``arecord`` dies, the reader notices EOF and the monitor
restarts capture on its next tick.

On this machine ``arecord -l`` shows card 1 (``sof-hda-dsp``): device 0 is the analog mic and
device 6 is the DMIC array, so ``plughw:1,0`` and ``plughw:1,6`` both capture.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["AudioUnavailable", "AudioCapture", "list_devices", "pcm16_to_float"]


class AudioUnavailable(RuntimeError):
    """``arecord`` is missing or the device cannot be opened."""


def pcm16_to_float(raw: bytes) -> np.ndarray:
    """Little-endian int16 bytes -> float32 in [-1, 1)."""
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def list_devices() -> List[str]:
    """Parse ``arecord -l`` into ``plughw:<card>,<device>`` names for use in the config."""
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    names: List[str] = []
    for line in out.stdout.splitlines():
        # "card 1: sofhdadsp [sof-hda-dsp], device 0: HDA Analog (*)"
        if not line.startswith("card "):
            continue
        try:
            card = line.split("card ", 1)[1].split(":", 1)[0].strip()
            dev = line.split("device ", 1)[1].split(":", 1)[0].strip()
        except IndexError:
            continue
        names.append(f"plughw:{card},{dev}")
    return names


@dataclass
class AudioCapture:
    """A continuous raw-PCM stream from ``arecord``.

    ``read`` returns exactly one chunk (``chunk_ms`` worth) or ``None`` at end of stream.  The
    reader loop is expected to be the only caller, so no locking is needed here.
    """

    device: str = "plughw:1,0"
    rate: int = 16000
    channels: int = 1
    chunk_ms: int = 32
    sample_width: int = 2          # S16_LE

    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _started_at: float = 0.0

    @property
    def bytes_per_chunk(self) -> int:
        return max(1, int(self.rate * self.chunk_ms / 1000)) * self.channels * self.sample_width

    def start(self) -> None:
        cmd = [
            "arecord", "-q", "-D", self.device,
            "-f", "S16_LE", "-r", str(self.rate), "-c", str(self.channels),
            "-t", "raw", "-",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
            )
        except FileNotFoundError as exc:
            raise AudioUnavailable("audio capture needs `arecord` (alsa-utils)") from exc
        self._started_at = time.monotonic()

    def read(self) -> Optional[bytes]:
        """Read one chunk, blocking until it is available (or the stream ends)."""
        if self._proc is None or self._proc.stdout is None:
            return None
        want = self.bytes_per_chunk
        buf = bytearray()
        while len(buf) < want:
            piece = self._proc.stdout.read(want - len(buf))
            if not piece:
                return None
            buf += piece
        return bytes(buf)

    def read_float(self) -> Optional[np.ndarray]:
        raw = self.read()
        return None if raw is None else pcm16_to_float(raw)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except (subprocess.SubprocessError, OSError):
            self._proc.kill()
        finally:
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            self._proc = None

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
