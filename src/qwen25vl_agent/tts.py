"""Text-to-speech for the hourly broadcast.

The monitor is a long-running process, so nothing here is allowed to raise out of a tick: a
missing binary, a dead network or a broken player degrades to "no audio" and the loop carries
on.  That is also why synthesis and playback run on one worker thread -- a two-second utterance
must never stall a perception tick.

Backends, in the order ``auto`` prefers them:

* ``piper``   -- a local neural voice; fully offline, but needs ``piper`` and a voice ``.onnx``.
* ``edge``    -- Microsoft Edge's online voices via ``edge-tts``.  Best Chinese quality of the
  group, but it needs the network at speak time.
* ``espeak``  -- ``espeak-ng``, offline and tiny; robotic, but it is always there.
* ``spd-say`` -- speech-dispatcher, when it is configured with an output module.
* ``command`` -- an operator-supplied shell template, for anything else.

Audio lands in a temp file and is played by the first available of
``pw-play``/``paplay``/``ffplay``/``aplay``.  ``aplay`` only understands WAV, so an MP3 from
``edge`` is transcoded through ``soundfile`` before it is handed over.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Optional

__all__ = ["TTSEngine", "TTSUnavailable", "detect_backend", "detect_player"]

BACKENDS = ("auto", "piper", "edge", "espeak", "spd-say", "command", "none")
_PLAYER_ORDER = ("pw-play", "paplay", "ffplay", "aplay")


class TTSUnavailable(RuntimeError):
    """Raised when no usable speech backend exists on this machine."""


def detect_player(preferred: Optional[str] = None) -> Optional[str]:
    """First available player, or the explicitly requested one when it exists."""
    if preferred and preferred not in ("", "auto"):
        return preferred if shutil.which(preferred) else None
    for name in _PLAYER_ORDER:
        if shutil.which(name):
            return name
    return None


def _has_edge() -> bool:
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False


def detect_backend(preferred: str = "auto") -> str:
    """Resolve ``auto`` to a concrete backend, or pass a requested one through."""
    if preferred and preferred != "auto":
        return preferred
    if shutil.which("piper"):
        return "piper"
    if _has_edge():
        return "edge"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    if shutil.which("spd-say"):
        return "spd-say"
    return "none"


class TTSEngine:
    """Synthesize text to a file and play it, on a worker thread when asked asynchronously."""

    def __init__(self, *, backend: str = "auto", voice: str = "zh-CN-XiaoxiaoNeural",
                 player: str = "auto", rate: str = "+0%", espeak_voice: str = "cmn",
                 max_chars: int = 400, timeout: float = 60.0,
                 command: Optional[str] = None, tmp_dir: Optional[str] = None,
                 on_error: Optional[Callable[[str], None]] = None) -> None:
        self.backend = detect_backend(backend)
        self.voice = voice
        self.player = detect_player(player)
        self.rate = rate or ""
        self.espeak_voice = espeak_voice
        self.max_chars = int(max_chars)
        self.timeout = float(timeout)
        self.command = command
        self.tmp_dir = tmp_dir
        self._on_error = on_error

        self._q: "queue.Queue[str]" = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_error: str = ""

    # -- capability ------------------------------------------------------------------
    def available(self) -> bool:
        """True when this backend can plausibly produce audio right now."""
        if self.backend == "none":
            return False
        if self.backend == "command":
            return bool(self.command)
        if self.backend == "piper":
            return bool(shutil.which("piper")) and bool(self.voice)
        if self.backend == "edge":
            return _has_edge()
        if self.backend == "espeak":
            return bool(shutil.which("espeak-ng") or shutil.which("espeak"))
        if self.backend == "spd-say":
            return bool(shutil.which("spd-say"))
        return False

    def describe(self) -> str:
        """A one-line summary for ``doctor``."""
        if not self.available():
            return f"backend={self.backend} (unavailable)"
        return f"backend={self.backend} player={self.player or 'none'} voice={self.voice}"

    # -- text handling ---------------------------------------------------------------
    def _clean(self, text: str) -> str:
        """Collapse whitespace and cap the length so an hour of log never becomes an hour of audio."""
        text = " ".join((text or "").split())
        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "……"
        return text

    def _error(self, message: str) -> None:
        self.last_error = message
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:
                pass

    def _run(self, cmd, *, stdin: Optional[str] = None) -> bool:
        try:
            proc = subprocess.run(
                cmd, input=stdin, capture_output=True, timeout=self.timeout, text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._error(f"{cmd[0]}: {exc}")
            return False
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            self._error(f"{cmd[0]} exited {proc.returncode}: {detail[-1] if detail else ''}")
            return False
        return True

    # -- synthesis -------------------------------------------------------------------
    def _synth(self, text: str, out_path: str) -> bool:
        if self.backend == "edge":
            cmd = [sys.executable, "-m", "edge_tts", "--voice", self.voice,
                   "--text", text, "--write-media", out_path]
            if self.rate and self.rate != "+0%":
                cmd += ["--rate", self.rate]
            return self._run(cmd)
        if self.backend == "piper":
            # For piper, `voice` is the path to the voice .onnx; text arrives on stdin.
            return self._run(["piper", "--model", self.voice, "--output_file", out_path],
                             stdin=text)
        if self.backend == "espeak":
            binary = shutil.which("espeak-ng") or shutil.which("espeak") or "espeak"
            return self._run([binary, "-v", self.espeak_voice, "-w", out_path, text])
        if self.backend == "command":
            rendered = self.command.format(text=text, out=out_path)
            return self._run(["bash", "-lc", rendered])
        return False

    def _play(self, path: str) -> bool:
        if not self.player:
            self._error("no audio player found (looked for pw-play/paplay/ffplay/aplay)")
            return False
        path = self._playable(path)
        if self.player == "ffplay":
            return self._run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        return self._run([self.player, path])

    def _playable(self, path: str) -> str:
        """``aplay`` is WAV-only, so transcode an MP3 when that is the player we have."""
        if self.player != "aplay" or not path.endswith(".mp3"):
            return path
        wav = path[:-4] + ".wav"
        try:
            import soundfile as sf

            data, sr = sf.read(path)
            sf.write(wav, data, sr)
            return wav
        except Exception as exc:
            self._error(f"mp3->wav for aplay failed: {exc}")
            return path

    # -- public API ------------------------------------------------------------------
    def say(self, text: str) -> bool:
        """Synthesize and play, blocking.  Returns False (never raises) on any failure."""
        text = self._clean(text)
        if not text or not self.available():
            return False

        fd, path = tempfile.mkstemp(prefix="tts-", suffix=self._suffix(), dir=self.tmp_dir)
        os.close(fd)
        try:
            if self.backend == "spd-say":
                ok = self._run(["spd-say", "-l", "zh", text])
            else:
                ok = self._synth(text, path) and os.path.getsize(path) > 0
            if not ok:
                return False
            return self._play(path)
        except OSError as exc:
            self._error(str(exc))
            return False
        finally:
            leftovers = [path]
            if path.endswith(".mp3"):
                leftovers.append(path[:-4] + ".wav")   # a transcoded copy, if aplay made one
            for candidate in leftovers:
                try:
                    os.unlink(candidate)
                except OSError:
                    pass

    def _suffix(self) -> str:
        if self.backend == "edge":
            return ".mp3"
        if self.backend == "piper":
            return ".wav"
        return ".wav"

    def say_async(self, text: str) -> bool:
        """Queue an utterance for the worker thread.  Drops it when the queue is full."""
        text = self._clean(text)
        if not text or not self.available():
            return False
        self._ensure_worker()
        try:
            self._q.put_nowait(text)
        except queue.Full:
            return False
        return True

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="tts", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.say(text)
            except Exception as exc:      # a bad utterance must not kill the thread
                self._error(f"tts worker: {exc}")

    def close(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


def make_engine(cfg, *, on_error: Optional[Callable[[str], None]] = None) -> Optional[TTSEngine]:
    """Build the engine the config asks for, or ``None`` when TTS is off/unusable."""
    tts = cfg.tts
    if not tts.enabled or tts.backend == "none":
        return None
    engine = TTSEngine(
        backend=tts.backend, voice=tts.voice, player=tts.player, rate=tts.rate,
        espeak_voice=tts.espeak_voice, max_chars=tts.max_chars, timeout=tts.timeout,
        command=tts.command, on_error=on_error,
    )
    return engine if engine.available() else None
