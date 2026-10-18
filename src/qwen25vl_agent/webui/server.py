"""The web frontend's backend: a stdlib HTTP server over the agent's memory and live events.

No web framework.  ``http.server`` is enough for a local, single-user UI, and the sibling
``qwen25vl-lite`` engine already serves its own UI the same way -- keeping the dependency list
at numpy/pillow/opencv/pyyaml is a deliberate constraint of this repo.

Two things make this more than a static file server:

* ``/api/events`` is Server-Sent Events, streamed from an :class:`EventBus` the monitor publishes
  into, so the page shows scenes, speech, faces and alerts as they happen rather than polling;
* the read endpoints open a short-lived ``Memory`` on the same SQLite file.  WAL mode lets that
  reader run alongside the monitor's writer, and the connection's 30 s timeout absorbs the
  brief moments the two overlap on a write.

The write actions (enroll, rules) are the one place worth care: they share SQLite's single writer
with the monitor, so each runs as one short transaction behind a lock.
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import numpy as np

from ..imaging import rgb_to_jpeg
from ..llm import LLMClient, LLMError
from ..memory import Memory
from .events import EventBus

__all__ = ["WebApp", "make_server", "serve"]

_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
_HEALTH_TTL = 5.0          # cache service health so a 5 s page poll cannot hammer the engines
_SSE_PING = 15.0


def _as_int(query: Dict[str, list], key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def _as_float(query: Dict[str, list], key: str) -> Optional[float]:
    try:
        return float(query.get(key, [""])[0])
    except (TypeError, ValueError):
        return None


def _one(query: Dict[str, list], key: str) -> Optional[str]:
    values = query.get(key)
    return values[0] if values else None


def _strip_data_url(value: str) -> str:
    """Drop a ``data:image/...;base64,`` prefix, so both a raw blob and a data URL work."""
    text = (value or "").strip()
    if text.startswith("data:") and "," in text:
        return text.split(",", 1)[1]
    return text


class WebApp:
    """Everything the HTTP handlers call.  Kept free of ``http.server`` so it is easy to test."""

    def __init__(self, cfg, *, monitor: Optional[Any] = None, bus: Optional[EventBus] = None,
                 tts: Optional[Any] = None, memory_factory=None) -> None:
        self.cfg = cfg
        self.monitor = monitor
        self.bus = bus or EventBus()
        self._tts = tts
        self._tts_tried = tts is not None
        self._memory_factory = memory_factory or (
            lambda: Memory(cfg, frames_dir=cfg.frames_dir))
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_limits: tuple = (0, 0.0)
        self._monitor_error: Optional[str] = None
        self._health_cache: tuple = (0.0, None)
        self._shutdown = threading.Event()
        if self.monitor is not None:
            # Register once, at construction: Monitor.run() can only be called once, so there is
            # never a second start that would double-register the listener.
            self.monitor.on_event(self.bus.publish)

    # -- lifecycle ------------------------------------------------------------------
    @property
    def shutdown(self) -> threading.Event:
        return self._shutdown

    def _memory(self) -> Memory:
        return self._memory_factory()

    def _tts_engine(self):
        if not self._tts_tried:
            from ..tts import make_engine

            self._tts = make_engine(self.cfg)
            self._tts_tried = True
        return self._tts

    # -- monitor control ------------------------------------------------------------
    def monitor_running(self) -> bool:
        thread = self._monitor_thread
        return bool(thread and thread.is_alive())

    def start_monitor(self, *, max_ticks: int = 0, duration_s: float = 0.0) -> bool:
        with self._lock:
            if self.monitor is None or self.monitor_running():
                return False
            self._monitor_limits = (int(max_ticks or 0), float(duration_s or 0.0))
            self._monitor_thread = threading.Thread(
                target=self._run_monitor, name="monitor", daemon=True)
            self._monitor_thread.start()
        return True

    def stop_monitor(self) -> bool:
        if self.monitor is None:
            return False
        self.monitor.stop()
        return True

    def join_monitor(self, timeout: float = 5.0) -> None:
        thread = self._monitor_thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _run_monitor(self) -> None:
        max_ticks, duration_s = self._monitor_limits
        self._monitor_error = None
        try:
            self.monitor.run(max_ticks=max_ticks, duration_s=duration_s)
        except Exception as exc:
            # A failed startup (camera busy, database unreadable) kills the monitor thread, and the
            # error event would be published before any browser subscribed -- so it is also kept
            # here, where ``/api/state`` can hand it to a page that loads afterwards.
            self._monitor_error = str(exc)
            self.bus.publish({"ts": time.time(), "kind": "error",
                              "message": f"monitor stopped: {exc}", "importance": 0.0})

    # -- state ----------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        stats = self.monitor.stats if self.monitor is not None else {}
        try:
            with self._memory() as mem:
                memory_stats = mem.stats()
        except Exception as exc:      # a locked or missing database must not break the page
            memory_stats = {"error": str(exc)}

        engine = self._tts_engine()
        return {
            "monitor": {"running": self.monitor_running(), "stats": stats,
                        "error": self._monitor_error},
            "config": {
                "llm_url": self.cfg.llm.url,
                "text_url": self.cfg.llm.text_url,
                "camera": f"/dev/video{self.cfg.camera.device}",
                "audio": self.cfg.audio.device if self.cfg.audio.enabled else None,
                "db_path": self.cfg.db_path,
                "frames_dir": self.cfg.frames_dir,
                "tts": engine.describe() if engine is not None else "unavailable",
                "describe_interval": self.cfg.monitor.describe_interval,
                "tick_interval": self.cfg.monitor.tick_interval,
            },
            "services": self._services(),
            "memory": memory_stats,
        }

    def _services(self) -> Dict[str, Any]:
        now = time.time()
        cached_at, cached = self._health_cache
        if cached is not None and now - cached_at < _HEALTH_TTL:
            return cached
        out: Dict[str, Any] = {"llm": self._health(self.cfg.llm.url)}
        if self.cfg.llm.text_url:
            out["text"] = self._health(self.cfg.llm.text_url)
        else:
            out["text"] = {"ok": False, "detail": "not configured"}
        self._health_cache = (now, out)
        return out

    def _health(self, url: str) -> Dict[str, Any]:
        client = LLMClient(url=url, timeout=min(float(self.cfg.llm.timeout), 5.0))
        name: Optional[str] = None
        try:
            name = client.health().get("model")
        except LLMError:
            pass
        if not name:
            # llama.cpp's /health carries no model field; /v1/models does.
            try:
                ids = client.models()
                name = ids[0] if ids else None
            except LLMError as exc:
                return {"ok": False, "detail": str(exc)}
        if not name:
            return {"ok": False, "detail": "unreachable"}
        return {"ok": True, "detail": name}

    # -- reads ----------------------------------------------------------------------
    def persons(self) -> list:
        with self._memory() as mem:
            return [{**p.as_dict(), **mem.person.embedding_counts(p.id)} for p in mem.person.all()]

    def person_history(self, person_id: Any) -> list:
        with self._memory() as mem:
            return mem.person.history(int(person_id), limit=50)

    def episodes(self, limit: int = 50, kind: Optional[str] = None,
                 since: Optional[float] = None) -> list:
        with self._memory() as mem:
            return [e.as_dict() for e in mem.episodic.recent(limit=limit, kind=kind, since=since)]

    def facts(self, limit: int = 50) -> list:
        with self._memory() as mem:
            return [f.as_dict() for f in mem.semantic.get(limit=limit)]

    def rules(self) -> list:
        with self._memory() as mem:
            return [r.as_dict() for r in mem.procedural.all()]

    def search(self, query: str, limit: int = 50) -> Dict[str, list]:
        with self._memory() as mem:
            return {
                "episodes": [e.as_dict() for e in mem.episodic.search(query, limit=limit)],
                "facts": [f.as_dict() for f in mem.semantic.search(query, limit=limit)],
            }

    # -- images ---------------------------------------------------------------------
    def latest_jpeg(self) -> Optional[bytes]:
        """The newest frame as JPEG.

        While the monitor runs this is the frame it just grabbed, encoded on demand -- no second
        reader on the camera, and no per-tick encode when nobody is watching.  With no monitor it
        falls back to the most recently saved frame on disk.
        """
        rgb = self.monitor.latest_rgb() if self.monitor is not None else None
        if rgb is not None:
            return rgb_to_jpeg(rgb, self.cfg.camera.jpeg_quality)
        name = self._newest_frame()
        return self.frame_bytes(name) if name else None

    def _newest_frame(self) -> Optional[str]:
        try:
            names = [n for n in os.listdir(self.cfg.frames_dir) if n.endswith(".jpg")]
        except OSError:
            return None
        if not names:
            return None
        return max(names, key=lambda n: os.path.getmtime(os.path.join(self.cfg.frames_dir, n)))

    def frame_bytes(self, name: Optional[str]) -> Optional[bytes]:
        """Read one saved frame, refusing anything that is not a bare filename.

        A path like ``../memory.db`` or ``sub/x.jpg`` must never escape ``frames_dir``, so the
        only accepted input is a name that equals its own basename and ends in ``.jpg``.
        """
        safe = os.path.basename(name or "")
        if not safe or safe != name or not safe.endswith(".jpg"):
            return None
        try:
            with open(os.path.join(self.cfg.frames_dir, safe), "rb") as fh:
                return fh.read()
        except OSError:
            return None

    # -- actions --------------------------------------------------------------------
    def action_speak(self, text: str) -> Dict[str, Any]:
        engine = self._tts_engine()
        if engine is None:
            return {"ok": False, "error": "tts unavailable"}
        ok = engine.say_async(str(text))
        return {"ok": bool(ok), "backend": engine.backend, "player": engine.player}

    def action_summary(self, window: Optional[float] = None, say: bool = True) -> Dict[str, Any]:
        from ..agent import Agent
        from ..summary import HourlySummarizer

        window_s = float(window) if window else self.cfg.summary.window_s
        with self._memory() as mem:
            result = HourlySummarizer(self.cfg, Agent(self.cfg)).summarize(
                mem, time.time(), window_s=window_s)
        out = result.as_dict()
        if say and result.ok and result.text:
            engine = self._tts_engine()
            if engine is not None and engine.say_async(result.text):
                out["spoken"] = True
        return out

    def action_enroll(self, name: str, image_b64: Optional[str] = None,
                      audio_seconds: float = 0.0, notes: str = "") -> Dict[str, Any]:
        """Enroll a person from an uploaded photo (and optionally a voice sample).

        The image path is the practical one for a web UI: the monitor owns the camera while it
        runs, so the page cannot open a second capture -- a photo the operator uploads avoids
        that contention entirely.
        """
        from ..perception.face import FaceEngine

        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "a name is required"}
        if not image_b64:
            return {"ok": False, "error": "no image provided"}

        try:
            from PIL import Image

            raw = base64.b64decode(_strip_data_url(image_b64))
            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            return {"ok": False, "error": f"could not read the image: {exc}"}

        face = FaceEngine(self.cfg.detector_model_path, self.cfg.recognizer_model_path,
                          detect_score=self.cfg.face.detect_score,
                          min_face_px=self.cfg.face.min_face_px)
        if not face.available():
            return {"ok": False, "error": "face models unavailable (run scripts/fetch_models.sh)"}
        usable = [f for f in face.analyze(rgb) if f.embedding is not None]
        if not usable:
            return {"ok": False, "error": "no usable face in the image"}

        best = max(usable, key=lambda f: f.area)
        with self._lock:
            with self._memory() as mem:
                person = mem.person.ensure(name, notes=notes or "")
                mem.person.add_face(person, best.embedding, quality=best.score)
                mem.person.touch(person.id)
                if audio_seconds and float(audio_seconds) > 0:
                    voice = self._capture_voice(float(audio_seconds))
                    if voice is not None:
                        mem.person.add_voice(person, voice, quality=1.0)
                counts = mem.person.embedding_counts(person.id)
        return {"ok": True, "name": name, "faces": counts["faces"], "voices": counts["voices"]}

    def _capture_voice(self, seconds: float):
        """Record ``seconds`` from the microphone and return a voiceprint, or ``None``."""
        from ..perception.audio import AudioCapture, AudioUnavailable
        from ..perception.speaker import SpeakerEngine

        engine = SpeakerEngine(model=self.cfg.speaker.model)
        try:
            capture = AudioCapture(device=self.cfg.audio.device, rate=self.cfg.audio.rate,
                                   channels=self.cfg.audio.channels, chunk_ms=self.cfg.audio.chunk_ms)
            capture.start()
        except AudioUnavailable:
            return None
        chunks = []
        try:
            deadline = time.time() + seconds
            while time.time() < deadline:
                chunk = capture.read_float()
                if chunk is None:
                    break
                chunks.append(chunk)
        finally:
            capture.stop()
        if not chunks:
            return None
        try:
            return engine.embed(np.concatenate(chunks), rate=self.cfg.audio.rate)
        except Exception:
            return None

    def action_rules(self, action: str, name: str) -> Dict[str, Any]:
        with self._lock:
            with self._memory() as mem:
                if action == "enable":
                    ok = mem.procedural.set_enabled(name, True)
                elif action == "disable":
                    ok = mem.procedural.set_enabled(name, False)
                elif action == "remove":
                    ok = mem.procedural.remove(name)
                else:
                    return {"ok": False, "error": f"unknown action {action!r}"}
                return {"ok": bool(ok), "rules": [r.as_dict() for r in mem.procedural.all()]}


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen25vl-agent-webui"

    @property
    def app(self) -> WebApp:
        return self.server.app          # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:      # keep the terminal readable
        return

    # -- routing --------------------------------------------------------------------
    def do_GET(self) -> None:      # noqa: N802
        path, query = self._split()
        try:
            if path in ("/", "/index.html"):
                return self._send_file(_INDEX, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send_json(self.app.state())
            if path == "/api/events":
                return self._sse()
            if path == "/api/persons":
                return self._send_json(self.app.persons())
            if path.startswith("/api/persons/"):
                rest = path[len("/api/persons/"):].rstrip("/")
                if rest.endswith("/history"):
                    rest = rest[:-len("/history")]
                return self._send_json(self.app.person_history(rest))
            if path == "/api/episodes":
                return self._send_json(self.app.episodes(
                    limit=_as_int(query, "limit", 50), kind=_one(query, "kind"),
                    since=_as_float(query, "since")))
            if path == "/api/facts":
                return self._send_json(self.app.facts(limit=_as_int(query, "limit", 50)))
            if path == "/api/rules":
                return self._send_json(self.app.rules())
            if path == "/api/search":
                return self._send_json(self.app.search(
                    _one(query, "q") or "", limit=_as_int(query, "limit", 50)))
            if path == "/api/latest.jpg":
                return self._send_bytes(self.app.latest_jpeg(), "image/jpeg")
            if path.startswith("/api/frame/"):
                name = urllib.parse.unquote(path[len("/api/frame/"):])
                return self._send_bytes(self.app.frame_bytes(name), "image/jpeg")
            return self._send_json({"error": "not found", "path": path}, status=404)
        except Exception as exc:      # a handler bug returns 500 rather than dropping the socket
            return self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:      # noqa: N802
        path, _ = self._split()
        body = self._read_body()
        try:
            if path == "/api/actions/speak":
                return self._send_json(self.app.action_speak(str(body.get("text", ""))))
            if path == "/api/actions/summary":
                return self._send_json(self.app.action_summary(
                    body.get("window"), bool(body.get("say", True))))
            if path == "/api/actions/enroll":
                return self._send_json(self.app.action_enroll(
                    body.get("name", ""), body.get("image"), body.get("audio_seconds", 0.0),
                    body.get("notes", "")))
            if path == "/api/actions/rules":
                return self._send_json(self.app.action_rules(
                    str(body.get("action", "")), str(body.get("name", ""))))
            if path == "/api/actions/monitor":
                action = str(body.get("action", ""))
                if action == "start":
                    return self._send_json({"ok": self.app.start_monitor()})
                if action == "stop":
                    return self._send_json({"ok": self.app.stop_monitor()})
                return self._send_json({"ok": False, "error": f"unknown action {action!r}"},
                                       status=400)
            return self._send_json({"error": "not found", "path": path}, status=404)
        except Exception as exc:
            return self._send_json({"error": str(exc)}, status=500)

    # -- helpers --------------------------------------------------------------------
    def _split(self):
        parts = urllib.parse.urlsplit(self.path)
        return parts.path, urllib.parse.parse_qs(parts.query)

    def _read_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _send_json(self, obj: Any, status: int = 200) -> None:
        payload = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data: Optional[bytes], ctype: str, status: int = 200) -> None:
        if not data:
            return self._send_json({"error": "no image available"}, status=404)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: str, ctype: str) -> None:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return self._send_json({"error": "index.html missing"}, status=500)
        self._send_bytes(data, ctype)

    def _sse(self) -> None:
        q = self.app.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            # An initial comment so the browser's EventSource fires onopen right away.
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while not self.app.shutdown.is_set():
                try:
                    item = q.get(timeout=_SSE_PING)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")      # heartbeat keeps proxies from closing it
                    self.wfile.flush()
                    continue
                payload = json.dumps(item, ensure_ascii=False, default=str)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                       # the browser tab closed; that is not an error
        finally:
            self.app.bus.unsubscribe(q)


def make_server(app: WebApp, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, int(port)), _Handler)
    httpd.daemon_threads = True
    httpd.app = app                 # type: ignore[attr-defined]
    return httpd


def serve(app: WebApp, host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Run the UI until interrupted.  Returns after Ctrl-C."""
    host = host or app.cfg.web.host
    port = int(port if port is not None else app.cfg.web.port)
    httpd = make_server(app, host, port)
    print(f"qwen25vl-agent webui on http://{host}:{port}   (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        app.shutdown.set()
        httpd.server_close()
