"""The web frontend's backend, exercised over a real HTTP socket.

The server is started on an ephemeral port in a background thread, so these tests hit it the way
a browser would -- no mocking of ``http.server``.  A hermetic config (``make_cfg``) keeps every
database and frame directory inside ``tmp_path``, and the monitor is omitted: the point here is
the API surface, not the tick loop (that is ``test_monitor.py``).
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from qwen25vl_agent.webui import EventBus, WebApp, make_server
from qwen25vl_agent.webui.server import _INDEX


class StubTTS:
    """Records what was asked to be spoken; never touches a real audio backend."""

    def __init__(self):
        self.backend = "stub"
        self.player = "none"
        self.said = []

    def describe(self):
        return "backend=stub"

    def say_async(self, text):
        self.said.append(text)
        return True


@pytest.fixture
def web(make_cfg):
    """A running server plus its app, on an ephemeral loopback port."""
    cfg = make_cfg()
    app = WebApp(cfg, monitor=None, tts=StubTTS())
    httpd = make_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, app, cfg
    finally:
        app.shutdown.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(base, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _png_b64(size=(64, 64)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -- page + state -----------------------------------------------------------
def test_index_is_served(web):
    base, _, _ = web
    with urllib.request.urlopen(base + "/", timeout=5) as resp:
        body = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "qwen25vl-agent" in body and "/api/events" in body


def test_state_reports_monitor_memory_and_services(web):
    base, _, _ = web
    state = _get(base, "/api/state")
    assert state["monitor"]["running"] is False
    assert state["monitor"]["stats"] == {}
    assert "episodes" in state["memory"]
    assert state["config"]["camera"].startswith("/dev/video")
    # With no engine listening, health degrades to ok=False rather than raising.
    assert state["services"]["llm"]["ok"] in (True, False)


# -- live events ------------------------------------------------------------
def test_sse_streams_a_published_event(web):
    base, app, _ = web
    resp = urllib.request.urlopen(base + "/api/events", timeout=5)
    assert resp.readline().startswith(b":")          # the primer comment
    resp.readline()                                  # its blank line
    app.bus.publish({"ts": time.time(), "kind": "scene", "message": "测试画面", "importance": 0.2})
    line = resp.readline()
    assert line.startswith(b"data: ")
    payload = json.loads(line[len(b"data: "):].decode("utf-8"))
    assert payload["kind"] == "scene"
    assert payload["message"] == "测试画面"
    resp.close()


def test_bus_drops_oldest_when_a_subscriber_is_full():
    bus = EventBus(max_queue=2)
    q = bus.subscribe()
    for i in range(4):
        bus.publish({"ts": time.time(), "kind": "tick", "message": f"m{i}"})
    # The ring keeps the newest two, not the oldest -- a live feed must not fall behind.
    assert [q.get_nowait()["message"] for _ in range(2)] == ["m2", "m3"]


# -- memory reads -----------------------------------------------------------
def test_persons_are_listed_with_embedding_counts(web):
    base, _, cfg = web
    from qwen25vl_agent.memory import Memory

    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        person = mem.person.ensure("小明")
        mem.person.add_face(person, np.ones(128, dtype=np.float32))

    rows = _get(base, "/api/persons")
    assert len(rows) == 1
    assert rows[0]["name"] == "小明"
    assert rows[0]["faces"] == 1 and rows[0]["voices"] == 0


def test_person_history_endpoint(web):
    base, _, cfg = web
    from qwen25vl_agent.memory import Memory

    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        person = mem.person.ensure("小明")
        mem.person.note(person.id, "在书桌前坐下", kind="note")
        pid = person.id

    rows = _get(base, f"/api/persons/{pid}/history")
    assert rows and rows[0]["text"] == "在书桌前坐下"


def test_rules_can_be_toggled(web):
    base, _, _ = web
    rules = _get(base, "/api/rules")
    assert rules, "seed_defaults should have created the starter rules"
    name = rules[0]["name"]

    status, body = _post(base, "/api/actions/rules", {"action": "disable", "name": name})
    assert status == 200 and body["ok"] is True
    assert next(r for r in body["rules"] if r["name"] == name)["enabled"] is False

    status, body = _post(base, "/api/actions/rules", {"action": "enable", "name": name})
    assert body["ok"] is True


# -- images -----------------------------------------------------------------
def test_frame_bytes_rejects_traversal(web):
    """The guard is the app method, so it is asserted directly rather than through URL rewriting."""
    _, app, _ = web
    assert app.frame_bytes("../memory.db") is None
    assert app.frame_bytes("sub/x.jpg") is None
    assert app.frame_bytes("/etc/passwd") is None
    assert app.frame_bytes("notes.txt") is None
    assert app.frame_bytes(None) is None


def test_frame_endpoint_returns_404_for_a_bad_name(web):
    base, _, _ = web
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/api/frame/sub%2Fx.jpg", timeout=5)
    assert exc.value.code == 404


def test_a_saved_frame_is_served(web):
    base, _, cfg = web
    import os

    os.makedirs(cfg.frames_dir, exist_ok=True)
    path = os.path.join(cfg.frames_dir, "20260101-000000-000000.jpg")
    with open(path, "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0jpegbytes")

    with urllib.request.urlopen(base + "/api/frame/20260101-000000-000000.jpg", timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "image/jpeg"
        assert resp.read().startswith(b"\xff\xd8")


def test_latest_jpeg_falls_back_to_the_newest_saved_frame(web):
    base, _, cfg = web
    import os

    os.makedirs(cfg.frames_dir, exist_ok=True)
    with open(os.path.join(cfg.frames_dir, "a.jpg"), "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0old")
    with open(os.path.join(cfg.frames_dir, "b.jpg"), "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0new")

    with urllib.request.urlopen(base + "/api/latest.jpg", timeout=5) as resp:
        assert resp.read().endswith(b"new")


# -- actions ----------------------------------------------------------------
def test_speak_action_uses_the_tts_engine(web):
    base, app, _ = web
    status, body = _post(base, "/api/actions/speak", {"text": "测试播报"})
    assert status == 200 and body["ok"] is True
    assert app._tts.said == ["测试播报"]


def test_monitor_actions_report_false_without_a_monitor(web):
    base, _, _ = web
    status, body = _post(base, "/api/actions/monitor", {"action": "start"})
    assert status == 200 and body["ok"] is False


class FakeMonitor:
    """A monitor stand-in, so the thread wiring can be checked without a camera."""

    def __init__(self):
        self.listeners = []
        self.limits = None
        self.stopped = False

    def on_event(self, fn):
        self.listeners.append(fn)

    def run(self, *, max_ticks=0, duration_s=0.0):
        self.limits = (max_ticks, duration_s)
        for fn in self.listeners:
            fn(type("E", (), {"ts": time.time(), "kind": "tick", "message": "started",
                              "importance": 0.0, "data": {}})())

    def stop(self):
        self.stopped = True

    def latest_rgb(self):
        return None

    @property
    def stats(self):
        return {"ticks": 1}


def test_start_monitor_honors_the_test_limits(make_cfg):
    """``--max-ticks`` / ``--duration`` must reach ``Monitor.run``; they were being dropped."""
    app = WebApp(make_cfg(), monitor=FakeMonitor())
    assert app.start_monitor(max_ticks=3, duration_s=1.5) is True
    app.join_monitor(timeout=3)
    assert app.monitor.limits == (3, 1.5)
    assert app.stop_monitor() is True
    assert app.monitor.stopped is True


class ExplodingMonitor(FakeMonitor):
    def run(self, *, max_ticks=0, duration_s=0.0):
        raise RuntimeError("could not open /dev/video0")


def test_a_failed_monitor_start_is_surfaced_in_state(make_cfg):
    """A page loaded *after* the monitor died must still learn why it is not running."""
    app = WebApp(make_cfg(), monitor=ExplodingMonitor())
    assert app.start_monitor() is True
    app.join_monitor(timeout=3)

    monitor = app.state()["monitor"]
    assert monitor["running"] is False
    assert monitor["error"] == "could not open /dev/video0"


def test_enroll_requires_a_name_and_an_image(web):
    base, app, _ = web
    assert app.action_enroll("", image_b64="x")["ok"] is False
    assert app.action_enroll("小明", image_b64="")["ok"] is False
    assert app.action_enroll("小明", image_b64="not-base64!!")["ok"] is False


def test_enroll_stores_a_face_embedding(web, monkeypatch):
    """The enroll plumbing (decode -> detect -> store) without needing the real ONNX models."""
    base, _, cfg = web
    import qwen25vl_agent.perception.face as face_mod

    class FakeFaceEngine:
        def __init__(self, *args, **kwargs):
            pass

        def available(self):
            return True

        def analyze(self, rgb):
            return [face_mod.FaceDetection(box=(0, 0, 50, 50), score=0.9,
                                           embedding=np.ones(128, dtype=np.float32))]

    monkeypatch.setattr(face_mod, "FaceEngine", FakeFaceEngine)

    status, body = _post(base, "/api/actions/enroll", {"name": "小明", "image": _png_b64()})
    assert status == 200 and body["ok"] is True
    assert body["faces"] == 1 and body["voices"] == 0

    rows = _get(base, "/api/persons")
    assert [p["name"] for p in rows] == ["小明"]
