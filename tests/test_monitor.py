"""The monitoring loop, driven with fake hardware so the tick pipeline is testable without a
camera, a microphone, or a running model server.

The point of these tests is the wiring: does a tick sample a frame, match faces against person
memory, call the agent, write memory, evaluate rules, and keep going after a failure?
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from qwen25vl_agent.agent import Decision
from qwen25vl_agent.memory import Memory
from qwen25vl_agent.monitor import Monitor
from qwen25vl_agent.perception.face import FaceDetection
from qwen25vl_agent.perception.video import CameraUnavailable, Frame


@pytest.fixture
def monitor_cfg(make_cfg):
    """A config whose tick loop describes every tick, so a 2-tick run exercises the agent twice."""
    return make_cfg(monitor={"tick_interval": 0.01, "describe_interval": 0.0,
                             "min_describe_gap": 0.0})


class FakeCamera:
    """Hands back a blank frame on every grab."""

    def __init__(self, shape=(48, 64)):
        self.shape = shape
        self.opened = False
        self._index = 0

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    @property
    def is_open(self):
        return self.opened

    def grab(self):
        self._index += 1
        height, width = self.shape
        return Frame(rgb=np.zeros((height, width, 3), dtype=np.uint8),
                     ts=time.time(), index=self._index)


class DeadCamera:
    """A camera that never produces a frame, as if the USB device vanished."""

    def open(self):
        pass

    def close(self):
        pass

    def grab(self):
        return None


class FakeFace:
    """Returns a fixed list of detections, so no ONNX model is needed."""

    def __init__(self, detections=None):
        self._detections = list(detections or [])

    @staticmethod
    def models_present(detector_model, recognizer_model):
        return True

    def available(self):
        return True

    def analyze(self, rgb):
        return list(self._detections)


class FakeAgent:
    """Always returns the same decision, counting how many times it was asked."""

    def __init__(self, decision=None):
        self.decision = decision if decision is not None else Decision(scene="场景")
        self.calls = 0

    def decide(self, observation, context):
        self.calls += 1
        return self.decision


def _reopen(cfg):
    """Open a second Memory over the same database, to inspect what the run wrote."""
    return Memory(cfg, frames_dir=cfg.frames_dir)


def test_described_ticks_write_scene_episodes(monitor_cfg):
    agent = FakeAgent(Decision(scene="一个安静的测试场景"))
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=agent)

    assert monitor.run(max_ticks=2) == 2
    assert monitor.stats["describes"] == 2
    assert agent.calls == 2

    with _reopen(monitor_cfg) as check:
        scenes = [e for e in check.episodic.recent(limit=10) if e.kind == "scene"]
        assert len(scenes) == 2
        assert scenes[0].summary == "一个安静的测试场景"
        # Every tick logs a raw frame observation, whether or not the agent was called.
        assert check.store.stats()["observations"] == 2


def test_unknown_face_fires_the_alert_rule_once(monitor_cfg):
    detection = FaceDetection(box=(10, 10, 60, 60), score=0.9,
                              embedding=np.ones(128, dtype=np.float32))
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([detection]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="有人")))

    assert monitor.run(max_ticks=2) == 2

    with _reopen(monitor_cfg) as check:
        alerts = [e for e in check.episodic.recent(limit=20) if e.kind == "alert"]
        assert len(alerts) == 1                 # the 120 s cooldown holds across the two ticks
        assert "陌生人" in alerts[0].summary
        assert alerts[0].importance == pytest.approx(0.8)


def test_known_face_touches_the_person_record(monitor_cfg):
    face_vec = np.ones(128, dtype=np.float32)
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    person = mem.person.ensure("小明")
    mem.person.add_face(person, face_vec)

    detection = FaceDetection(box=(10, 10, 60, 60), score=0.9, embedding=face_vec.copy())
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([detection]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="小明在")))

    assert monitor.run(max_ticks=1) == 1

    with _reopen(monitor_cfg) as check:
        seen = check.person.get("小明")
        assert seen.seen_count == 1
        assert seen.first_seen is not None
        assert seen.last_seen is not None


def test_an_unparsed_reply_degrades_the_tick_but_keeps_the_scene(monitor_cfg):
    bad = Decision(scene="原始文本", ok=False, error="no JSON object in reply")
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(bad))

    assert monitor.run(max_ticks=2) == 2
    assert monitor.stats["describe_errors"] == 2

    with _reopen(monitor_cfg) as check:
        assert any(e.summary == "原始文本" for e in check.episodic.recent(limit=10))


def test_camera_failure_degrades_a_tick_not_the_run(monitor_cfg):
    errors = []
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=DeadCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="x")))

    def collect(event):
        if event.kind == "error":
            errors.append(event.message)

    monitor.on_event(collect)

    assert monitor.run(max_ticks=3) == 3
    assert any("no frame" in message for message in errors)


def test_frames_are_saved_for_described_ticks(monitor_cfg):
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="s")))

    assert monitor.run(max_ticks=2) == 2

    saved = [name for name in os.listdir(monitor_cfg.frames_dir) if name.endswith(".jpg")]
    assert len(saved) == 2


def test_the_hourly_archive_pass_runs_once(monitor_cfg):
    events = []
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="s")))
    monitor.on_event(lambda event: events.append(event))

    assert monitor.run(max_ticks=2) == 2

    archived = [e for e in events if e.kind == "tick" and e.message == "archived memory"]
    assert len(archived) == 1


def test_agent_receives_the_memory_context(monitor_cfg):
    """The context handed to the agent must already contain the enrolled roster."""
    captured = {}

    class CapturingAgent(FakeAgent):
        def decide(self, observation, context):
            captured.update(context)
            return Decision(scene="s")

    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    mem.person.ensure("小明")
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=CapturingAgent())

    assert monitor.run(max_ticks=1) == 1
    assert captured["known_people"][0]["name"] == "小明"
    assert "unknown_face" in captured["rules"]


class ExplodingCamera:
    """A camera that fails to open, as if the device is missing at startup."""

    def open(self):
        raise CameraUnavailable("could not open /dev/video0")

    def close(self):
        pass

    def grab(self):
        return None


def test_startup_failure_still_releases_resources(monitor_cfg):
    """A failed ``_setup`` must still close the camera and the database connection."""
    closed = []
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    original_close = mem.close

    def spy_close():
        closed.append(True)
        original_close()

    mem.close = spy_close

    monitor = Monitor(monitor_cfg, memory=mem, camera=ExplodingCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="x")))

    with pytest.raises(CameraUnavailable):
        monitor.run(max_ticks=1)

    assert closed == [True]
