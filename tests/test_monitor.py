"""The monitoring loop, driven with fake hardware so the tick pipeline is testable without a
camera, a microphone, or a running model server.

The point of these tests is the wiring: does a tick sample a frame, match faces against person
memory, call the agent, write memory, evaluate rules, and keep going after a failure?
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from qwen25vl_agent.agent import Decision
from qwen25vl_agent.memory import Memory
from qwen25vl_agent.monitor import Monitor, _Utterance
from qwen25vl_agent.perception.face import FaceDetection
from qwen25vl_agent.perception.video import CameraUnavailable, Frame
from qwen25vl_agent.summary import HourSummary


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
        # Scoped to the rule under test: the seeded ``night_intruder`` rule also fires on an
        # unknown face, so a bare "count the alerts" assertion would depend on the wall clock.
        alerts = [e for e in check.episodic.recent(limit=20)
                  if e.kind == "alert" and e.detail.get("rule") == "unknown_face"]
        assert len(alerts) == 1                 # the 120 s cooldown holds across the two ticks
        assert "陌生人" in alerts[0].summary
        assert alerts[0].importance == pytest.approx(0.8)


class AlternatingFace:
    """Returns a different detection set on each call, so presence changes between ticks."""

    def __init__(self, sets):
        self._sets = list(sets)
        self._i = 0

    @staticmethod
    def models_present(detector_model, recognizer_model):
        return True

    def available(self):
        return True

    def analyze(self, rgb):
        current = self._sets[min(self._i, len(self._sets) - 1)]
        self._i += 1
        return list(current)


def test_face_presence_is_recorded_once_per_appearance(monitor_cfg):
    """An unchanged face across several ticks is one appearance, not one row per tick."""
    detection = FaceDetection(box=(10, 10, 60, 60), score=0.9,
                              embedding=np.ones(128, dtype=np.float32))
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([detection]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="有人")))

    assert monitor.run(max_ticks=3) == 3

    with _reopen(monitor_cfg) as check:
        faces = [e for e in check.episodic.recent(limit=20) if e.kind == "face"]
        assert len(faces) == 1
        assert "未知人脸" in faces[0].summary
        assert faces[0].ts > 0              # the row carries the time, which the summary reads


def test_face_presence_episodes_follow_the_changes(make_cfg):
    """Appeared, left, appeared again -- two presences, two rows when merging is disabled."""
    cfg = make_cfg(monitor={"tick_interval": 0.01, "describe_interval": 0.0,
                            "min_describe_gap": 0.0, "face_presence_merge_s": 0.0})
    vec = np.ones(128, dtype=np.float32)
    mem = Memory(cfg, frames_dir=cfg.frames_dir)
    person = mem.person.ensure("小明")
    mem.person.add_face(person, vec)

    seen = FaceDetection(box=(10, 10, 60, 60), score=0.9, embedding=vec.copy())
    face = AlternatingFace([[seen], [], [seen]])
    monitor = Monitor(cfg, memory=mem, camera=FakeCamera(), face=face,
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="有人")))

    assert monitor.run(max_ticks=3) == 3

    with _reopen(cfg) as check:
        faces = [e for e in check.episodic.recent(limit=20) if e.kind == "face"]
        assert len(faces) == 2
        assert all("小明" in episode.summary for episode in faces)


def test_a_flickering_face_merges_into_one_episode(monitor_cfg):
    """Vanish-and-return inside the merge window keeps extending one row instead of stacking.

    A detector that drops a frame here and there used to leave a trail of identical
    "画面中出现：..." rows, which then filled the whole episode budget in the agent's context.
    """
    vec = np.ones(128, dtype=np.float32)
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    person = mem.person.ensure("小明")
    mem.person.add_face(person, vec)

    seen = FaceDetection(box=(10, 10, 60, 60), score=0.9, embedding=vec.copy())
    face = AlternatingFace([[seen], [], [seen], []])
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=face,
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="有人")))

    assert monitor.run(max_ticks=4) == 4

    with _reopen(monitor_cfg) as check:
        faces = [e for e in check.episodic.recent(limit=20) if e.kind == "face"]
        assert len(faces) == 1
        assert faces[0].t_end is not None
        assert faces[0].t_end > faces[0].ts        # the row grew instead of being rewritten


def test_the_same_person_detected_twice_is_one_identity(monitor_cfg):
    """Two boxes of the same person in one frame must not become two people in the prompt."""
    vec = np.ones(128, dtype=np.float32)
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    person = mem.person.ensure("小明")
    mem.person.add_face(person, vec)

    first = FaceDetection(box=(10, 10, 60, 60), score=0.9, embedding=vec.copy())
    second = FaceDetection(box=(90, 10, 60, 60), score=0.8, embedding=vec.copy())
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(),
                      face=FakeFace([first, second]), asr=None, speaker=None,
                      agent=FakeAgent(Decision(scene="有人")))

    identities = monitor._identities([first, second], time.time())

    assert len(identities) == 1
    assert identities[0]["name"] == "小明"
    assert identities[0]["person_id"] == person.id


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


# -- speech reaches the agent -------------------------------------------------
def _push_speech(monitor, text, language="zh", embedding=None):
    monitor._speech_q.put_nowait(_Utterance(
        ts=time.time(), text=text, language=language, emotion="unknown",
        duration_s=1.0, voice_embedding=embedding,
    ))


class CapturingAgent(FakeAgent):
    """Records the observation it was handed, so a test can assert on the prompt's content."""

    def __init__(self):
        super().__init__(Decision(scene="s"))
        self.observations = []

    def decide(self, observation, context):
        self.observations.append(observation)
        return self.decision


def test_speech_is_handed_to_the_agent(monitor_cfg):
    """The bug this guards: speech was captured and stored, but never reached the model."""
    agent = CapturingAgent()
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=agent)
    _push_speech(monitor, "这是一段测试语音", language="zh")

    assert monitor.run(max_ticks=1) == 1

    assert agent.observations, "the agent should have been called"
    speech = agent.observations[0].speech
    assert len(speech) == 1
    assert speech[0]["text"] == "这是一段测试语音"
    assert speech[0]["language"] == "zh"
    # And it shows up in the text the model actually reads.
    assert "这是一段测试语音" in agent.observations[0].as_text()


def test_noise_transcripts_stay_out_of_the_agent_prompt(monitor_cfg):
    """A lone ASR filler is logged, but must not dilute the agent's prompt."""
    agent = CapturingAgent()
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=agent)
    _push_speech(monitor, "Yeah.", language="en")

    assert monitor.run(max_ticks=1) == 1

    assert agent.observations[0].speech == []
    assert monitor.stats["noise"] == 1
    assert monitor.stats["utterances"] == 0
    with _reopen(monitor_cfg) as check:
        # The raw log still keeps it, so the record stays faithful.
        assert check.store.stats()["observations"] == 2      # one frame + one speech


def test_rules_evaluation_survives_speech_in_working_memory(monitor_cfg):
    """``_evaluate_rules`` read ``u.language`` off a WorkingItem that has no such attribute."""
    errors = []
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="s")))
    monitor.on_event(lambda e: errors.append(e.message) if e.kind == "error" else None)
    mem.working.add("speech", "测试语音", language="zh", speaker_name="unknown")

    assert monitor.run(max_ticks=2) == 2
    assert not any("tick failed" in message for message in errors)


def test_monitor_runs_on_a_background_thread(monitor_cfg):
    """``qwen25vl-agent web`` runs the monitor off the main thread.

    The monitor is built here and run there, so its SQLite connection is created in one thread and
    used in another -- which sqlite3 rejects unless the connection allows it.
    """
    errors = []
    mem = Memory(monitor_cfg, frames_dir=monitor_cfg.frames_dir)
    monitor = Monitor(monitor_cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                      asr=None, speaker=None, agent=FakeAgent(Decision(scene="s")))
    monitor.on_event(lambda e: errors.append(e.message) if e.kind == "error" else None)

    result = {}
    thread = threading.Thread(target=lambda: result.update(ticks=monitor.run(max_ticks=2)))
    thread.start()
    thread.join(timeout=10)

    assert result.get("ticks") == 2
    assert not errors, errors


# -- the hourly broadcast clock ----------------------------------------------------------------
def _monitor_for_clock(cfg):
    mem = Memory(cfg, frames_dir=cfg.frames_dir)
    return Monitor(cfg, memory=mem, camera=FakeCamera(), face=FakeFace([]),
                   asr=None, speaker=None, agent=FakeAgent(Decision(scene="s")))


def test_a_failed_broadcast_retries_sooner_than_the_next_hour(monitor_cfg):
    """A failed broadcast must not buy an hour of silence.

    Observed live: the summary timed out once, the clock had already been advanced, and the next
    attempt was a full hour away -- so the user saw no broadcast and no hint of why.
    """
    monitor = _monitor_for_clock(monitor_cfg)
    base = 1000.0

    monitor._note_summary(base, ok=False)
    assert not monitor._summary_due(base + 60.0)                    # not on the very next tick
    assert monitor._summary_due(base + monitor_cfg.summary.retry_s)  # but within minutes

    monitor._note_summary(base + monitor_cfg.summary.retry_s, ok=True)
    assert not monitor._summary_due(base + monitor_cfg.summary.retry_s + 3599.0)
    assert monitor._summary_due(base + monitor_cfg.summary.retry_s + 3600.0)


def test_a_failed_broadcast_is_reported(monitor_cfg):
    monitor = _monitor_for_clock(monitor_cfg)

    class _FailingSummarizer:
        def summarize(self, memory, now, **kwargs):
            return HourSummary(ts=now, ok=False, error="engine down", items=5, window_s=3600.0)

    monitor.summarizer = _FailingSummarizer()
    assert monitor._hourly_summary(1000.0) is False
    assert monitor.stats["summary_errors"] == 1


def test_a_quiet_hour_counts_as_success(monitor_cfg):
    """Nothing worth saying is not a failure -- retrying it in five minutes would not help."""
    monitor = _monitor_for_clock(monitor_cfg)

    class _QuietSummarizer:
        def summarize(self, memory, now, **kwargs):
            return HourSummary(ts=now, ok=True, items=0, window_s=3600.0, quiet=True)

    monitor.summarizer = _QuietSummarizer()
    assert monitor._hourly_summary(1000.0) is True
    assert monitor.stats["summary_errors"] == 0
    assert monitor.stats["summaries"] == 0        # a quiet hour is not a broadcast
