"""Procedural memory: rule matching, cooldowns, and time windows."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from qwen25vl_agent.memory import Memory, TickContext


def _ts(hour: int, minute: int = 0, day: int = 1) -> float:
    """A fixed wall-clock timestamp so day/night window tests are deterministic."""
    return time.mktime(datetime(2026, 1, day, hour, minute, 0).timetuple())


def test_seed_defaults_inserts_four_rules(cfg):
    with Memory(cfg) as mem:
        names = {r.name for r in mem.procedural.all()}
        assert names == {"unknown_face", "night_intruder", "distress_keyword", "long_silence"}
        # Seeding is once-only.
        assert mem.procedural.seed_defaults() == 0


def test_unknown_face_fires_then_cools_down(cfg):
    with Memory(cfg) as mem:
        ts = _ts(12)
        fired = mem.procedural.evaluate(TickContext(ts=ts, identities=[{"known": False}]))
        assert [f.rule_name for f in fired] == ["unknown_face"]
        assert fired[0].importance == pytest.approx(0.8)
        assert "陌生人" in fired[0].message

        # Inside the 120 s cooldown.
        assert mem.procedural.evaluate(
            TickContext(ts=ts + 10, identities=[{"known": False}])) == []

        # Past it.
        assert [f.rule_name for f in mem.procedural.evaluate(
            TickContext(ts=ts + 121, identities=[{"known": False}]))] == ["unknown_face"]


def test_known_person_rule_named(cfg):
    with Memory(cfg) as mem:
        mem.procedural.add("ming_here", {"type": "known_person", "name": "小明"},
                           {"type": "alert", "importance": 0.2, "message": "{name} 出现了"},
                           cooldown_s=0.0)
        ts = _ts(12)
        fired = {f.rule_name: f for f in mem.procedural.evaluate(
            TickContext(ts=ts, identities=[{"known": True, "name": "小明"}], started_ts=ts))}
        assert "ming_here" in fired
        assert fired["ming_here"].message == "小明 出现了"


def test_known_person_rule_any(cfg):
    with Memory(cfg) as mem:
        mem.procedural.add("any_known", {"type": "known_person", "name": "any"},
                           {"type": "alert", "message": "{name}"}, cooldown_s=0.0)
        ts = _ts(12)
        fired = [f for f in mem.procedural.evaluate(
            TickContext(ts=ts, identities=[{"known": True, "name": "小红"}], started_ts=ts))
            if f.rule_name == "any_known"]
        assert fired and fired[0].message == "小红"


def test_speech_keyword(cfg):
    with Memory(cfg) as mem:
        ts = _ts(12)
        fired = {f.rule_name: f for f in mem.procedural.evaluate(
            TickContext(ts=ts, speech=[{"text": "救命啊，快报警", "speaker_name": "小明"}]))}
        assert "distress_keyword" in fired
        assert "救命啊，快报警" in fired["distress_keyword"].message
        assert fired["distress_keyword"].importance == pytest.approx(0.95)


def test_person_absent(cfg):
    with Memory(cfg) as mem:
        mem.procedural.add("ming_absent",
                           {"type": "person_absent", "name": "小明", "seconds": 600},
                           {"type": "alert", "message": "小明已 {seconds} 秒未出现"},
                           cooldown_s=0.0)
        ts = _ts(12)
        fired = [f for f in mem.procedural.evaluate(
            TickContext(ts=ts, person_last_seen={"小明": ts - 700}, started_ts=ts))
            if f.rule_name == "ming_absent"]
        assert fired
        assert "700" in fired[0].message


def test_no_speech_for_uses_the_monitor_start_time(cfg):
    with Memory(cfg) as mem:
        ts = _ts(12)
        # Nothing has been heard yet, so the monitor's own start time is the reference -- this
        # is what stops a freshly started agent from firing "quiet for an hour" immediately.
        fired = {f.rule_name for f in mem.procedural.evaluate(
            TickContext(ts=ts, started_ts=ts - 4000))}
        assert "long_silence" in fired


def test_no_speech_for_does_not_fire_on_a_fresh_start(cfg):
    with Memory(cfg) as mem:
        ts = _ts(12)
        fired = {f.rule_name for f in mem.procedural.evaluate(TickContext(ts=ts, started_ts=ts))}
        assert "long_silence" not in fired


def test_no_speech_for_needs_a_reference_point(cfg):
    with Memory(cfg) as mem:
        ts = _ts(12)
        fired = {f.rule_name for f in mem.procedural.evaluate(TickContext(ts=ts))}
        assert "long_silence" not in fired


def test_long_silence_action_is_remember(cfg):
    with Memory(cfg) as mem:
        ts = _ts(12)
        fired = [f for f in mem.procedural.evaluate(TickContext(ts=ts, started_ts=ts - 4000))
                 if f.rule_name == "long_silence"]
        assert fired and fired[0].action["type"] == "remember"


def test_when_window_crossing_midnight(cfg):
    with Memory(cfg) as mem:
        mem.procedural.add("night_only",
                           {"type": "unknown_face", "when": {"after": "22:00", "before": "06:00"}},
                           {"type": "alert", "message": "night"}, cooldown_s=0.0)

        def fires(hour: int) -> bool:
            return "night_only" in {f.rule_name for f in mem.procedural.evaluate(
                TickContext(ts=_ts(hour), identities=[{"known": False}]))}

        # Chronological order matters, because cooldowns compare timestamps.
        assert fires(2) is True       # 02:00 is inside a window that crosses midnight
        assert fires(12) is False     # noon is outside
        assert fires(23) is True


def test_enable_disable_remove(cfg):
    with Memory(cfg) as mem:
        assert mem.procedural.set_enabled("unknown_face", False) is True
        assert "unknown_face" not in {r.name for r in mem.procedural.all(enabled_only=True)}

        assert mem.procedural.remove("unknown_face") is True
        assert mem.procedural.get("unknown_face") is None
        assert mem.procedural.remove("unknown_face") is False
        assert mem.procedural.set_enabled("nope", True) is False


def test_add_upserts_by_name(cfg):
    with Memory(cfg) as mem:
        rule = mem.procedural.add("r1", {"type": "unknown_face"},
                                  {"type": "alert", "message": "m"},
                                  description="d", cooldown_s=5.0)
        assert rule.name == "r1"
        assert rule.description == "d"
        assert rule.cooldown_s == 5.0

        mem.procedural.add("r1", {"type": "unknown_face"},
                           {"type": "alert", "message": "m2"}, cooldown_s=9.0)
        assert mem.procedural.get("r1").cooldown_s == 9.0
        assert len([r for r in mem.procedural.all() if r.name == "r1"]) == 1


def test_disabled_rule_never_fires(cfg):
    with Memory(cfg) as mem:
        mem.procedural.set_enabled("unknown_face", False)
        ts = _ts(12)
        assert mem.procedural.evaluate(TickContext(ts=ts, identities=[{"known": False}])) == []
