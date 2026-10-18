"""Text-to-speech: backend selection, text capping, and failure handling.

Nothing here touches real audio hardware: synthesis and playback are monkeypatched, so the tests
assert the *contract* -- that a bad backend degrades instead of raising, and that the text handed
to a backend is collapsed and capped.
"""

from __future__ import annotations

import shutil

from qwen25vl_agent.tts import TTSEngine, detect_backend, detect_player


def test_an_explicit_backend_is_passed_through():
    assert detect_backend("none") == "none"
    assert detect_backend("command") == "command"
    assert detect_backend("spd-say") == "spd-say"


def test_auto_resolves_to_a_real_backend():
    assert detect_backend("auto") in ("piper", "edge", "espeak", "spd-say", "none")


def test_an_unknown_player_is_not_invented():
    assert detect_player("definitely-not-a-player") is None


def test_engine_with_backend_none_is_unavailable():
    engine = TTSEngine(backend="none")
    assert not engine.available()
    assert engine.say("hello") is False
    assert engine.say_async("hello") is False


def test_command_backend_needs_a_template():
    assert not TTSEngine(backend="command").available()
    assert TTSEngine(backend="command", command="say {text} > {out}").available()


def test_text_is_collapsed_and_capped():
    engine = TTSEngine(backend="none", max_chars=10)
    assert engine._clean("a  b\n c") == "a b c"
    assert engine._clean("x" * 50) == "x" * 10 + "……"


def test_empty_text_is_not_synthesized(monkeypatch):
    engine = TTSEngine(backend="command", command="true")
    monkeypatch.setattr(engine, "_synth", lambda text, out: (_ for _ in ()).throw(
        AssertionError("must not synthesize empty text")))
    assert engine.say("   ") is False


def test_say_synthesizes_then_plays(monkeypatch):
    engine = TTSEngine(backend="command", command="true", player="echo")
    seen = []

    def fake_synth(text, out_path):
        with open(out_path, "wb") as fh:
            fh.write(b"audio")
        seen.append(("synth", text))
        return True

    def fake_play(path):
        seen.append(("play", path))
        return True

    monkeypatch.setattr(engine, "_synth", fake_synth)
    monkeypatch.setattr(engine, "_play", fake_play)

    assert engine.say("你好   世界") is True
    assert [step for step, _ in seen] == ["synth", "play"]
    assert seen[0][1] == "你好 世界"      # whitespace collapsed before synthesis


def test_a_failed_synthesis_returns_false_without_raising(monkeypatch):
    engine = TTSEngine(backend="command", command="false")
    monkeypatch.setattr(engine, "_synth", lambda text, out: False)
    assert engine.say("hi") is False


def test_a_missing_player_is_reported_not_raised(monkeypatch):
    engine = TTSEngine(backend="command", command="true")
    engine.player = None

    def fake_synth(text, out_path):
        with open(out_path, "wb") as fh:
            fh.write(b"audio")
        return True

    monkeypatch.setattr(engine, "_synth", fake_synth)
    assert engine.say("hi") is False
    assert "no audio player" in engine.last_error


def test_say_async_survives_a_worker_that_fails(monkeypatch):
    """The worker thread must swallow a backend error and keep serving the next utterance."""
    engine = TTSEngine(backend="command", command="true")
    calls = []

    def fake_say(text):
        calls.append(text)
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(engine, "say", fake_say)
    assert engine.say_async("第一条") is True
    assert engine.say_async("第二条") is True
    for _ in range(200):
        if len(calls) >= 2:
            break
        import time

        time.sleep(0.01)
    engine.close()
    assert calls[:2] == ["第一条", "第二条"]


def test_errors_are_reported_through_the_callback():
    seen = []
    engine = TTSEngine(backend="none", on_error=seen.append)
    engine._error("boom")
    assert engine.last_error == "boom"
    assert seen == ["boom"]


def test_describe_mentions_availability():
    assert "unavailable" in TTSEngine(backend="none").describe()
    described = TTSEngine(backend="command", command="true", player="echo").describe()
    assert "command" in described


def test_piper_requires_the_binary():
    engine = TTSEngine(backend="piper", voice="/nonexistent/voice.onnx")
    assert engine.available() == bool(shutil.which("piper"))
