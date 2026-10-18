"""The hourly summary: what it reads, what it skips, and how it survives a dead endpoint."""

from __future__ import annotations

import time

from qwen25vl_agent.llm import LLMError
from qwen25vl_agent.summary import (SUMMARY_KIND, HourlySummarizer, build_prompt,
                                    clean_reply, collect_window)


class _FakeAgent:
    """Stands in for the real agent: records the prompt and returns a canned reply."""

    def __init__(self, reply: str = "", error: str = "") -> None:
        self.reply = reply
        self.error = error
        self.prompts = []
        self.text_client = None
        self.client = self

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise LLMError(self.error)
        return self.reply


class _RaisingClient:
    def __init__(self, message: str) -> None:
        self.message = message

    def ask(self, prompt: str) -> str:
        raise LLMError(self.message)


class _FallbackAgent:
    """A text endpoint that is down, and a vision engine that still answers."""

    def __init__(self) -> None:
        self.text_client = _RaisingClient("text endpoint down")
        self.client = self
        self.prompts = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "回退到视觉引擎后的播报。"


def test_collect_window_counts_and_excludes_previous_summaries(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = 1_700_000_000.0
    with Memory(cfg) as mem:
        mem.episodic.record("scene", "有人坐在书桌前", ts=now - 600)
        mem.episodic.record("alert", "检测到未知人脸", ts=now - 300, importance=0.8)
        mem.episodic.record(SUMMARY_KIND, "上一小时的播报", ts=now - 1800)
        mem.episodic.observe("speech", "我下午要开会", ts=now - 120,
                             payload={"speaker_name": "小明"})
        mem.episodic.record("scene", "很久以前", ts=now - 7200)      # outside the window
        context = collect_window(mem, now)

    assert context["items"] == 3                                     # scene + alert + speech
    assert [s["summary"] for s in context["scenes"]] == ["有人坐在书桌前"]
    assert context["alerts"][0]["summary"] == "检测到未知人脸"
    assert context["speech"][0]["speaker"] == "小明"
    assert context["window_minutes"] == 60


def test_collect_window_lists_people_seen_in_the_window(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = time.time()
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        mem.person.touch(person.id, ts=now - 60)
        context = collect_window(mem, now)
    assert [p["name"] for p in context["people"]] == ["小明"]


def test_a_quiet_window_is_not_announced(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    agent = _FakeAgent(reply="不该被调用")
    with Memory(cfg) as mem:
        result = HourlySummarizer(cfg, agent).summarize(mem, time.time())
    assert result.ok and result.quiet
    assert result.text == ""
    assert agent.prompts == []          # a quiet hour must not cost a model call


def test_summary_is_cleaned_and_prompt_carries_the_scene(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = 1_700_000_000.0
    agent = _FakeAgent(reply='  “过去一小时有人坐在书桌前，没有异常。”  ')
    with Memory(cfg) as mem:
        mem.episodic.record("scene", "有人坐在书桌前", ts=now - 60)
        result = HourlySummarizer(cfg, agent).summarize(mem, now)

    assert result.ok and not result.quiet
    assert result.text == "过去一小时有人坐在书桌前，没有异常。"
    assert result.items == 1
    assert "【画面描述】" in agent.prompts[0]
    assert "有人坐在书桌前" in agent.prompts[0]


def test_a_dead_endpoint_degrades_instead_of_raising(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = time.time()
    agent = _FakeAgent(error="connection refused")
    with Memory(cfg) as mem:
        mem.episodic.record("scene", "有人出现", ts=now - 30)
        result = HourlySummarizer(cfg, agent).summarize(mem, now)

    assert not result.ok
    assert "connection refused" in result.error


def test_the_vision_engine_is_the_fallback_for_the_text_endpoint(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = time.time()
    agent = _FallbackAgent()
    with Memory(cfg) as mem:
        mem.episodic.record("scene", "有人出现", ts=now - 30)
        result = HourlySummarizer(cfg, agent).summarize(mem, now)

    assert result.ok
    assert result.text == "回退到视觉引擎后的播报。"


def test_an_empty_reply_is_a_failure(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = time.time()
    with Memory(cfg) as mem:
        mem.episodic.record("scene", "有人出现", ts=now - 30)
        result = HourlySummarizer(cfg, _FakeAgent(reply="   ")).summarize(mem, now)
    assert not result.ok
    assert "empty summary" in result.error


def test_build_prompt_truncates_a_very_busy_hour():
    context = {
        "window_minutes": 60, "items": 999, "people": [],
        "scenes": [{"time": "12:00", "summary": "x" * 200} for _ in range(200)],
        "events": [], "alerts": [], "speech": [],
    }
    prompt = build_prompt(context)
    assert len(prompt) <= 4000


def test_clean_reply_strips_fences_and_quotes():
    assert clean_reply("```\n你好。\n```") == "你好。"
    assert clean_reply('"你好。"') == "你好。"
    assert clean_reply("  你\n好  ") == "你 好"


# -- the three timestamped streams ----------------------------------------------------
def test_collect_window_includes_faces_with_times(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = 1_700_000_000.0
    with Memory(cfg) as mem:
        mem.episodic.record("face", "画面中出现：小明", ts=now - 900)
        mem.episodic.record("face", "画面中出现：2张未知人脸", ts=now - 60)
        mem.episodic.record("scene", "房间空无一人", ts=now - 30)
        context = collect_window(mem, now)

    assert context["items"] == 3
    assert [f["summary"] for f in context["faces"]] == ["画面中出现：小明",
                                                        "画面中出现：2张未知人脸"]
    assert all(f["time"] for f in context["faces"])     # each face row carries its time


def test_build_prompt_has_a_face_section():
    context = {"window_minutes": 60, "items": 1, "people": [],
               "scenes": [], "events": [], "alerts": [], "speech": [],
               "faces": [{"time": "15:02", "summary": "画面中出现：小明"}]}
    prompt = build_prompt(context)
    assert "【人脸】" in prompt
    assert "15:02 画面中出现：小明" in prompt


def test_speech_comes_from_the_newest_end_of_a_busy_hour(make_cfg):
    """With a cap, the tail of an hour is more useful than its first few utterances."""
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = 1_700_000_000.0
    with Memory(cfg) as mem:
        for i in range(6):
            mem.episodic.observe("speech", f"第{i}句", ts=now - 600 + i * 10,
                                 payload={"speaker_name": "小明"})
        context = collect_window(mem, now, max_items=3)

    assert [s["text"] for s in context["speech"]] == ["第3句", "第4句", "第5句"]


def test_a_face_only_hour_is_still_announced(make_cfg):
    from qwen25vl_agent.memory import Memory

    cfg = make_cfg()
    now = 1_700_000_000.0
    agent = _FakeAgent(reply="这一小时有人出现在画面里。")
    with Memory(cfg) as mem:
        mem.episodic.record("face", "画面中出现：小明", ts=now - 60)
        result = HourlySummarizer(cfg, agent).summarize(mem, now)

    assert result.ok and not result.quiet
    assert result.items == 1
    assert "【人脸】" in agent.prompts[0]
