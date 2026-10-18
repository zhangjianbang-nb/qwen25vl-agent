"""The hourly spoken summary.

Once an hour the monitor distills the last hour of memory into a few sentences and, when TTS is
on, reads them aloud.  Three timestamped streams feed it, which is the whole point of the pass:

* **faces** -- ``face`` episodes, written by the monitor only when the set of people in frame
  changes, so an hour of one person sitting still costs one row;
* **speech** -- ``speech`` observations, each carrying its time, language and speaker;
* **scenes** -- ``scene`` episodes, the Qwen2.5-VL descriptions of what the camera saw.

Two things make this pass different from the per-tick agent call:

* it is text-only, so it can point at a cheaper text model (``llm.text_url`` -- the local
  Qwen3.5-4B) instead of the vision engine that describes frames;
* its own output is written back to episodic memory under a distinct kind, and that kind is
  excluded from the next window -- otherwise every summary would summarize the previous summary,
  and a quiet hour could never read as quiet.

Everything here degrades rather than raises: a dead text endpoint falls back to the vision
engine, and a window with nothing in it produces no utterance at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import Config
from .llm import LLMError

__all__ = ["SUMMARY_KIND", "DEFAULT_INSTRUCTION", "HourSummary", "HourlySummarizer",
           "collect_window", "build_prompt", "clean_reply"]

# Its own kind, so the next window can exclude it (see the module docstring).
SUMMARY_KIND = "hourly_summary"

DEFAULT_INSTRUCTION = (
    "你是监控播报员。下面给你过去一小时的监控记忆，分三类且都带时间：出现的人脸、听到的语音、"
    "画面的描述。请用 2-4 句口语化中文总结这一小时看到和听到的情况，适合直接语音播报。"
    "只讲记忆里确实出现的内容，不要编造人名或事件；"
    "如果这一小时很平静，就直说没有异常。不要输出 JSON，不要加标题或列表符号，只输出要播报的话。"
)

# A 4B model on CPU answers faster from a tidy list than from raw JSON, and the whole prompt is
# capped so a busy hour cannot turn a one-second summary into a minute of prefill.
_PROMPT_CHAR_BUDGET = 4000


@dataclass
class HourSummary:
    """The outcome of one hourly pass."""

    ts: float = 0.0
    text: str = ""
    ok: bool = False
    error: str = ""
    items: int = 0            # episodes/speech/alerts that fed the summary
    window_s: float = 0.0
    quiet: bool = False       # nothing happened, so there was nothing to say

    def as_dict(self) -> dict:
        return {
            "ts": self.ts, "text": self.text, "ok": self.ok, "error": self.error,
            "items": self.items, "window_s": self.window_s, "quiet": self.quiet,
        }


def _clock(ts: float) -> str:
    import time

    return time.strftime("%H:%M", time.localtime(ts))


def collect_window(memory, now: float, *, window_s: float = 3600.0,
                   max_items: int = 200) -> Dict[str, Any]:
    """Gather the last ``window_s`` of memory into a small, prompt-ready slice."""
    since = now - window_s

    episodes = [e for e in memory.episodic.recent(limit=max_items, since=since)
                if e.kind != SUMMARY_KIND]
    episodes.sort(key=lambda e: e.ts)          # recent() is newest-first; a summary reads forward

    scenes = [{"time": _clock(e.ts), "summary": e.summary} for e in episodes if e.kind == "scene"]
    faces = [{"time": _clock(e.ts), "summary": e.summary} for e in episodes if e.kind == "face"]
    events = [{"time": _clock(e.ts), "kind": e.kind, "summary": e.summary}
              for e in episodes if e.kind in ("event", "rule")]
    alerts = [{"time": _clock(e.ts), "summary": e.summary, "importance": round(e.importance, 2)}
              for e in episodes if e.kind == "alert"]

    # Newest speech first, then reversed back to chronological order: with a cap, the tail of a
    # busy hour is more useful than its opening 200 utterances.
    speech_rows = memory.episodic.observations(since=since, kind="speech", limit=max_items,
                                               desc=True)
    speech = [{"time": _clock(o["ts"]), "text": o["text"],
               "speaker": (o["payload"] or {}).get("speaker_name", "unknown")}
              for o in reversed(speech_rows) if o["text"]]

    people = [{"name": p.name, "seen_count": p.seen_count}
              for p in memory.person.all()
              if (p.last_seen or 0.0) >= since]

    items = len(scenes) + len(faces) + len(events) + len(alerts) + len(speech)
    return {
        "window_minutes": int(round(window_s / 60.0)),
        "since": since,
        "items": items,
        "people": people,
        "scenes": scenes,
        "faces": faces,
        "events": events,
        "alerts": alerts,
        "speech": speech,
    }


def build_prompt(context: Dict[str, Any]) -> str:
    """Render the window as a tidy Chinese list -- the shape a small model summarizes best.

    Every section is read with ``.get`` so a partial context renders what it has instead of
    raising: this function is the last step before a model call, and a KeyError here would turn
    a summary into a crash.
    """
    parts: List[str] = [DEFAULT_INSTRUCTION, ""]
    parts.append(f"【时间窗口】过去 {context.get('window_minutes', 60)} 分钟")

    people = context.get("people") or []
    if people:
        roster = "、".join(f"{p['name']}(出现{p['seen_count']}次)" for p in people)
        parts.append(f"【出现的人】{roster}")

    faces = context.get("faces") or []
    if faces:
        lines = [f"- {f['time']} {f['summary']}" for f in faces]
        parts.append("【人脸】\n" + "\n".join(lines))

    scenes = context.get("scenes") or []
    if scenes:
        lines = [f"- {s['time']} {s['summary']}" for s in scenes]
        parts.append("【画面描述】\n" + "\n".join(lines))

    events = context.get("events") or []
    if events:
        lines = [f"- {e['time']} [{e['kind']}] {e['summary']}" for e in events]
        parts.append("【事件】\n" + "\n".join(lines))

    speech = context.get("speech") or []
    if speech:
        lines = [f"- {s['time']} {s['speaker']}: {s['text']}" for s in speech]
        parts.append("【听到的话】\n" + "\n".join(lines))

    alerts = context.get("alerts") or []
    if alerts:
        lines = [f"- {a['time']} {a['summary']}(重要度{a['importance']})" for a in alerts]
        parts.append("【提醒】\n" + "\n".join(lines))

    parts.append("请用 2-4 句中文口语播报这一小时。")
    prompt = "\n\n".join(parts)
    return prompt[:_PROMPT_CHAR_BUDGET]


class HourlySummarizer:
    """Turns one hour of memory into a spoken sentence or two."""

    def __init__(self, cfg: Config, agent) -> None:
        self.cfg = cfg
        self.agent = agent

    def summarize(self, memory, now: float, *, window_s: Optional[float] = None) -> HourSummary:
        window_s = self.cfg.summary.window_s if window_s is None else window_s
        context = collect_window(memory, now, window_s=window_s,
                                 max_items=self.cfg.summary.max_items)
        items = context["items"]

        if items < self.cfg.summary.min_items:
            # A quiet hour is a valid answer, and saying nothing is better than saying "nothing
            # happened" every hour.
            return HourSummary(ts=now, ok=True, items=items, window_s=window_s, quiet=True)

        try:
            reply = self._ask(build_prompt(context))
        except LLMError as exc:
            return HourSummary(ts=now, ok=False, error=str(exc), items=items, window_s=window_s)

        text = clean_reply(reply)
        if not text:
            return HourSummary(ts=now, ok=False, error="empty summary", items=items,
                               window_s=window_s)
        return HourSummary(ts=now, text=text, ok=True, items=items, window_s=window_s)

    def _ask(self, prompt: str) -> str:
        """Ask the text model; fall back to the vision engine if the text endpoint is down."""
        text_client = self.agent.text_client
        try:
            return (text_client or self.agent.client).ask(prompt)
        except LLMError:
            if text_client is None:
                raise
            # The VL engine can still write prose -- a summary with no image is a fine ask for it.
            return self.agent.client.ask(prompt)


def clean_reply(reply: str) -> str:
    """Strip fences, quotes and stray whitespace from a model's prose."""
    text = (reply or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    text = text.strip().strip('"').strip("“”").strip()
    return " ".join(text.split())
