"""The agent step: turn one observation plus memory into a structured decision.

The model here is a small local Qwen2.5-VL (served by ``qwen25vl-lite``), so the prompt is built
to be answerable by a 3B model rather than to be clever:

* the memory context is a *small* slice (known people, a few recent episodes, a few facts);
* the output contract is a single JSON object with fixed keys, and nothing else;
* every field is optional and an empty list is an explicit, expected answer.

Parsing is defensive.  A 3B model will sometimes wrap JSON in prose or a code fence, so the
parser finds the first balanced object and falls back to treating the whole reply as the scene
description.  A malformed reply degrades the tick; it never raises out of the monitor loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config
from .llm import LLMClient, LLMError

__all__ = ["Observation", "Decision", "Agent", "SYSTEM_PROMPT", "parse_decision"]

SYSTEM_PROMPT = """你是一个 24 小时摄像头监控智能体。每次你会收到：当前画面、人脸识别结果（已知人物名或 unknown）、最近听到的语音转写、以及长期记忆摘要。

只输出一个 JSON 对象，不要输出任何解释或多余文字，格式严格如下：
{
  "scene": "对当前画面的简短客观描述，一句话",
  "people": [{"name": "已知人物名", "action": "此人正在做什么"}],
  "events": [{"kind": "事件类型", "summary": "一句话描述", "importance": 0.0}],
  "remember": [{"subject": "主语", "predicate": "关系", "object": "宾语", "confidence": 0.0}],
  "notes": [{"name": "人物名", "text": "值得记入该人物档案的一句话"}]
}

规则：
- scene 只描述【当前感知】这一帧里你确实看到的东西：物体、位置、动作、光线、画面清晰度。
  禁止复述【近期情景记忆】里的句子，也不要写"出现了几张人脸"这种识别结果；
- people 只填【已知人物】里出现过的名字（最多 5 个），不要写 unknown，不要重复；
  没有已知人物时返回空数组 []；
- 没有事件时 events 返回空数组 []；没有可记忆内容时 remember / notes 返回空数组；
- importance 取值 0.0~1.0，越高越需要提醒（0.0 表示日常，0.9 表示异常）；
- 输出要短，只输出 JSON，不要加 ``` 代码块。"""


@dataclass
class Observation:
    """One tick's perception, already reduced to what the model needs."""

    ts: float
    frame_jpeg: Optional[bytes] = None
    faces: List[Dict[str, Any]] = field(default_factory=list)
    speech: List[Dict[str, Any]] = field(default_factory=list)
    frame_path: Optional[str] = None
    note: str = ""

    def as_text(self) -> str:
        lines = ["【当前感知】"]
        if self.faces:
            described = []
            for face in self.faces:
                name = face.get("name", "unknown")
                known = "已知" if face.get("known") else "未知"
                described.append(f"{name}({known}, 相似度{face.get('score', 0):.2f})")
            lines.append("人脸：" + "、".join(described))
        else:
            lines.append("人脸：无")
        if self.speech:
            for utterance in self.speech:
                who = utterance.get("speaker_name") or "unknown"
                lines.append(
                    f"语音[{utterance.get('language', '?')}] {who}: {utterance.get('text', '')}"
                )
        else:
            lines.append("语音：无")
        if self.note:
            lines.append(f"备注：{self.note}")
        return "\n".join(lines)


@dataclass
class Decision:
    """The parsed reply.  ``ok`` is False when the model's output could not be parsed."""

    scene: str = ""
    people: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    remember: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    ok: bool = True
    error: str = ""

    @property
    def importance(self) -> float:
        if not self.events:
            return 0.0
        return max(float(e.get("importance", 0.0) or 0.0) for e in self.events)

    def as_dict(self) -> dict:
        return {
            "scene": self.scene, "people": self.people, "events": self.events,
            "remember": self.remember, "notes": self.notes, "ok": self.ok, "error": self.error,
        }


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _first_json_object(text: str) -> Optional[str]:
    """Return the balanced ``{...}`` block that starts at the first ``{``, or ``None``.

    Only the first ``{`` is tried.  The output contract is a single JSON object, so if that object
    is unterminated -- a reply cut off by ``max_tokens`` -- the honest answer is "no JSON", not a
    *nested* fragment that happens to be balanced.  Returning the nested piece would parse fine
    and hand back an empty scene, silently discarding the description instead of reporting it.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_decision(text: str) -> Decision:
    """Parse a model reply into a :class:`Decision`, never raising."""
    raw = (text or "").strip()
    candidate = None

    fence = _FENCE_RE.search(raw)
    if fence:
        candidate = fence.group(1).strip()
    if candidate is None:
        candidate = _first_json_object(raw)
    if candidate is None:
        if "{" in raw:
            # An object was opened but never closed: the reply was truncated by ``max_tokens``.
            # Report it instead of pretending the tick succeeded with nothing recorded.
            return Decision(scene="", raw=raw, ok=False,
                            error="incomplete JSON object (reply likely truncated)")
        # No JSON at all: keep the prose as the scene so the tick still carries information.
        return Decision(scene=raw, raw=raw, ok=False, error="no JSON object in reply")

    try:
        data = json.loads(candidate)
    except ValueError as exc:
        truncated = candidate.lstrip().startswith("{") and not candidate.rstrip().endswith("}")
        return Decision(
            scene="" if truncated else raw, raw=raw, ok=False,
            error="incomplete JSON object (reply likely truncated)" if truncated
            else f"invalid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return Decision(scene=raw, raw=raw, ok=False, error="JSON was not an object")

    def _list(key: str) -> List[Dict[str, Any]]:
        value = data.get(key) or []
        if isinstance(value, dict):
            return [value]
        return [v for v in value if isinstance(v, dict)]

    scene = data.get("scene") or data.get("description") or ""
    return Decision(
        scene=str(scene).strip(),
        people=_list("people"),
        events=_list("events"),
        remember=_list("remember"),
        notes=_list("notes"),
        raw=raw,
        ok=True,
    )


class Agent:
    """Builds the prompt, calls the local Qwen model, and parses the decision back."""

    def __init__(self, cfg: Config, client: Optional[LLMClient] = None) -> None:
        self.cfg = cfg
        self.client = client or LLMClient(
            url=cfg.llm.url,
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.timeout,
            max_vision_tokens=cfg.llm.max_vision_tokens,
        )
        self._text_client: Optional[LLMClient] = None
        if cfg.llm.text_url:
            from .llm import text_client

            self._text_client = text_client(cfg.llm.text_url, cfg.llm.text_model,
                                            max_tokens=cfg.llm.max_tokens,
                                            timeout=cfg.llm.timeout)

    # -- prompt --------------------------------------------------------------------
    @property
    def text_client(self) -> Optional[LLMClient]:
        """The text-only endpoint (``llm.text_url``), or None when only vision is configured."""
        return self._text_client

    def build_prompt(self, observation: Observation, context: Dict[str, Any]) -> str:
        parts: List[str] = []

        people = context.get("known_people") or []
        if people:
            roster = "、".join(
                f"{p['name']}(见过{p.get('seen_count', 0)}次)" for p in people[:20]
            )
            parts.append(f"【已知人物】{roster}")
        else:
            parts.append("【已知人物】暂无")

        episodes = context.get("recent_episodes") or []
        if episodes:
            lines = [f"- {e['summary']}" for e in episodes[:8]]
            parts.append("【近期情景记忆】\n" + "\n".join(lines))

        facts = context.get("facts") or []
        if facts:
            parts.append("【语义记忆】" + "；".join(facts[:12]))

        parts.append(observation.as_text())
        parts.append("请按系统提示的 JSON 格式输出。")
        return "\n\n".join(parts)

    # -- main call ------------------------------------------------------------------
    def decide(self, observation: Observation, context: Dict[str, Any],
               *, max_tokens: Optional[int] = None) -> Decision:
        """One reasoning step.  Returns an unparsed ``Decision`` on transport errors too."""
        prompt = self.build_prompt(observation, context)
        try:
            reply = self.client.ask(prompt, observation.frame_jpeg,
                                    system=SYSTEM_PROMPT, max_tokens=max_tokens)
        except LLMError as exc:
            return Decision(scene="", raw="", ok=False, error=str(exc))
        return parse_decision(reply)

    def consolidate(self, context: Dict[str, Any], *, instruction: str = "") -> str:
        """Optional text-only pass over memory (needs ``llm.text_url``).

        Used to merge overlapping facts or write a daily summary; it takes no image, so it can
        point at a cheaper text endpoint than the VL engine.
        """
        client = self._text_client or self.client
        summary = json.dumps(context, ensure_ascii=False)[:6000]
        prompt = (
            (instruction or "请把下面的监控记忆整理成 3-5 条要点，用中文。")
            + "\n\n记忆：\n" + summary
        )
        return client.ask(prompt)
