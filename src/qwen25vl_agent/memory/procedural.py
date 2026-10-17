"""Procedural memory: the monitor's rules.

Procedural memory is "how this house is watched": a rule is a trigger plus an action, stored as
JSON so new trigger kinds need no schema change.  The agent does not *invent* rules at runtime
-- it evaluates the ones the user configured, which keeps a 24/7 monitor predictable.  The
model's job is to describe and summarize; deciding when to alert is deterministic code.

Supported triggers:

``unknown_face``    an unrecognised face this tick (``min_count``)
``known_person``    a specific person (``name``), or any known person (``name: "any"``)
``speech_keyword``  any of ``keywords`` appears in a transcript
``person_absent``   ``name`` has not been seen for ``seconds``
``no_speech_for``   no speech for ``seconds``

Any trigger may carry a ``when`` gate, e.g. ``{"after": "22:00", "before": "06:00"}`` (a window
that crosses midnight is handled).

Actions:

``alert``     log/post a message with an importance
``remember``  write a semantic fact
``snapshot``  mark the tick's frame as worth keeping

Message templates may use ``{time}``, ``{name}``, ``{text}``, ``{count}`` and ``{rule}``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .store import MemoryStore

__all__ = ["Rule", "TickContext", "FiredRule", "ProceduralMemory"]


@dataclass
class Rule:
    id: int
    name: str
    description: str
    trigger: dict
    action: dict
    enabled: bool
    cooldown_s: float

    def as_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "trigger": self.trigger, "action": self.action,
            "enabled": self.enabled, "cooldown_s": self.cooldown_s,
        }


@dataclass
class TickContext:
    """Everything a rule can look at for one tick of the monitor."""

    ts: float
    identities: List[dict] = field(default_factory=list)   # IdentityMatch.as_dict()
    speech: List[dict] = field(default_factory=list)        # transcript dicts
    last_speech_ts: Optional[float] = None
    person_last_seen: Dict[str, float] = field(default_factory=dict)  # name -> ts
    started_ts: Optional[float] = None   # when the monitor started; see `no_speech_for`


@dataclass
class FiredRule:
    rule_name: str
    action: dict
    message: str
    importance: float
    ts: float

    def as_dict(self) -> dict:
        return {
            "rule": self.rule_name, "action": self.action.get("type", "alert"),
            "message": self.message, "importance": round(self.importance, 3),
            "ts": self.ts,
        }


def _row_to_rule(row) -> Rule:
    return Rule(
        id=int(row["id"]), name=row["name"], description=row["description"] or "",
        trigger=MemoryStore.loads(row["trigger"], {}), action=MemoryStore.loads(row["action"], {}),
        enabled=bool(row["enabled"]), cooldown_s=float(row["cooldown_s"] or 0.0),
    )


def _in_window(when: Optional[dict], ts: float) -> bool:
    """True when ``ts`` falls inside a ``{"after": HH:MM, "before": HH:MM}`` gate."""
    if not when:
        return True
    moment = datetime.fromtimestamp(ts).time()
    after = when.get("after")
    before = when.get("before")

    def _parse(value: str):
        hh, mm = (int(p) for p in value.split(":"))
        return hh * 60 + mm

    now = moment.hour * 60 + moment.minute
    if after is not None and before is not None:
        a, b = _parse(after), _parse(before)
        if a <= b:
            return a <= now < b
        return now >= a or now < b        # window crosses midnight
    if after is not None:
        return now >= _parse(after)
    if before is not None:
        return now < _parse(before)
    return True


class ProceduralMemory:
    """Rule storage plus the evaluator the monitor calls each tick."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    # -- storage -------------------------------------------------------------------
    def add(self, name: str, trigger: dict, action: dict, *, description: str = "",
            cooldown_s: float = 60.0, enabled: bool = True) -> Rule:
        self.store.execute(
            "INSERT OR REPLACE INTO procedures(name, description, trigger, action, enabled,"
            " cooldown_s, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, description, MemoryStore.dumps(trigger), MemoryStore.dumps(action),
             int(bool(enabled)), float(cooldown_s), time.time()),
        )
        return self.get(name)  # type: ignore[return-value]

    def get(self, name: str) -> Optional[Rule]:
        row = self.store.query_one("SELECT * FROM procedures WHERE name = ?", (name,))
        return _row_to_rule(row) if row is not None else None

    def all(self, enabled_only: bool = False) -> List[Rule]:
        sql = "SELECT * FROM procedures"
        if enabled_only:
            sql += " WHERE enabled = 1"
        return [_row_to_rule(r) for r in self.store.query(sql + " ORDER BY name")]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        cur = self.store.execute(
            "UPDATE procedures SET enabled = ?, updated_at = ? WHERE name = ?",
            (int(bool(enabled)), time.time(), name),
        )
        return bool(cur.rowcount)

    def remove(self, name: str) -> bool:
        cur = self.store.execute("DELETE FROM procedures WHERE name = ?", (name,))
        return bool(cur.rowcount)

    def seed_defaults(self) -> int:
        """Insert the starter rule set if the table is empty.  Returns how many were added."""
        if self.all():
            return 0
        defaults = [
            ("unknown_face", {"type": "unknown_face", "min_count": 1},
             {"type": "alert", "importance": 0.8,
              "message": "陌生人出现在画面中 ({time})"}, 120.0,
             "An unrecognised face is present."),
            ("night_intruder", {"type": "unknown_face", "min_count": 1,
                                "when": {"after": "22:00", "before": "06:00"}},
             {"type": "alert", "importance": 0.95,
              "message": "夜间检测到陌生人 ({time})"}, 300.0,
             "An unrecognised face at night is treated as an intruder."),
            ("distress_keyword", {"type": "speech_keyword",
                                  "keywords": ["救命", "报警", "help", "emergency"]},
             {"type": "alert", "importance": 0.95,
              "message": "听到求助/报警词：{text}"}, 60.0,
             "A spoken distress keyword."),
            ("long_silence", {"type": "no_speech_for", "seconds": 3600},
             {"type": "remember", "subject": "room", "predicate": "was", "object": "quiet",
              "confidence": 0.3}, 1800.0,
             "The room has been quiet for an hour."),
        ]
        added = 0
        for name, trigger, action, cooldown, description in defaults:
            self.add(name, trigger, action, description=description, cooldown_s=cooldown)
            added += 1
        return added

    # -- cooldown bookkeeping -------------------------------------------------------
    def _last_fired(self, name: str) -> float:
        row = self.store.query_one("SELECT value FROM meta WHERE key = ?", (f"rule_last:{name}",))
        return float(row["value"]) if row else 0.0

    def _mark_fired(self, name: str, ts: float) -> None:
        self.store.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (f"rule_last:{name}", str(ts)),
        )

    # -- evaluation -----------------------------------------------------------------
    def evaluate(self, ctx: TickContext) -> List[FiredRule]:
        """Return the rules that fired for this tick, honoring per-rule cooldowns."""
        fired: List[FiredRule] = []
        for rule in self.all(enabled_only=True):
            if not _in_window(rule.trigger.get("when"), ctx.ts):
                continue
            if ctx.ts - self._last_fired(rule.name) < rule.cooldown_s:
                continue
            match = self._match(rule.trigger, ctx)
            if match is None:
                continue
            self._mark_fired(rule.name, ctx.ts)
            message = self._format(rule, ctx, match)
            importance = float(rule.action.get("importance", 0.5))
            fired.append(FiredRule(rule.name, rule.action, message, importance, ctx.ts))
        return fired

    def _match(self, trigger: dict, ctx: TickContext) -> Optional[dict]:
        kind = trigger.get("type")
        if kind == "unknown_face":
            unknown = [i for i in ctx.identities if not i.get("known")]
            if len(unknown) >= int(trigger.get("min_count", 1)):
                return {"count": len(unknown), "name": "unknown"}
            return None
        if kind == "known_person":
            want = trigger.get("name", "any")
            known = [i for i in ctx.identities if i.get("known")]
            if want != "any":
                known = [i for i in known if i.get("name") == want]
            if known:
                return {"name": known[0].get("name", "unknown"), "count": len(known)}
            return None
        if kind == "speech_keyword":
            keywords = [str(k) for k in trigger.get("keywords", [])]
            for utterance in ctx.speech:
                text = str(utterance.get("text", ""))
                for kw in keywords:
                    if kw and kw in text:
                        return {"text": text, "keyword": kw,
                                "name": utterance.get("speaker_name", "unknown")}
            return None
        if kind == "person_absent":
            name = trigger.get("name")
            seconds = float(trigger.get("seconds", 600))
            last = ctx.person_last_seen.get(str(name))
            if last is not None and (ctx.ts - last) >= seconds:
                return {"name": name, "seconds": ctx.ts - last}
            return None
        if kind == "no_speech_for":
            seconds = float(trigger.get("seconds", 600))
            # "No speech for N seconds" needs a reference point.  If nothing has been heard
            # yet, the monitor's own start time is the honest one -- otherwise a freshly
            # started agent would fire a "quiet for an hour" rule on its very first tick.
            reference = ctx.last_speech_ts if ctx.last_speech_ts is not None else ctx.started_ts
            if reference is None:
                return None
            if (ctx.ts - reference) >= seconds:
                return {"seconds": ctx.ts - reference}
            return None
        return None

    def _format(self, rule: Rule, ctx: TickContext, match: dict) -> str:
        template = str(rule.action.get("message", rule.name))
        fields = {
            "time": datetime.fromtimestamp(ctx.ts).strftime("%H:%M:%S"),
            "date": datetime.fromtimestamp(ctx.ts).strftime("%Y-%m-%d"),
            "rule": rule.name,
            **{k: (f"{v:.0f}" if isinstance(v, float) else v) for k, v in match.items()},
        }
        try:
            return template.format(**fields)
        except (KeyError, IndexError):
            return template
