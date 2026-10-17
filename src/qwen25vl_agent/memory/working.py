"""Working memory: the last few minutes of perception, in process.

This is the short-term buffer the agent reads from on every reasoning step -- recent scene
descriptions, faces, and utterances -- and it is deliberately *not* persisted.  What survives
a restart is the episodic/semantic/person memory; working memory is the scratchpad that makes
"what just happened" cheap to answer without querying the database every tick.

Two eviction rules apply: a size cap (the newest ``size`` items) and a TTL (items older than
``ttl_s`` are dropped on read).  The TTL is what keeps the agent's context window from slowly
filling with yesterday's observations.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

__all__ = ["WorkingItem", "WorkingMemory"]


@dataclass
class WorkingItem:
    ts: float
    kind: str                 # 'scene' | 'face' | 'speech' | 'tick' | 'event'
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ts": round(self.ts, 3), "kind": self.kind, "text": self.text, **self.data}


class WorkingMemory:
    """A TTL ring buffer of recent observations."""

    def __init__(self, size: int = 40, ttl_s: float = 300.0) -> None:
        self.size = max(1, int(size))
        self.ttl_s = float(ttl_s)
        self._items: Deque[WorkingItem] = deque(maxlen=self.size)

    def push(self, item: WorkingItem) -> None:
        self._items.append(item)

    def add(self, kind: str, text: str = "", ts: Optional[float] = None, **data: Any) -> WorkingItem:
        item = WorkingItem(ts=ts if ts is not None else time.time(), kind=kind, text=text, data=data)
        self.push(item)
        return item

    def _prune(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        while self._items and (now - self._items[0].ts) > self.ttl_s:
            self._items.popleft()

    def recent(self, n: Optional[int] = None, kind: Optional[str] = None,
               now: Optional[float] = None) -> List[WorkingItem]:
        """Newest-last list, optionally filtered by kind, after TTL pruning."""
        self._prune(now)
        items = list(self._items)
        if kind is not None:
            items = [i for i in items if i.kind == kind]
        if n is not None:
            items = items[-n:]
        return items

    def latest(self, kind: str) -> Optional[WorkingItem]:
        items = self.recent(kind=kind)
        return items[-1] if items else None

    def snapshot(self, n: Optional[int] = None) -> List[dict]:
        return [i.as_dict() for i in self.recent(n)]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        self._prune()
        return len(self._items)
