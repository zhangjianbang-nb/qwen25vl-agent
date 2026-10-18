"""A thread-safe fan-out for live monitor events.

The monitor publishes from its tick thread; every browser connection drains its own bounded
queue from an HTTP thread.  A slow or dead client must never stall a tick, so each subscriber
gets a small ring: when it is full the *oldest* event is dropped, not the newest.  For a live
feed, missing an old line beats freezing on a backlog.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List

__all__ = ["EventBus", "serialize_event"]

_DEFAULT_MAX = 200


def serialize_event(event: Any) -> Dict[str, Any]:
    """A JSON-ready event that keeps the epoch timestamp.

    ``MonitorEvent.as_dict()`` formats a ``time`` string and drops ``ts``.  The page needs the
    epoch so it can sort and dedupe across reconnects, so this keeps both, and spreads the
    event's ``data`` to the top level exactly like ``as_dict()`` does.
    """
    return {
        "ts": float(event.ts),
        "time": time.strftime("%H:%M:%S", time.localtime(event.ts)),
        "kind": str(event.kind),
        "message": str(event.message),
        "importance": round(float(event.importance), 3),
        **event.data,
    }


class EventBus:
    """Publish monitor events to any number of independent subscribers."""

    def __init__(self, max_queue: int = _DEFAULT_MAX) -> None:
        self._lock = threading.Lock()
        self._subs: List["queue.Queue[Dict[str, Any]]"] = []
        self._max = max(1, int(max_queue))

    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self._max)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, event: Any) -> None:
        """Fan an event out to every subscriber.  Accepts a ``MonitorEvent`` or a plain dict."""
        item = event if isinstance(event, dict) else serialize_event(event)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except queue.Full:
                # Full subscriber: drop its oldest event and retry, so the newest always lands.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(item)
                except queue.Full:
                    pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)
