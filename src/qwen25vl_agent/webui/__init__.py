"""The optional web frontend.

``qwen25vl-agent web`` starts the monitor in a background thread and this server on the main
thread, so the browser gets a live event stream (SSE) and can read memory and trigger actions.
``--no-monitor`` serves the same UI without owning the camera, for when a monitor is already
running elsewhere.

The package is deliberately framework-free: stdlib ``http.server`` plus one self-contained
``index.html``, matching how the sibling ``qwen25vl-lite`` engine serves its own page.
"""

from __future__ import annotations

from .events import EventBus, serialize_event
from .server import WebApp, make_server, serve

__all__ = ["EventBus", "serialize_event", "WebApp", "make_server", "serve"]
