"""HTTP client for the reasoning engine.

The agent does not load a model.  It talks to a *running* ``qwen25vl-lite serve`` over the
OpenAI-compatible API that engine exposes.  On this machine that is not a stylistic choice:
the served Qwen2.5-VL holds ~5.9 GB of an 8 GB card, so a second in-process copy would OOM,
and reusing the server also skips its ~20 s model load.

Only the standard library is used, so the client adds no dependency beyond the engine itself.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["LLMClient", "LLMError", "chat_messages", "image_part"]


class LLMError(RuntimeError):
    """Raised when the engine cannot be reached or answers with an error."""


def image_part(jpeg: bytes) -> Dict[str, Any]:
    """One OpenAI-style image content part, as a data URL."""
    from .imaging import jpeg_data_url

    return {"type": "image_url", "image_url": {"url": jpeg_data_url(jpeg)}}


def chat_messages(text: str, jpeg: Optional[bytes] = None) -> List[Dict[str, Any]]:
    """Build a single user turn with an optional image.

    Image first, then text: that is the order the engine's parser expects for a Qwen2.5-VL
    prompt, and it keeps the vision block contiguous.
    """
    content: List[Dict[str, Any]] = []
    if jpeg is not None:
        content.append(image_part(jpeg))
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


@dataclass
class LLMClient:
    """A thin client over ``/v1/chat/completions``.

    ``max_vision_tokens`` is sent in the request body (not as a server flag) because that is
    the field the engine actually reads to size the image budget.
    """

    url: str = "http://127.0.0.1:8081"
    model: str = "Qwen2.5-VL-3B-Instruct"
    max_tokens: int = 256
    temperature: float = 0.0
    timeout: float = 120.0
    max_vision_tokens: Optional[int] = 128
    retries: int = 2
    retry_backoff: float = 0.5

    # -- plumbing -----------------------------------------------------------------
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url.rstrip('/')}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
        raise LLMError(f"{type(last).__name__}: {last} ({self.url}{path})")

    def _get(self, path: str) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.url.rstrip('/')}{path}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMError(f"{type(exc).__name__}: {exc} ({self.url}{path})") from exc

    # -- public API ---------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """``/health`` -- used by ``doctor`` to confirm the engine is up."""
        return self._get("/health")

    def stats(self) -> Dict[str, Any]:
        """``/stats`` -- scheduler/KV counters, useful when the monitor looks slow."""
        return self._get("/stats")

    def chat(self, messages: List[Dict[str, Any]], *, max_tokens: Optional[int] = None,
             extra: Optional[Dict[str, Any]] = None) -> str:
        """Send messages and return the assistant text."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
        }
        if self.max_vision_tokens is not None:
            body["max_vision_tokens"] = self.max_vision_tokens
        if extra:
            body.update(extra)
        reply = self._post("/v1/chat/completions", body)
        try:
            return reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected reply shape: {json.dumps(reply)[:400]}") from exc

    def ask(self, prompt: str, jpeg: Optional[bytes] = None, *,
            system: Optional[str] = None, max_tokens: Optional[int] = None,
            extra: Optional[Dict[str, Any]] = None) -> str:
        """Convenience wrapper: one system prompt (optional) + one user turn."""
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(chat_messages(prompt, jpeg))
        return self.chat(messages, max_tokens=max_tokens, extra=extra)


def text_client(url: str, model: Optional[str] = None, **kwargs) -> LLMClient:
    """A client for a text-only endpoint (e.g. lm-studio) with no image budget fields."""
    return LLMClient(url=url, model=model or "local-model", max_vision_tokens=None, **kwargs)
