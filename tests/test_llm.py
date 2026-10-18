"""The HTTP client for the reasoning engine, exercised against a fake local server.

A real ``http.server`` on an ephemeral port is used rather than monkeypatching ``urlopen`` so the
request body the engine actually receives can be asserted -- in particular that the vision budget
travels in the request body, which is the field that really sizes the image.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qwen25vl_agent.llm import LLMClient, LLMError, chat_messages, extract_text, text_client


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):        # keep the test output clean
        pass

    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"model": "fake-model", "status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) or b"{}"
        _Handler.requests.append(json.loads(raw))
        self._reply({"choices": [{"message": {"content": "hi there"}}]})


@pytest.fixture
def server():
    _Handler.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chat_messages_text_only():
    messages = chat_messages("hello")
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_chat_messages_puts_the_image_first():
    messages = chat_messages("hi", b"\xff\xd8fakejpeg")
    content = messages[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "hi"}


def test_chat_sends_vision_budget_in_the_body(server):
    client = LLMClient(url=server, max_vision_tokens=128, retries=0)
    assert client.chat([{"role": "user", "content": "hi"}]) == "hi there"
    body = _Handler.requests[-1]
    assert body["max_vision_tokens"] == 128
    assert body["temperature"] == 0.0
    assert body["model"] == "Qwen2.5-VL-3B-Instruct"


def test_text_client_omits_the_vision_budget(server):
    client = text_client(server, "text-model", retries=0)
    client.chat([{"role": "user", "content": "hi"}])
    assert "max_vision_tokens" not in _Handler.requests[-1]
    assert _Handler.requests[-1]["model"] == "text-model"


def test_health_returns_the_model(server):
    assert LLMClient(url=server, retries=0).health()["model"] == "fake-model"


def test_ask_with_system_prompt_prepends_a_system_turn(server):
    client = LLMClient(url=server, retries=0)
    assert client.ask("question", system="you are a monitor") == "hi there"
    messages = _Handler.requests[-1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "you are a monitor"
    assert messages[-1]["role"] == "user"


def test_max_tokens_override(server):
    client = LLMClient(url=server, retries=0, max_tokens=64)
    client.chat([{"role": "user", "content": "hi"}], max_tokens=7)
    assert _Handler.requests[-1]["max_tokens"] == 7


def test_extra_body_fields_are_forwarded(server):
    client = LLMClient(url=server, retries=0)
    client.chat([{"role": "user", "content": "hi"}], extra={"top_k": 1})
    assert _Handler.requests[-1]["top_k"] == 1


def test_unexpected_reply_shape_raises(server):
    """A reply without choices[0].message.content must be an LLMError, not a KeyError."""

    class _BadHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.dumps({"error": "boom"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _BadHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = LLMClient(url=f"http://127.0.0.1:{httpd.server_address[1]}", retries=0)
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_unreachable_engine_raises_llm_error():
    # Bind and release a port so we know nothing is listening on it.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    client = LLMClient(url=f"http://127.0.0.1:{port}", retries=0, timeout=2.0)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_a_hung_engine_is_not_retried():
    """A server that accepts the connection and never replies is attempted exactly once.

    Each attempt costs the full ``timeout``, so retrying a hang multiplies the stall -- a 24/7
    monitor's tick loop stays frozen for the whole of it (observed live: 3 x 120 s = 6 minutes,
    with ``ticks`` stuck).  A stuck server will not answer the next attempt either, so the only
    useful response is to fail fast and let the loop continue.
    """
    hits = {"n": 0}

    class _HangHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            hits["n"] += 1
            time.sleep(1.0)                 # outlive the client's timeout
            try:                            # the client is gone by now; writing may fail
                self.send_response(200)
                self.end_headers()
            except OSError:
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _HangHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = LLMClient(url=f"http://127.0.0.1:{httpd.server_address[1]}",
                           retries=2, timeout=0.3)
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])
        assert hits["n"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_refused_connection_is_still_retried():
    """The fast failure mode keeps its retries -- only the slow one is cut short."""
    hits = {"n": 0}

    class _RefuseHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            hits["n"] += 1
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RefuseHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = LLMClient(url=f"http://127.0.0.1:{httpd.server_address[1]}",
                           retries=2, timeout=2.0, retry_backoff=0.0)
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])
        assert hits["n"] == 3
    finally:
        httpd.shutdown()
        httpd.server_close()


# -- thinking models ------------------------------------------------------------------
def test_extract_text_prefers_content_over_reasoning():
    reply = {"choices": [{"message": {"content": "答案", "reasoning_content": "思考过程"}}]}
    assert extract_text(reply) == "答案"


def test_extract_text_falls_back_to_reasoning_when_content_is_empty():
    """Qwen3.5 with thinking on returns an empty content and the answer in reasoning_content."""
    reply = {"choices": [{"message": {"content": "", "reasoning_content": "思考后的答案"}}]}
    assert extract_text(reply) == "思考后的答案"


def test_extract_text_strips_inline_think_tags():
    reply = {"choices": [{"message": {"content": "<thinking>噪音</thinking>你好。"}}]}
    assert extract_text(reply) == "你好。"


def test_extract_text_raises_on_a_shape_without_a_message():
    with pytest.raises(LLMError):
        extract_text({"error": "boom"})


def test_chat_returns_the_reasoning_fallback(server):
    """A reply whose content is empty must not silently become an empty decision."""

    class _ThinkingHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.dumps({"choices": [{"message": {
                "content": "", "reasoning_content": "只有思考内容"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ThinkingHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = LLMClient(url=f"http://127.0.0.1:{httpd.server_address[1]}", retries=0)
        assert client.chat([{"role": "user", "content": "hi"}]) == "只有思考内容"
    finally:
        httpd.shutdown()
        httpd.server_close()
