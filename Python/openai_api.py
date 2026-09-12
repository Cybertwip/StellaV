#!/usr/bin/env python3
"""OpenAI-compatible HTTP API for the Python Stella V agent."""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from reasoner import normalize_reason_size, ollama_model_for, reason
from tensor_engine import get_engine


MODELS = ("stella-v", "stella-v-1.5b", "stella-v-3b", "stella-v-local")


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def _last_user(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() in {"user", "human"}:
            text = _flatten_content(msg.get("content"))
            if text:
                return text
    if not messages:
        return ""
    return _flatten_content(messages[-1].get("content"))


class StellaOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "StellaV-OpenAI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[openai]", fmt % args, flush=True)

    def _json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, OpenAI-Beta")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/health", "/"}:
            self._json(200, {"ok": True, "name": "Stella V", "openai": "/v1"})
            return
        if path == "/v1/models":
            now = int(time.time())
            self._json(200, {
                "object": "list",
                "data": [{"id": mid, "object": "model", "created": now, "owned_by": "stella-v"} for mid in MODELS],
            })
            return
        self._error(404, f"unknown path {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._error(400, str(exc))
            return
        if path in {"/v1/chat/completions", "/v1/completions", "/v1/responses"}:
            self._complete(path, body)
            return
        self._error(404, f"unknown path {path}")

    def _complete(self, path: str, body: dict[str, Any]) -> None:
        model = str(body.get("model") or "stella-v-1.5b")
        messages = list(body.get("messages") or [])
        if not messages:
            prompt = body.get("input") or body.get("prompt") or ""
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
        question = _last_user(messages)
        local_only = "local" in model.lower()
        size = "3b" if "3b" in model.lower() else "1.5b"
        if local_only:
            from persistent_kb import answer_from_kb, load_kb_into_tensor_engine
            from model import LinearTokenLanguageModel

            load_kb_into_tensor_engine()
            linear = LinearTokenLanguageModel()
            linear.load()
            text, conf, _ = linear.predict(question)
            source = "local"
            _ = conf
        else:
            result = reason(question, size=normalize_reason_size(size), live_search=True)
            text = result["text"]
            source = result["source"]
        created = int(time.time())
        chat_id = "chatcmpl-" + uuid.uuid4().hex[:12]
        if path.endswith("/responses"):
            self._json(200, {
                "id": "resp_" + chat_id[9:],
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": model,
                "output_text": text,
                "output": [{
                    "id": "msg_" + uuid.uuid4().hex[:8],
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                }],
            })
            return
        payload = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(question.split()), "completion_tokens": len(text.split()), "total_tokens": 0},
            "stella_source": source,
            "stella_device": get_engine().status().get("device"),
            "stella_reasoner": ollama_model_for(size),
        }
        payload["usage"]["total_tokens"] = payload["usage"]["prompt_tokens"] + payload["usage"]["completion_tokens"]
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(200, payload)


def serve_openai(addr: str = "127.0.0.1:8765") -> None:
    host, _, port = addr.partition(":")
    httpd = ThreadingHTTPServer((host, int(port or "8765")), StellaOpenAIHandler)
    print(f"Stella V OpenAI API: http://{addr}/v1", flush=True)
    print(f"OpenCode baseURL:    http://{addr}/v1  models: stella-v-1.5b, stella-v-3b", flush=True)
    httpd.serve_forever()
