#!/usr/bin/env python3
"""
agent_client.py
──────────────
Robust Ollama REST client used by StellaV's agentic teacher.

This client streams Ollama output by default so the terminal shows real progress
while the teacher model is generating. It also disables generation read timeouts
by default, which prevents long local Ollama generations from being killed while
they are still producing tokens.

Environment:
    STELLAV_OLLAMA_MODEL=qwen2.5:1.5b
    STELLAV_OLLAMA_BASE=http://localhost:11434
    STELLAV_OLLAMA_STREAM=1
    STELLAV_OLLAMA_TIMEOUT=90       # 0 = no generation read timeout
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Iterable

OLLAMA_BASE = os.environ.get("STELLAV_OLLAMA_BASE", "http://localhost:11434").rstrip("/")
MODEL_NAME = os.environ.get("STELLAV_OLLAMA_MODEL", os.environ.get("STELLAV_REASON_1_5B", "qwen2.5:1.5b"))

STREAM_OUTPUT = os.environ.get("STELLAV_OLLAMA_STREAM", "1") != "0"
_GEN_TIMEOUT_VALUE = float(os.environ.get("STELLAV_OLLAMA_TIMEOUT", "90"))
GEN_TIMEOUT: float | None = None if _GEN_TIMEOUT_VALUE <= 0 else _GEN_TIMEOUT_VALUE

SYSTEM_PROMPT = (
    "You are Stella V's local medical-research model. You help select and reason "
    "over ScienceOpen preprints and established study methods.\n\n"
    "Hard constraints:\n"
    "1. Stay in medical research: clinical trials, epidemiology, oncology, "
    "cardiology, immunology, infectious disease, pharmacology, genomics, "
    "neurology, and public health.\n"
    "2. Treat preprints as unreviewed evidence and say so when you rely on them.\n"
    "3. This is not diagnosis or treatment advice.\n"
    "4. When the user prompt asks for compact p/q/a lines, output only that format.\n"
    "5. Do not discuss game engines, graphics, or gameplay systems.\n"
    "6. Prefer citing ScienceOpen DOIs when passages are provided."
)


def _request(endpoint: str, payload: dict[str, Any]) -> urllib.request.Request:
    url = f"{OLLAMA_BASE}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _post(endpoint: str, payload: dict[str, Any], timeout: float | None = GEN_TIMEOUT) -> dict[str, Any]:
    req = _request(endpoint, payload)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _stream_post_text(endpoint: str, payload: dict[str, Any], timeout: float | None = GEN_TIMEOUT) -> str:
    """
    Stream Ollama JSONL chunks to stdout while collecting the final text.

    /api/generate chunks:
        {"response": "...", "done": false}
        {"thinking": "...", "done": false}

    /api/chat chunks:
        {"message": {"content": "..."}, "done": false}
        {"thinking": "...", "done": false}
    """
    req = _request(endpoint, payload)
    collected: list[str] = []
    printed_any = False

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))

            thinking = chunk.get("thinking")
            if isinstance(thinking, str) and thinking:
                if STREAM_OUTPUT:
                    if not printed_any:
                        print("[ollama] stream begin", flush=True)
                        printed_any = True
                    print(thinking, end="", flush=True)
                continue

            if endpoint == "/api/chat":
                msg = chunk.get("message")
                text = str(msg.get("content", "")) if isinstance(msg, dict) else ""
            else:
                text = str(chunk.get("response", ""))

            if text:
                collected.append(text)
                if STREAM_OUTPUT:
                    if not printed_any:
                        print("[ollama] stream begin", flush=True)
                        printed_any = True
                    print(text, end="", flush=True)

            if chunk.get("done"):
                break

    if STREAM_OUTPUT and printed_any:
        print("\n[ollama] stream end", flush=True)

    return "".join(collected).strip()


def is_alive() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def available_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return [str(m.get("name", "")) for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _build_options(
    *,
    temperature: float,
    max_tokens: int,
    top_p: float,
    repeat_penalty: float,
    top_k: int | None,
    seed: int | None,
    num_ctx: int | None,
    stop: Iterable[str] | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
    }
    if top_k is not None:
        options["top_k"] = top_k
    if seed is not None:
        options["seed"] = seed
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if stop is not None:
        options["stop"] = list(stop)
    return options


def generate(
    prompt: str,
    *,
    temperature: float = 0.25,
    max_tokens: int = 2048,
    json_mode: bool = False,
    top_p: float = 0.9,
    repeat_penalty: float = 1.05,
    top_k: int | None = None,
    seed: int | None = None,
    num_ctx: int | None = None,
    stop: Iterable[str] | None = None,
    include_system: bool = True,
) -> str:
    """Single-turn generation through Ollama /api/generate."""
    full_prompt = (SYSTEM_PROMPT + "\n\n" + prompt) if include_system else prompt
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": STREAM_OUTPUT,
        "options": _build_options(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            seed=seed,
            num_ctx=num_ctx,
            stop=stop,
        ),
    }
    if json_mode:
        payload["format"] = "json"

    if STREAM_OUTPUT:
        return _stream_post_text("/api/generate", payload)

    result = _post("/api/generate", payload)
    return str(result.get("response", "")).strip()


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.25,
    max_tokens: int = 2048,
    json_mode: bool = False,
    top_p: float = 0.9,
    repeat_penalty: float = 1.05,
    top_k: int | None = None,
    seed: int | None = None,
    num_ctx: int | None = None,
    stop: Iterable[str] | None = None,
) -> str:
    """Multi-turn chat through Ollama /api/chat."""
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": full_messages,
        "stream": STREAM_OUTPUT,
        "options": _build_options(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            seed=seed,
            num_ctx=num_ctx,
            stop=stop,
        ),
    }
    if json_mode:
        payload["format"] = "json"

    if STREAM_OUTPUT:
        return _stream_post_text("/api/chat", payload)

    result = _post("/api/chat", payload)
    return str(result.get("message", {}).get("content", "")).strip()
