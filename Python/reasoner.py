#!/usr/bin/env python3
"""Selectable 1.5B / 3B reasoning model for Stella V.

The reasoner asks the GPU tensor engine for ranked ScienceOpen passages, then
uses a lightweight local LLM (Ollama by default, Conqueror if configured) to
select the relevant evidence and write a grounded research answer.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from scienceopen import search_europepmc_scienceopen, search_scienceopen
from tensor_engine import RetrievedChunk, TensorEngine, get_engine

REASON_1_5B = "1.5b"
REASON_3B = "3b"
DEFAULT_REASON_SIZE = os.environ.get("STELLAV_REASON_SIZE", REASON_1_5B).strip().lower() or REASON_1_5B

OLLAMA_MODELS = {
    REASON_1_5B: os.environ.get("STELLAV_REASON_1_5B", "qwen2.5:1.5b"),
    REASON_3B: os.environ.get("STELLAV_REASON_3B", "qwen2.5:3b"),
}

CONQUEROR_MODELS = {
    REASON_1_5B: os.environ.get("STELLAV_CONQUEROR_1_5B", "hf-qwen-qwen2-5-1-5b"),
    REASON_3B: os.environ.get("STELLAV_CONQUEROR_3B", "hf-qwen-qwen2-5-3b"),
}

SYSTEM_PROMPT = (
    "You are Stella V, a local medical-research assistant. "
    "You retrieve ScienceOpen preprints through a tensor engine, then reason over that evidence. "
    "Rules:\n"
    "1. Use only the supplied preprint passages plus basic established medical research knowledge.\n"
    "2. Select the passages that actually bear on the question; ignore the rest.\n"
    "3. Cite ScienceOpen DOIs for claims drawn from the passages.\n"
    "4. Distinguish preprint (not peer-reviewed) evidence from established methods.\n"
    "5. If the evidence is weak or off-topic, say so clearly.\n"
    "6. This is research assistance, not diagnosis or treatment advice.\n"
    "7. Write concise, technical prose. No markdown headings unless useful for structure."
)


def normalize_reason_size(value: str | None) -> str:
    key = str(value or DEFAULT_REASON_SIZE).strip().lower().replace("_", "-")
    if key in {REASON_3B, "3", "3.0b", "qwen2.5:3b", "qwen2.5-3b", "llama3.2:3b"}:
        return REASON_3B
    return REASON_1_5B


def ollama_model_for(size: str | None) -> str:
    return OLLAMA_MODELS[normalize_reason_size(size)]


def conqueror_model_for(size: str | None) -> str:
    return CONQUEROR_MODELS[normalize_reason_size(size)]


def format_passages(hits: Sequence[RetrievedChunk], *, limit: int = 6) -> str:
    lines: list[str] = []
    for i, hit in enumerate(hits[:limit], start=1):
        doi = hit.doi or "n/a"
        title = hit.title or "Untitled preprint"
        url = hit.url or ""
        snippet = " ".join((hit.text or "").split())
        if len(snippet) > 900:
            snippet = snippet[:897] + "..."
        lines.append(
            f"[{i}] {title}\n"
            f"DOI: {doi}\n"
            f"URL: {url}\n"
            f"Tensor score: {hit.score:.4f}\n"
            f"Passage: {snippet}"
        )
    return "\n\n".join(lines) if lines else "(no preprint passages retrieved)"


def build_reason_prompt(question: str, hits: Sequence[RetrievedChunk], *, local_draft: str = "") -> str:
    passages = format_passages(hits)
    draft = ""
    if local_draft.strip():
        draft = f"\nLocal tensor-memory draft (may be incomplete):\n{local_draft.strip()}\n"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question.strip()}\n"
        f"{draft}\n"
        f"ScienceOpen preprint passages ranked by the Stella V tensor engine:\n{passages}\n\n"
        "Select the relevant passages, reason over them, and answer the question. "
        "End with a short evidence note listing the DOIs you used."
    )


def _reason_with_ollama(prompt: str, size: str) -> str:
    import agent_client

    model = ollama_model_for(size)
    previous = agent_client.MODEL_NAME
    try:
        agent_client.MODEL_NAME = model
        if not agent_client.is_alive():
            raise RuntimeError(f"Ollama is not reachable at {agent_client.OLLAMA_BASE}")
        return agent_client.generate(prompt, temperature=0.15, max_tokens=640, num_ctx=4096)
    finally:
        agent_client.MODEL_NAME = previous


def retrieve_for_question(
    question: str,
    *,
    engine: TensorEngine | None = None,
    live_search: bool = True,
    k: int = 6,
) -> dict[str, Any]:
    engine = engine or get_engine()
    live: list[dict[str, Any]] = []
    errors: list[str] = []
    if live_search:
        try:
            live = search_scienceopen(question, rows=max(8, k))
        except Exception as exc:
            errors.append(f"scienceopen:{exc}")
            try:
                live = search_europepmc_scienceopen(question, rows=max(8, k))
            except Exception as exc2:
                errors.append(f"europepmc:{exc2}")
        if live:
            engine.add_many(live)
    packed = engine.select_for_reasoner(question, k=k, extra=live)
    packed["live_count"] = len(live)
    packed["errors"] = errors
    return packed


def reason(
    question: str,
    *,
    size: str | None = None,
    engine: TensorEngine | None = None,
    local_draft: str = "",
    live_search: bool = True,
    k: int = 6,
) -> dict[str, Any]:
    size = normalize_reason_size(size)
    retrieved = retrieve_for_question(question, engine=engine, live_search=live_search, k=k)
    hits: list[RetrievedChunk] = list(retrieved.get("hits") or [])
    prompt = build_reason_prompt(question, hits, local_draft=local_draft)
    source = f"reasoner:{size}:{ollama_model_for(size)}"
    text = ""
    error = ""
    try:
        text = _reason_with_ollama(prompt, size)
    except Exception as exc:
        error = str(exc)
        if hits:
            top = hits[0]
            text = (
                f"From ScienceOpen preprint {top.title} ({top.doi or 'DOI n/a'}): "
                f"{' '.join(top.text.split())[:700]} "
                "The 1.5B/3B reasoner was unavailable, so this is the tensor-ranked passage only. "
                "This is research information, not clinical advice."
            )
            source = "tensor-engine"
        elif local_draft.strip():
            text = local_draft.strip()
            source = "local-draft"
        else:
            text = (
                "I could not reach the reasoning model or ScienceOpen for this question. "
                "Index medical preprints first, or start Ollama with qwen2.5:1.5b / qwen2.5:3b."
            )
            source = "fallback"
    confidence = 0.82 if source.startswith("reasoner:") and text else 0.58
    if hits:
        confidence = min(0.94, max(confidence, 0.55 + max(0.0, hits[0].score) * 0.2))
    return {
        "text": text.strip(),
        "confidence": confidence,
        "source": source,
        "reason_size": size,
        "model": ollama_model_for(size),
        "hits": hits,
        "engine": retrieved.get("engine") or {},
        "live_count": retrieved.get("live_count", 0),
        "error": error,
        "prompt": prompt,
    }
