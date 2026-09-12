#!/usr/bin/env python3
"""GPU-capable tensor retrieval backend for Stella V.

Linear hashing is kept as the cheap embedding front-end. Retrieval itself uses
batched GEMM, multi-head scaled dot-product attention, and a residual projection
so the 1.5B/3B reasoner can pull ranked preprint passages quickly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch

    _TORCH = True
except Exception:  # pragma: no cover - torch is optional
    torch = None
    _TORCH = False

RETRIEVAL_DIM = int(os.environ.get("STELLAV_TENSOR_DIM", "128"))
RETRIEVAL_HEADS = int(os.environ.get("STELLAV_TENSOR_HEADS", "4"))
NGRAM_SIZES = (2, 3, 4)


def detect_device() -> str:
    forced = os.environ.get("STELLAV_TENSOR_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda", "mps"}:
        if forced == "cuda" and (not _TORCH or not torch.cuda.is_available()):
            return "cpu"
        if forced == "mps" and (not _TORCH or not torch.backends.mps.is_available()):
            return "cpu"
        return forced
    if _TORCH and torch.cuda.is_available():
        return "cuda"
    if _TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_available() -> bool:
    return detect_device() in {"cuda", "mps"}


def _fnv1a_u32(text: str) -> int:
    h = 2166136261
    for b in text.encode("utf-8", "ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def hashed_embedding(text: str, dim: int = RETRIEVAL_DIM) -> np.ndarray:
    """Deterministic hashed n-gram embedding, L2-normalized."""
    vec = np.zeros(dim, dtype=np.float32)
    blob = " ".join((text or "").lower().split())
    if not blob:
        return vec
    tokens = blob.split()
    features = list(tokens)
    compact = blob.replace(" ", "")
    for n in NGRAM_SIZES:
        if len(compact) >= n:
            features.extend(compact[i : i + n] for i in range(0, len(compact) - n + 1, n))
        features.extend(
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )
    for feat in features:
        h = _fnv1a_u32(feat)
        idx = h % dim
        sign = -1.0 if (h >> 31) else 1.0
        vec[idx] += sign
        vec[(h // dim) % dim] += 0.35 * sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= np.float32(norm)
    return vec


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp / denom


@dataclass
class RetrievedChunk:
    text: str
    score: float
    title: str = ""
    doi: str = ""
    url: str = ""
    topic: str = ""
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class TensorEngine:
    """Fast tensor memory the reasoning model uses to fetch preprint passages."""

    def __init__(self, dim: int = RETRIEVAL_DIM, heads: int = RETRIEVAL_HEADS, device: str | None = None) -> None:
        self.dim = max(32, int(dim))
        self.heads = max(1, int(heads))
        if self.dim % self.heads != 0:
            self.heads = 1
        self.device = device or detect_device()
        self.use_torch = bool(_TORCH and self.device != "cpu")
        rng = np.random.default_rng(20260911)
        scale = 1.0 / math.sqrt(self.dim)
        self.query_proj = (rng.standard_normal((self.dim, self.dim)).astype(np.float32) * scale)
        self.key_proj = (rng.standard_normal((self.dim, self.dim)).astype(np.float32) * scale)
        self.value_proj = (rng.standard_normal((self.dim, self.dim)).astype(np.float32) * scale)
        self.out_proj = (rng.standard_normal((self.dim, self.dim)).astype(np.float32) * scale)
        self._ids: list[str] = []
        self._chunks: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._torch_matrix = None

    def status(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "gpu": self.device in {"cuda", "mps"},
            "torch": bool(self.use_torch),
            "dim": self.dim,
            "heads": self.heads,
            "documents": len(self._chunks),
            "strategies": [
                "hashed-ngram-embedding",
                "linear-projection",
                "multihead-scaled-dot-product",
                "batched-gemm",
            ],
        }

    def clear(self) -> None:
        self._ids = []
        self._chunks = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._torch_matrix = None

    def embed(self, text: str) -> np.ndarray:
        base = hashed_embedding(text, self.dim)
        projected = base @ self.query_proj
        mixed = 0.65 * base + 0.35 * projected
        norm = float(np.linalg.norm(mixed))
        if norm > 0:
            mixed = mixed / np.float32(norm)
        return mixed.astype(np.float32, copy=False)

    def add(
        self,
        text: str,
        *,
        chunk_id: str = "",
        title: str = "",
        doi: str = "",
        url: str = "",
        topic: str = "",
        source: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        cid = chunk_id or f"doc-{len(self._ids)}"
        if cid in self._ids:
            idx = self._ids.index(cid)
            self._chunks[idx] = {
                "id": cid,
                "text": text,
                "title": title,
                "doi": doi,
                "url": url,
                "topic": topic,
                "source": source,
                "meta": meta or {},
            }
            self._matrix[idx] = self.embed(f"{title} {text}")
            self._torch_matrix = None
            return
        self._ids.append(cid)
        self._chunks.append({
            "id": cid,
            "text": text,
            "title": title,
            "doi": doi,
            "url": url,
            "topic": topic,
            "source": source,
            "meta": meta or {},
        })
        vec = self.embed(f"{title} {text}").reshape(1, -1)
        self._matrix = vec if self._matrix.size == 0 else np.vstack([self._matrix, vec])
        self._torch_matrix = None

    def add_paper(self, paper: dict[str, Any], *, topic: str = "") -> None:
        doi = str(paper.get("doi") or "")
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        url = str(paper.get("url") or "")
        body = abstract or title
        if not body:
            return
        self.add(
            body,
            chunk_id=doi or title,
            title=title,
            doi=doi,
            url=url,
            topic=topic or str(paper.get("topic") or ""),
            source=str(paper.get("source") or "scienceopen"),
            meta=paper,
        )

    def add_many(self, papers: Iterable[dict[str, Any]], *, topic: str = "") -> int:
        before = len(self._chunks)
        for paper in papers:
            self.add_paper(paper, topic=topic)
        return len(self._chunks) - before

    def _gemm(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if keys.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if self.use_torch:
            q = torch.from_numpy(np.ascontiguousarray(query)).to(self.device)
            k = self._keys_torch(keys)
            scores = torch.matmul(k, q)
            return scores.detach().to("cpu").numpy().astype(np.float32)
        return (keys @ query).astype(np.float32)

    def _keys_torch(self, keys: np.ndarray):
        if self._torch_matrix is None or self._torch_matrix.shape[0] != keys.shape[0]:
            self._torch_matrix = torch.from_numpy(np.ascontiguousarray(keys)).to(self.device)
        return self._torch_matrix

    def attention_scores(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Multi-head scaled dot-product scores over the document matrix."""
        if keys.size == 0:
            return np.zeros((0,), dtype=np.float32)
        head_dim = self.dim // self.heads
        q = query @ self.query_proj
        k = keys @ self.key_proj
        q_heads = q.reshape(self.heads, head_dim)
        k_heads = k.reshape(keys.shape[0], self.heads, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        scores = np.einsum("hd,nhd->nh", q_heads, k_heads) * scale
        return scores.mean(axis=1).astype(np.float32)

    def context_vector(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if keys.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        scores = self.attention_scores(query, keys)
        weights = _softmax(scores)
        values = keys @ self.value_proj
        pooled = weights @ values
        residual = 0.5 * query + 0.5 * (pooled @ self.out_proj)
        norm = float(np.linalg.norm(residual))
        if norm > 0:
            residual = residual / np.float32(norm)
        return residual.astype(np.float32)

    def search(self, query: str, *, k: int = 6, candidates: Sequence[dict[str, Any]] | None = None) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []
        q = self.embed(query)
        if candidates:
            keys = np.vstack([self.embed(f"{c.get('title', '')} {c.get('abstract') or c.get('text', '')}") for c in candidates])
            rows = list(candidates)
        else:
            if self._matrix.size == 0:
                return []
            keys = self._matrix
            rows = self._chunks
        linear = self._gemm(q, keys)
        attn = self.attention_scores(q, keys)
        scores = 0.45 * linear + 0.55 * attn
        order = np.argsort(-scores)
        out: list[RetrievedChunk] = []
        for idx in order[: max(1, k)]:
            row = rows[int(idx)]
            text = str(row.get("text") or row.get("abstract") or "")
            out.append(
                RetrievedChunk(
                    text=text,
                    score=float(scores[int(idx)]),
                    title=str(row.get("title") or ""),
                    doi=str(row.get("doi") or ""),
                    url=str(row.get("url") or ""),
                    topic=str(row.get("topic") or ""),
                    source=str(row.get("source") or ""),
                    meta=dict(row.get("meta") or row),
                )
            )
        return out

    def select_for_reasoner(self, query: str, *, k: int = 6, extra: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        hits = self.search(query, k=k)
        extra_hits = self.search(query, k=k, candidates=list(extra)) if extra else []
        merged: dict[str, RetrievedChunk] = {}
        for hit in extra_hits + hits:
            key = hit.doi or hit.title or hit.text[:80]
            prev = merged.get(key)
            if prev is None or hit.score > prev.score:
                merged[key] = hit
        ranked = sorted(merged.values(), key=lambda h: h.score, reverse=True)[:k]
        q = self.embed(query)
        keys = np.vstack([self.embed(f"{h.title} {h.text}") for h in ranked]) if ranked else np.zeros((0, self.dim), dtype=np.float32)
        ctx = self.context_vector(q, keys) if ranked else np.zeros(self.dim, dtype=np.float32)
        return {
            "query": query,
            "hits": ranked,
            "context_norm": float(np.linalg.norm(ctx)),
            "engine": self.status(),
        }


_ENGINE: TensorEngine | None = None


def get_engine() -> TensorEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TensorEngine()
    return _ENGINE
