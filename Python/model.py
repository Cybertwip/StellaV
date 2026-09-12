from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from bootstrap_data import semantic_fallback_response

MODEL_VERSION = 2
MAX_TOKENS = 4096
LINEAR_INPUT_TOKENS = 256
MAX_OUTPUT_TOKENS = 512
HASH_BUCKETS = 256
FEATURE_SIZE = LINEAR_INPUT_TOKENS + HASH_BUCKETS + 8
MODEL_PATH = "qa_linear_token_model.json"

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "user:", "assistant:"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID = range(6)

TOKEN_RE = re.compile(r"[a-z0-9_']+|[^\w\s]", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())[:MAX_TOKENS]


def fnv1a(text: str) -> int:
    h = 2166136261
    for b in text.encode("utf-8", "ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _normalized_text(text: str) -> str:
    return " ".join(tokenize(text))


def _bigram_set(tokens: List[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _compress_ids(ids: List[int], limit: int) -> List[int]:
    if len(ids) <= limit:
        return ids
    if limit <= 1:
        return ids[:1]
    step = (len(ids) - 1) / float(limit - 1)
    return [ids[min(len(ids) - 1, round(i * step))] for i in range(limit)]


@dataclass
class Sample:
    question: str
    answer: str


class LinearTokenLanguageModel:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.samples: List[Sample] = []
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIALS)}
        self.id_to_token: List[str] = list(SPECIALS)
        self.weights: np.ndarray | None = None

    def teach_vocabulary(self, tokens: Iterable[str]) -> None:
        """Preload lexical tokens before training samples are added."""
        for raw in tokens:
            token = str(raw).strip().lower()
            if not token:
                continue
            self.token_id(token, allow_add=True)
        self.weights = None

    def vocabulary_tokens(self) -> List[str]:
        return list(self.id_to_token)

    def token_id(self, token: str, allow_add: bool) -> int:
        token = token.lower()
        if token in self.token_to_id:
            return self.token_to_id[token]
        if not allow_add:
            return UNK_ID
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def encode_tokens(self, tokens: List[str], allow_add: bool) -> List[int]:
        return [self.token_id(t, allow_add) for t in tokens[:MAX_TOKENS]]

    def add_sample(self, question: str, answer: str) -> None:
        self.samples.append(Sample(question, answer))
        prompt = ["<bos>", "user:"] + tokenize(question) + ["assistant:"]
        self.encode_tokens(prompt, True)
        self.encode_tokens(tokenize(answer) + ["<eos>"], True)
        self.weights = None

    def _expected_weight_shape(self) -> tuple[int, int]:
        return (FEATURE_SIZE + 1, MAX_OUTPUT_TOKENS)

    def _weights_compatible(self) -> bool:
        if self.weights is None:
            return True
        return bool(
            isinstance(self.weights, np.ndarray)
            and self.weights.ndim == 2
            and tuple(self.weights.shape) == self._expected_weight_shape()
        )

    def make_features(self, ids: List[int]) -> np.ndarray:
        x = np.zeros(FEATURE_SIZE, dtype=np.float64)
        vocab = max(1, len(self.id_to_token))
        compressed_ids = _compress_ids(ids, LINEAR_INPUT_TOKENS)

        for i in range(LINEAR_INPUT_TOKENS):
            tid = compressed_ids[i] if i < len(compressed_ids) else PAD_ID
            x[i] = tid / vocab

        for i, tid in enumerate(compressed_ids[:LINEAR_INPUT_TOKENS]):
            tok = self.id_to_token[tid] if 0 <= tid < len(self.id_to_token) else "<unk>"
            x[LINEAR_INPUT_TOKENS + (fnv1a(tok) % HASH_BUCKETS)] += 1.0 / LINEAR_INPUT_TOKENS
            if i + 1 < len(compressed_ids):
                tok2 = self.id_to_token[compressed_ids[i + 1]] if 0 <= compressed_ids[i + 1] < len(self.id_to_token) else "<unk>"
                x[LINEAR_INPUT_TOKENS + (fnv1a(tok + "_" + tok2) % HASH_BUCKETS)] += 0.5 / LINEAR_INPUT_TOKENS

        off = LINEAR_INPUT_TOKENS + HASH_BUCKETS
        x[off + 0] = min(1.0, len(ids) / MAX_TOKENS)
        x[off + 1] = 1.0 if self.token_to_id.get("?", -1) in compressed_ids else 0.0
        x[off + 2] = compressed_ids.count(self.token_to_id.get(".", -1)) / 8.0
        x[off + 3] = compressed_ids.count(self.token_to_id.get(",", -1)) / 8.0
        x[off + 4] = (compressed_ids[0] / vocab) if compressed_ids else 0.0
        x[off + 5] = (compressed_ids[-1] / vocab) if compressed_ids else 0.0
        x[off + 6] = 1.0
        x[off + 7] = len(self.id_to_token) / 10000.0
        return x

    def train(self, ridge: float = 1e-3) -> bool:
        if not self.samples:
            self.weights = None
            return False

        x_rows: List[np.ndarray] = []
        y_rows: List[np.ndarray] = []
        for s in self.samples:
            prompt = ["<bos>", "user:"] + tokenize(s.question) + ["assistant:"]
            input_ids = self.encode_tokens(prompt, False)
            x_rows.append(self.make_features(input_ids))

            answer_ids = self.encode_tokens(tokenize(s.answer)[:MAX_OUTPUT_TOKENS], False) + [EOS_ID]
            y = np.full(MAX_OUTPUT_TOKENS, PAD_ID, dtype=np.float64)
            for i, tid in enumerate(answer_ids[:MAX_OUTPUT_TOKENS]):
                y[i] = tid
            y_rows.append(y)

        X = np.vstack(x_rows)
        Y = np.vstack(y_rows)
        Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        I = np.eye(Xb.shape[1], dtype=np.float64)
        I[-1, -1] = 0.0
        self.weights = np.linalg.pinv(Xb.T @ Xb + ridge * I) @ Xb.T @ Y
        return True

    def retrieve(self, question: str) -> Tuple[str, float, List[str]] | None:
        if not self.samples:
            return None

        q_tokens = tokenize(question)
        if not q_tokens:
            return None

        q_set = set(q_tokens)
        q_bigrams = _bigram_set(q_tokens)
        q_norm = " ".join(q_tokens)

        best_sample: Sample | None = None
        best_score = 0.0

        for sample in self.samples:
            sample_tokens = tokenize(sample.question)
            if not sample_tokens:
                continue

            sample_set = set(sample_tokens)
            overlap = len(q_set & sample_set)
            if overlap == 0:
                continue

            union = len(q_set | sample_set)
            token_score = overlap / max(1, union)

            sample_bigrams = _bigram_set(sample_tokens)
            bigram_overlap = len(q_bigrams & sample_bigrams)
            bigram_union = len(q_bigrams | sample_bigrams)
            bigram_score = bigram_overlap / max(1, bigram_union) if q_bigrams or sample_bigrams else 0.0

            sample_norm = " ".join(sample_tokens)
            exact_bonus = 0.45 if q_norm == sample_norm else 0.0
            contains_bonus = 0.18 if q_norm in sample_norm or sample_norm in q_norm else 0.0
            score = token_score + (0.35 * bigram_score) + exact_bonus + contains_bonus

            if score > best_score:
                best_score = score
                best_sample = sample

        if best_sample is None or best_score < 0.45:
            return None

        conf = min(0.99, 0.58 + min(best_score, 1.0) * 0.37)
        return best_sample.answer, conf, tokenize(best_sample.answer)

    def predict(self, question: str) -> Tuple[str, float, List[str]]:
        try:
            from persistent_kb import answer_from_kb, load_kb_into_tensor_engine

            load_kb_into_tensor_engine()
            kb_answer = answer_from_kb(question)
            if kb_answer is not None:
                return kb_answer
        except Exception:
            pass
        semantic = semantic_fallback_response(question)
        if semantic is not None:
            text, conf = semantic
            return text, conf, tokenize(text)
        retrieved = self.retrieve(question)
        if retrieved is not None:
            return retrieved
        if not self._weights_compatible():
            self.weights = None
            if self.samples:
                self.train()
        if self.weights is None:
            return "", 0.0, []
        prompt = ["<bos>", "user:"] + tokenize(question) + ["assistant:"]
        ids = self.encode_tokens(prompt, False)
        x = self.make_features(ids)
        xb = np.append(x, 1.0)
        raw = xb @ self.weights
        pred = np.rint(raw).astype(int)
        pred = np.clip(pred, PAD_ID, max(PAD_ID, len(self.id_to_token) - 1))

        tokens: List[str] = []
        conf = 0.0
        used = 0
        for raw_val, tid in zip(raw, pred):
            used += 1
            conf += 1.0 / (1.0 + abs(float(raw_val) - int(tid)))
            if tid in (PAD_ID, BOS_ID, USER_ID, ASSISTANT_ID):
                continue
            if tid == EOS_ID:
                break
            tok = self.id_to_token[int(tid)] if 0 <= int(tid) < len(self.id_to_token) else "<unk>"
            if tok != "<unk>":
                tokens.append(tok)

        text = ""
        no_space_before = set(".,;:)]}>?")
        no_space_after = set("([{<#:.'\"")
        prev = ""
        for tok in tokens:
            if tok == "\\n":
                text += "\n"
            elif not text or tok in no_space_before or prev in no_space_after:
                text += tok
            else:
                text += " " + tok
            prev = tok
        return text, conf / max(1, used), tokens

    def save(self, path: str = MODEL_PATH) -> None:
        data = {
            "model_version": MODEL_VERSION,
            "config": {
                "max_tokens": MAX_TOKENS,
                "linear_input_tokens": LINEAR_INPUT_TOKENS,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "hash_buckets": HASH_BUCKETS,
                "feature_size": FEATURE_SIZE,
            },
            "samples": [asdict(s) for s in self.samples],
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "weights": self.weights.tolist() if self.weights is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = [Sample(**s) for s in data.get("samples", [])]
        self.token_to_id = {k: int(v) for k, v in data.get("token_to_id", {}).items()}
        self.id_to_token = list(data.get("id_to_token", SPECIALS))
        w = data.get("weights")
        self.weights = np.array(w, dtype=np.float64) if w is not None else None
        if not self._weights_compatible():
            self.weights = None
        if self.weights is None and self.samples:
            self.train()
            self.save(path)
        return True
