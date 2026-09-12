from __future__ import annotations

from typing import Iterable, List, Tuple

from model import LinearTokenLanguageModel, Sample, tokenize


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in tokenize(text) if t and t not in {";", "(", ")", "{", "}", ","}}


def evaluate_model(model: LinearTokenLanguageModel, val: List[Sample]) -> Tuple[float, float]:
    """Return (avg_confidence, token_match_rate)."""
    if not val:
        return 0.0, 0.0
    confs: list[float] = []
    matches = 0
    for s in val:
        _text, conf, tokens = model.predict(s.question)
        confs.append(conf)
        expected = _token_set(s.answer)
        predicted = {t.lower() for t in tokens}
        if expected & predicted:
            matches += 1
    return sum(confs) / len(confs), matches / len(val)


def train_iterative(
    model: LinearTokenLanguageModel,
    train_set: List[Sample],
    val_set: List[Sample],
    *,
    max_iters: int = 20,
    target_conf: float = 0.95,
    target_match: float = 0.9,
    vocabulary_tokens: Iterable[str] | None = None,
) -> dict:
    """Train the linear model and evaluate it.

    The model first receives the teacher vocabulary lexically, then the actual
    lexical/syntactic/semantic QA samples. Ridge regression is closed-form, so
    one solve is the real training step.
    """
    summary = {"iters": 0, "best_conf": 0.0, "best_match": 0.0}
    if not train_set:
        return summary

    model.clear()
    if vocabulary_tokens:
        model.teach_vocabulary(vocabulary_tokens)

    for s in train_set:
        model.add_sample(s.question, s.answer)

    trained = model.train()
    if not trained:
        return summary

    conf, match = evaluate_model(model, val_set or train_set[:1])
    summary["iters"] = 1
    summary["best_conf"] = conf
    summary["best_match"] = match
    summary["target_reached"] = bool(conf >= target_conf and match >= target_match)
    summary["requested_max_iters"] = max_iters
    summary["vocab_size"] = len(model.id_to_token)

    full_corpus = list(train_set) + list(val_set)
    if full_corpus:
        model.clear()
        if vocabulary_tokens:
            model.teach_vocabulary(vocabulary_tokens)
        for s in full_corpus:
            model.add_sample(s.question, s.answer)
        model.train()

    return summary
