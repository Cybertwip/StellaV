from __future__ import annotations

from typing import Dict, List

from knowledge_base import (
    DISCLAIMER,
    GENERAL_TOPIC_OVERVIEW,
    TOPICS,
    build_usage_items,
    build_vocab_sections,
    generate_dataset_items,
    semantic_response,
)


BOOTSTRAP_VOCAB_SECTIONS: Dict[str, List[str]] = build_vocab_sections()


def bootstrap_usage_items() -> List[dict[str, str]]:
    return [dict(item) for item in build_usage_items()]


def bootstrap_dataset_items() -> List[dict[str, str]]:
    return [dict(item) for item in generate_dataset_items()]


def semantic_fallback_response(question: str) -> tuple[str, float] | None:
    return semantic_response(question)


__all__ = [
    "BOOTSTRAP_VOCAB_SECTIONS",
    "DISCLAIMER",
    "GENERAL_TOPIC_OVERVIEW",
    "TOPICS",
    "bootstrap_dataset_items",
    "bootstrap_usage_items",
    "semantic_fallback_response",
]
