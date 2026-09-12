from __future__ import annotations

import random
from typing import List

from bootstrap_data import bootstrap_dataset_items
from model import Sample


def generate_dataset(num_samples: int = 200, seed: int | None = None) -> List[Sample]:
    """Generate a medical-research dataset from the structured knowledge base."""
    items = bootstrap_dataset_items()
    if seed is not None:
        random.seed(seed)

    if not items:
        return []

    shuffled = list(items)
    random.shuffle(shuffled)

    samples: List[Sample] = []
    while len(samples) < num_samples:
        for item in shuffled:
            samples.append(Sample(question=item["question"], answer=item["answer"]))
            if len(samples) >= num_samples:
                break

    return samples
