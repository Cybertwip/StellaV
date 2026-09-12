from __future__ import annotations

from bootstrap_data import bootstrap_dataset_items
from model import LinearTokenLanguageModel
from trainer import train_iterative, evaluate_model
from medical_domain_generator import seed_builtin_teacher_vocabulary, seed_builtin_usage_bootstrap, teacher_vocab_tokens
from model import Sample


def main() -> None:
    raw_items = bootstrap_dataset_items()
    samples = [Sample(question=item["question"], answer=item["answer"]) for item in raw_items]
    split = max(1, int(len(samples) * 0.85))
    train_set = samples[:split]
    val_set = samples[split:]

    seed_builtin_teacher_vocabulary()
    seed_builtin_usage_bootstrap()

    model = LinearTokenLanguageModel()
    summary = train_iterative(
        model,
        train_set,
        val_set,
        max_iters=5,
        target_conf=0.6,
        target_match=0.5,
        vocabulary_tokens=teacher_vocab_tokens(),
    )
    print("Train summary:", summary)
    conf, match = evaluate_model(model, val_set)
    print(f"Eval on val: avg_conf={conf:.4f}, match_rate={match:.4f}")


if __name__ == "__main__":
    main()
