#!/usr/bin/env python3
"""CLI for Stella V: ScienceOpen medical-research retrieval plus a 1.5B/3B reasoner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from model import LinearTokenLanguageModel, Sample
from bootstrap_data import bootstrap_dataset_items
from dataset_generator import generate_dataset
from trainer import train_iterative, evaluate_model
from medical_domain_generator import seed_builtin_teacher_vocabulary, seed_builtin_usage_bootstrap, teacher_vocab_tokens
from persistent_kb import (
    append_kb_pairs_to_dataset,
    default_scienceopen_queries,
    index_scienceopen_queries,
    index_text,
    kb_stats,
    load_kb_into_tensor_engine,
)
from reasoner import normalize_reason_size, reason
from tensor_engine import get_engine


BOOTSTRAP_DATASET_PATH = Path("bootstrap_dataset.jsonl")


def cmd_generate(args: argparse.Namespace) -> None:
    data = generate_dataset(num_samples=args.num, seed=args.seed)
    path = args.out
    # save as json lines (simple)
    import json

    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps({"question": s.question, "answer": s.answer}) + "\n")
    print(f"Wrote {len(data)} samples to {path}")


def load_samples_from_file(path: str) -> List[Sample]:
    import json

    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            out.append(Sample(question=j["question"], answer=j["answer"]))
    return out


def _resolve_dataset_path(path: str) -> str:
    if path:
        return path
    if BOOTSTRAP_DATASET_PATH.exists():
        return str(BOOTSTRAP_DATASET_PATH)
    return ""


def _bootstrap_vocabulary_tokens() -> List[str]:
    seed_builtin_teacher_vocabulary()
    seed_builtin_usage_bootstrap()
    return teacher_vocab_tokens()


def _write_bootstrap_dataset(path: str) -> List[Sample]:
    import json

    items = bootstrap_dataset_items()
    samples = [Sample(question=item["question"], answer=item["answer"]) for item in items]
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps({"question": sample.question, "answer": sample.answer}) + "\n")
    return samples


def _split_samples(samples: List[Sample]) -> tuple[List[Sample], List[Sample]]:
    if len(samples) <= 1:
        return samples, samples
    split = max(1, int(len(samples) * 0.85))
    if split >= len(samples):
        split = len(samples) - 1
    train_set = samples[:split]
    val_set = samples[split:] or samples[:1]
    return train_set, val_set


def cmd_bootstrap(args: argparse.Namespace) -> None:
    samples = _write_bootstrap_dataset(args.out)
    vocab_tokens = _bootstrap_vocabulary_tokens()
    train_set, val_set = _split_samples(samples)

    model = LinearTokenLanguageModel()
    summary = train_iterative(
        model,
        train_set,
        val_set,
        max_iters=args.max_iters,
        target_conf=args.target_conf,
        target_match=args.target_match,
        vocabulary_tokens=vocab_tokens,
    )
    model.save(args.model_out)
    full_conf, full_match = evaluate_model(model, samples)

    print(f"Wrote {len(samples)} bootstrap samples to {args.out}")
    print(f"Prepared {len(vocab_tokens)} vocabulary tokens in agent_teacher_vocab.json")
    print("Training summary:", summary)
    print(f"Full dataset eval: avg_conf={full_conf:.4f}, match_rate={full_match:.4f}")
    print(f"Saved model to {args.model_out}")


def cmd_train(args: argparse.Namespace) -> None:
    # prepare dataset
    data_path = _resolve_dataset_path(args.data)
    if data_path:
        samples = load_samples_from_file(data_path)
    else:
        samples = generate_dataset(num_samples=args.num, seed=args.seed)

    # split
    train_set, val_set = _split_samples(samples)

    model = LinearTokenLanguageModel()
    vocabulary_tokens = _bootstrap_vocabulary_tokens() if data_path else None
    summary = train_iterative(
        model,
        train_set,
        val_set,
        max_iters=args.max_iters,
        target_conf=args.target_conf,
        target_match=args.target_match,
        vocabulary_tokens=vocabulary_tokens,
    )
    full_conf, full_match = evaluate_model(model, samples)
    print("Training summary:", summary)
    print(f"Full dataset eval: avg_conf={full_conf:.4f}, match_rate={full_match:.4f}")
    model.save()


def cmd_eval(args: argparse.Namespace) -> None:
    model = LinearTokenLanguageModel()
    loaded = model.load()
    if not loaded:
        print("No saved model found. Train first.")
        return
    data_path = _resolve_dataset_path(args.data)
    if data_path:
        samples = load_samples_from_file(data_path)
    else:
        samples = generate_dataset(num_samples=args.num, seed=args.seed)
    conf, match = evaluate_model(model, samples)
    print(f"Eval: avg_conf={conf:.4f}, match_rate={match:.4f}")


def cmd_learn(args: argparse.Namespace) -> None:
    import agent_client
    from agent import run_agent

    if not args.no_ollama_check:
        if not agent_client.is_alive():
            print(
                f"Ollama is not reachable at {agent_client.OLLAMA_BASE}.\n"
                "Run `python setup_ollama.py` first, or start Ollama manually with `ollama serve`.",
                file=sys.stderr,
            )
            return

        models = agent_client.available_models()
        present = any(
            agent_client.MODEL_NAME == m or m.startswith(agent_client.MODEL_NAME + ":")
            for m in models
        )
        suffix = "" if present else "  (not listed by Ollama tags; generation may fail until pulled)"
        print(f"Ollama OK  •  model={agent_client.MODEL_NAME}{suffix}")

    run_agent(
        target_conf=args.target_conf,
        initial_batch=args.batch,
        knowledge_batch=args.knowledge_batch,
        vocab_growth=args.vocab_growth,
        max_iters=args.max_iters,
        fixed_topic=args.topic or None,
        scienceopen_queries=_parse_slug_args(args.scienceopen_query, args.scienceopen_query_file),
        scienceopen_default=not args.no_scienceopen_default,
        scienceopen_enabled=not args.no_scienceopen,
        learn_all_categories=not args.single_topic,
        kb_pair_limit=args.kb_pair_limit,
        reason_size=normalize_reason_size(args.reason),
    )


def _parse_slug_args(slugs: list[str] | None, slug_file: str | None = None) -> list[str]:
    out: list[str] = []
    for raw in slugs or []:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                out.append(part)
    if slug_file:
        path = Path(slug_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(line)
    return list(dict.fromkeys(out))



def cmd_scienceopen_defaults(args: argparse.Namespace) -> None:
    print(default_scienceopen_queries())


def cmd_index_scienceopen(args: argparse.Namespace) -> None:
    queries = _parse_slug_args(args.query, args.query_file)
    if not queries:
        print("No ScienceOpen queries or DOIs were provided.", file=sys.stderr)
        return
    result = index_scienceopen_queries(
        queries,
        rows_per_query=args.rows,
        category_hint=args.category or None,
        dataset_path=Path(args.dataset) if args.dataset else None,
        max_dataset_pairs=args.kb_pair_limit,
    )
    print("Indexed ScienceOpen preprints:", result)
    print("KB stats:", kb_stats())


def cmd_index_text(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        print(f"Text file not found: {path}", file=sys.stderr)
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    result = index_text(
        title=args.title or path.stem,
        slug=args.slug or path.stem,
        content_text=text,
        url=args.url or f"file://{path.resolve()}",
        category_hint=args.category or None,
        teach_vocab=True,
    )
    dataset = append_kb_pairs_to_dataset(Path(args.dataset), max_pairs=args.kb_pair_limit) if args.dataset else {}
    print("Indexed text knowledge:", result)
    if dataset:
        print("Dataset sync:", dataset)
    print("KB stats:", kb_stats())


def cmd_kb_stats(args: argparse.Namespace) -> None:
    print(kb_stats())


def cmd_ask(args: argparse.Namespace) -> None:
    load_kb_into_tensor_engine()
    model = LinearTokenLanguageModel()
    model.load(args.model)
    local_text, local_conf, _tokens = model.predict(args.question)
    size = normalize_reason_size(args.reason)
    if args.local_only:
        print(f"{local_text}\nconfidence={local_conf:.4f}\nsource=local\nreason={size}\ndevice={get_engine().status()['device']}")
        return
    result = reason(
        args.question,
        size=size,
        local_draft=local_text or "",
        live_search=not args.no_live_search,
    )
    print(result["text"])
    print(
        f"confidence={result['confidence']:.4f} source={result['source']} "
        f"reason={result['reason_size']} model={result['model']} "
        f"device={result['engine'].get('device')} hits={len(result['hits'])}"
    )


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stellav")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate")
    g.add_argument("--num", type=int, default=200)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--out", type=str, default="dataset.jsonl")
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("bootstrap")
    b.add_argument("--out", type=str, default=str(BOOTSTRAP_DATASET_PATH))
    b.add_argument("--model-out", type=str, default="qa_linear_token_model.json")
    b.add_argument("--max-iters", type=int, default=5)
    b.add_argument("--target-conf", type=float, default=0.55)
    b.add_argument("--target-match", type=float, default=0.5)
    b.set_defaults(func=cmd_bootstrap)

    l = sub.add_parser("learn")
    l.add_argument("--batch", type=int, default=16)
    l.add_argument("--knowledge-batch", type=int, default=8)
    l.add_argument(
        "--vocab-growth",
        type=int,
        default=0,
        help="Approximate new tokens to request per active section per iteration. Default is 0 for safer startup.",
    )
    l.add_argument("--max-iters", type=int, default=1)
    l.add_argument("--target-conf", type=float, default=0.95)
    l.add_argument("--topic", type=str, default="")
    l.add_argument(
        "--scienceopen-query",
        action="append",
        default=[],
        help="ScienceOpen search query or DOI to fetch, dedupe, persist, and teach before this learning run.",
    )
    l.add_argument(
        "--scienceopen-query-file",
        type=str,
        default="",
        help="Text file with one ScienceOpen query or DOI per line.",
    )
    l.add_argument(
        "--no-scienceopen",
        action="store_true",
        help="Disable ScienceOpen indexing for this run. By default, learn indexes the medical preprint manifest first.",
    )
    l.add_argument(
        "--no-scienceopen-default",
        action="store_true",
        help="Do not index the built-in medical ScienceOpen manifest; only index queries passed with --scienceopen-query.",
    )
    l.add_argument(
        "--reason",
        type=str,
        default="1.5b",
        help="Reasoning model size: 1.5b or 3b.",
    )
    l.add_argument(
        "--single-topic",
        action="store_true",
        help="Only train the selected --topic or rotating topic. By default, learn walks every medical category.",
    )
    l.add_argument(
        "--kb-pair-limit",
        type=int,
        default=1000,
        help="Maximum persisted-KB QA pairs to sync into the learning dataset.",
    )
    l.add_argument(
        "--no-ollama-check",
        action="store_true",
        help="Skip startup reachability check. Generation will still wait for Ollama.",
    )
    l.set_defaults(func=cmd_learn)

    t = sub.add_parser("train")
    t.add_argument("--num", type=int, default=200)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--max-iters", type=int, default=20)
    t.add_argument("--target-conf", type=float, default=0.95)
    t.add_argument("--target-match", type=float, default=0.9)
    t.add_argument("--data", type=str, default="")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--num", type=int, default=100)
    e.add_argument("--seed", type=int, default=None)
    e.add_argument("--data", type=str, default="")
    e.set_defaults(func=cmd_eval)


    gd = sub.add_parser("scienceopen-defaults")
    gd.set_defaults(func=cmd_scienceopen_defaults)

    kg = sub.add_parser("index-scienceopen")
    kg.add_argument("query", nargs="*", help="ScienceOpen queries or DOIs. Commas are accepted.")
    kg.add_argument("--query-file", type=str, default="")
    kg.add_argument("--rows", type=int, default=6)
    kg.add_argument("--category", type=str, default="")
    kg.add_argument("--dataset", type=str, default="agent_dataset.jsonl")
    kg.add_argument("--kb-pair-limit", type=int, default=1000)
    kg.set_defaults(func=cmd_index_scienceopen)

    kt = sub.add_parser("index-text")
    kt.add_argument("file", type=str)
    kt.add_argument("--title", type=str, default="")
    kt.add_argument("--slug", type=str, default="")
    kt.add_argument("--url", type=str, default="")
    kt.add_argument("--category", type=str, default="")
    kt.add_argument("--dataset", type=str, default="agent_dataset.jsonl")
    kt.add_argument("--kb-pair-limit", type=int, default=1000)
    kt.set_defaults(func=cmd_index_text)

    ks = sub.add_parser("kb-stats")
    ks.set_defaults(func=cmd_kb_stats)

    a = sub.add_parser("ask")
    a.add_argument("question", type=str)
    a.add_argument("--model", type=str, default="qa_linear_token_model.json")
    a.add_argument("--reason", type=str, default="1.5b", help="Reasoning model size: 1.5b or 3b.")
    a.add_argument("--local-only", action="store_true", help="Skip the 1.5B/3B reasoner and live ScienceOpen search.")
    a.add_argument("--no-live-search", action="store_true", help="Use the local tensor index only; do not query ScienceOpen at ask time.")
    a.set_defaults(func=cmd_ask)

    ui = sub.add_parser("gui")
    ui.set_defaults(func=lambda args: _run_gui())

    sv = sub.add_parser("serve")
    sv.add_argument("--addr", type=str, default="127.0.0.1:8765")
    sv.set_defaults(func=lambda args: _run_openai_serve(args.addr))

    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    args.func(args)
    return 0


def _run_openai_serve(addr: str) -> None:
    from openai_api import serve_openai
    from persistent_kb import load_kb_into_tensor_engine

    load_kb_into_tensor_engine()
    serve_openai(addr)


def _run_gui() -> None:
    # GUI is optional and imported lazily to avoid pygame requirement for CLI runs
    try:
        from ui import run_gui
    except Exception as e:
        print("GUI unavailable:", e)
        return

    seed_builtin_teacher_vocabulary()
    seed_builtin_usage_bootstrap()
    model = LinearTokenLanguageModel()
    loaded = model.load()
    if (not loaded) or not model.samples:
        samples = _write_bootstrap_dataset(str(BOOTSTRAP_DATASET_PATH))
        train_set, val_set = _split_samples(samples)
        train_iterative(
            model,
            train_set,
            val_set,
            max_iters=5,
            target_conf=0.55,
            target_match=0.5,
            vocabulary_tokens=teacher_vocab_tokens(),
        )
        model.save()
    run_gui(model)


if __name__ == "__main__":
    raise SystemExit(main())
