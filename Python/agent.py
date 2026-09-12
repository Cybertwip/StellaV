#!/usr/bin/env python3
"""
agent.py
─────────
Endless agentic training loop for StellaV + Model.

Important behavior:
  • No local curriculum is used.
  • StellaV first receives teacher vocabulary through compact p lines.
  • The teacher then teaches vocabulary usage through compact q/a CSV token ids
    for medical-research English and methods.
  • Normal QA and corrective QA also use compact p/q/a.
  • Ollama output streams to the terminal.
  • The first valid teacher chunk is appended and trained immediately.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import agent_client
from bootstrap_data import bootstrap_dataset_items
from confidence_tracker import ConfidenceTracker
from medical_domain_generator import (
    TOPICS as MEDICAL_TOPICS,
    ensure_all_category_vocabularies,
    expand_teacher_vocabulary,
    ensure_vocabulary_usage_bootstrap,
    generate_knowledge_batch,
    generate_medical_batch,
    iter_knowledge_batches,
    iter_medical_batches,
    teacher_vocab_tokens,
)
from model import LinearTokenLanguageModel, Sample
from trainer import train_iterative

try:
    from persistent_kb import (
        append_kb_pairs_to_dataset,
        index_default_scienceopen_categories,
        index_scienceopen_queries,
        kb_stats,
        load_kb_into_tensor_engine,
    )
except Exception:  # Keep the base trainer usable even if optional scraper deps are missing.
    append_kb_pairs_to_dataset = None
    index_default_scienceopen_categories = None
    index_scienceopen_queries = None
    kb_stats = None
    load_kb_into_tensor_engine = None

DATASET_PATH = Path("agent_dataset.jsonl")
MODEL_PATH = Path("qa_linear_token_model.json")
TARGET_CONF = 0.95
INITIAL_BATCH = 30
INITIAL_KNOWLEDGE_BATCH = 12
MAX_BATCH = 120
TRAIN_MAX_ITERS = 30
SLEEP_BETWEEN = 2
STAGNATION_BOOST = 20
MAX_FAILURES_SENT = 8
DEFAULT_VOCAB_GROWTH = 0
DEFAULT_KB_PAIR_LIMIT = 1000

TOPIC_RING = list(MEDICAL_TOPICS)


@dataclass
class PredictionFailure:
    question: str
    expected: str
    predicted: str
    confidence: float


def _load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    if not path.exists():
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" in j and "answer" in j:
                samples.append(Sample(question=str(j["question"]), answer=str(j["answer"])))
    return samples


def _existing_pair_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(j.get("question", "")).strip().lower()
            a = str(j.get("answer", "")).strip().lower()
            if q and a:
                keys.add((q, a))
    return keys


def _append_to_dataset(pairs: list[dict], path: Path) -> int:
    if pairs:
        with open(path, "a", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps({"question": p["question"], "answer": p["answer"]}) + "\n")
    return sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0


def _append_unique_to_dataset(pairs: list[dict], path: Path) -> tuple[int, int]:
    keys = _existing_pair_keys(path)
    added = 0
    if pairs:
        with open(path, "a", encoding="utf-8") as f:
            for p in pairs:
                q = str(p.get("question", "")).strip()
                a = str(p.get("answer", "")).strip()
                if not q or not a:
                    continue
                key = (q.lower(), a.lower())
                if key in keys:
                    continue
                keys.add(key)
                added += 1
                f.write(json.dumps({"question": q, "answer": a}) + "\n")
    total = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
    return total, added


def _seed_dataset_from_bootstrap(path: Path, target_conf: float) -> int:
    total, added = _append_unique_to_dataset(bootstrap_dataset_items(), path)
    print(
        f"[agent] Bootstrap dataset ready: added {added} local sample(s), dataset now {total}.",
        flush=True,
    )
    if added > 0:
        print("[agent] Training StellaV on the local bootstrap before Ollama learning…", flush=True)
        summary = _train_stellav_in_process(path, target_conf)
        print(f"[agent] Bootstrap train summary: {summary}", flush=True)

    _sync_persisted_knowledge_dataset(path, target_conf, DEFAULT_KB_PAIR_LIMIT)
    return total


def _sync_persisted_knowledge_dataset(path: Path, target_conf: float, max_pairs: int) -> int:
    """Append persisted KB pairs into the learning dataset without duplication."""
    if append_kb_pairs_to_dataset is None:
        return 0
    try:
        stats = append_kb_pairs_to_dataset(path, max_pairs=max_pairs)
    except Exception as exc:
        print(f"[agent] Persistent KB sync skipped: {exc}", flush=True)
        return 0

    added = int(stats.get("added_pairs", 0))
    if added > 0:
        print(
            f"[agent] Persistent KB sync: added {added} persisted knowledge pair(s); "
            f"dataset now {stats.get('dataset_total', '?')}.",
            flush=True,
        )
        summary = _train_stellav_in_process(path, target_conf)
        print(f"[agent] Persistent KB train summary: {summary}", flush=True)
    return added


def _train_stellav_in_process(dataset_path: Path, target_conf: float) -> dict:
    samples = _load_samples(dataset_path)
    if not samples:
        return {"iters": 0, "best_conf": 0.0, "best_match": 0.0, "error": "empty dataset"}

    split = max(1, int(len(samples) * 0.8))
    if split >= len(samples) and len(samples) > 1:
        split = len(samples) - 1

    train_set = samples[:split]
    val_set = samples[split:] or samples[:1]

    model = LinearTokenLanguageModel()
    summary = train_iterative(
        model,
        train_set,
        val_set,
        max_iters=TRAIN_MAX_ITERS,
        target_conf=target_conf,
        target_match=0.90,
        vocabulary_tokens=teacher_vocab_tokens(),
    )
    model.save(str(MODEL_PATH))
    return summary


def _run_main(args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "main.py"] + args,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode, result.stdout + result.stderr


def _parse_train_summary(text: str) -> dict:
    m = re.search(r"Training summary:\s*(\{.*\})", text)
    if m:
        try:
            return json.loads(m.group(1).replace("'", '"'))
        except Exception:
            pass
    return {}


def _collect_stellav_failures(
    dataset_path: Path,
    model_path: Path,
    n_eval: int = 100,
    max_failures: int = MAX_FAILURES_SENT,
) -> tuple[float, float, list[PredictionFailure]]:
    model = LinearTokenLanguageModel()
    if not model.load(str(model_path)):
        return 0.0, 0.0, []

    all_samples = _load_samples(dataset_path)
    if not all_samples:
        return 0.0, 0.0, []

    eval_set = random.sample(all_samples, min(n_eval, len(all_samples)))

    confs: list[float] = []
    matches = 0
    failures: list[PredictionFailure] = []

    for s in eval_set:
        predicted_text, conf, predicted_tokens = model.predict(s.question)
        confs.append(conf)

        expected_tokens = set(s.answer.lower().split())
        predicted_set = set(t.lower() for t in predicted_tokens)
        hit = bool(expected_tokens & predicted_set)

        if hit:
            matches += 1
        else:
            failures.append(PredictionFailure(
                question=s.question,
                expected=s.answer,
                predicted=predicted_text or "(empty)",
                confidence=conf,
            ))

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    match_rate = matches / len(eval_set) if eval_set else 0.0
    failures.sort(key=lambda f: f.confidence)
    return avg_conf, match_rate, failures[:max_failures]


def _ensure_vocabulary_and_usage_dataset(
    topic: str,
    iteration: int,
    target_conf: float,
    vocab_growth: int,
) -> int:
    print("[agent] Ensuring all separated vocabularies are learned first as append-only p lines…", flush=True)
    ensure_all_category_vocabularies(iteration)

    if vocab_growth > 0:
        print(
            f"[agent] Asking Ollama to extend active vocabularies "
            f"(language, code, {topic}) by about {vocab_growth} token(s) each…",
            flush=True,
        )
        expand_teacher_vocabulary(
            iteration,
            sections=["language", "methods", topic],
            extra_tokens_per_section=vocab_growth,
        )

    print("[agent] Ensuring separated vocabulary usage bootstrap for language, methods, and medical categories…", flush=True)
    usage_pairs = ensure_vocabulary_usage_bootstrap(topic, iteration)
    total, added = _append_unique_to_dataset(usage_pairs, DATASET_PATH)
    print(
        f"[agent] Vocabulary usage bootstrap ready: {len(usage_pairs)} pair(s), "
        f"added {added}, dataset now {total}.",
        flush=True,
    )

    if added > 0:
        print("[agent] Training StellaV on vocabulary usage bootstrap…", flush=True)
        summary = _train_stellav_in_process(DATASET_PATH, target_conf)
        print(f"[agent] Vocabulary usage train summary: {summary}", flush=True)

    return total


def _append_generated_batch_and_warm_train(
    *,
    chunks: Iterable[list[dict]],
    batch_size: int,
    label: str,
    target_conf: float,
) -> tuple[int, list[dict]]:
    pairs: list[dict] = []
    total_samples = _append_to_dataset([], DATASET_PATH)
    warmed = False

    for chunk_index, chunk in enumerate(chunks, start=1):
        pairs.extend(chunk)
        total_samples, added = _append_unique_to_dataset(chunk, DATASET_PATH)

        print(
            f"[agent] Accepted {label} chunk {chunk_index}: "
            f"{len(chunk)} parsed pair(s), {added} new, {len(pairs)}/{batch_size} for this iteration.",
            flush=True,
        )
        print(f"[agent] Dataset now has {total_samples} samples.", flush=True)

        if not warmed and added > 0:
            print(f"[agent] First {label} chunk accepted — training StellaV immediately…", flush=True)
            warm_summary = _train_stellav_in_process(DATASET_PATH, target_conf)
            print(f"[agent] Warm train summary: {warm_summary}", flush=True)
            warmed = True

        if len(pairs) >= batch_size:
            break

    return total_samples, pairs[:batch_size]


def _append_teacher_batch_and_warm_train(
    *,
    batch_size: int,
    topic: str,
    iteration: int,
    target_conf: float,
) -> tuple[int, list[dict]]:
    return _append_generated_batch_and_warm_train(
        chunks=iter_medical_batches(n=batch_size, topic=topic, iteration=iteration),
        batch_size=batch_size,
        label="medical teacher",
        target_conf=target_conf,
    )


def _append_knowledge_batch_and_warm_train(
    *,
    batch_size: int,
    topic: str,
    iteration: int,
    target_conf: float,
) -> tuple[int, list[dict]]:
    return _append_generated_batch_and_warm_train(
        chunks=iter_knowledge_batches(n=batch_size, topic=topic, iteration=iteration),
        batch_size=batch_size,
        label="knowledge teacher",
        target_conf=target_conf,
    )


def _ask_teacher_corrections(
    failures: list[PredictionFailure],
    n: int,
    topics: list[str],
    iteration: int,
) -> list[dict]:
    topic = topics[-1] if topics else "clinical_trials"
    failed_questions = [f.question for f in failures]
    if not failed_questions:
        failed_questions = [
            "correction for a weak medical-research explanation about clinical trial methods"
        ]

    return generate_medical_batch(
        n=n,
        topic=topic,
        iteration=iteration,
        failed_questions=failed_questions,
    )


def _ask_teacher_knowledge_corrections(
    failures: list[PredictionFailure],
    n: int,
    topics: list[str],
    iteration: int,
) -> list[dict]:
    topic = topics[-1] if topics else "clinical_trials"
    failed_questions = [f.question for f in failures]
    if not failed_questions:
        failed_questions = [
            "brief explanation correction for a weak natural-language answer about medical research methods"
        ]

    return generate_knowledge_batch(
        n=n,
        topic=topic,
        iteration=iteration,
        failed_questions=failed_questions,
    )


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _topic_plan_for_iteration(
    *,
    forced_topic: Optional[str],
    iteration: int,
    tracker: ConfidenceTracker,
    learn_all_categories: bool,
) -> list[str]:
    valid_topics = set(TOPIC_RING)
    if forced_topic in valid_topics:
        return [forced_topic]
    if learn_all_categories:
        ordered = list(TOPIC_RING)
        if tracker.is_stagnant():
            weak = [t for t in tracker.worst_topics() if t in ordered]
            return _dedupe_preserve_order(weak + ordered)
        return ordered
    return [TOPIC_RING[(iteration - 1) % len(TOPIC_RING)]]


def _index_scienceopen_for_learning(
    *,
    scienceopen_enabled: bool,
    scienceopen_default: bool,
    scienceopen_queries: Optional[list[str]],
    kb_pair_limit: int,
    target_conf: float,
) -> None:
    if not scienceopen_enabled:
        print("[agent] ScienceOpen indexing disabled for this run.", flush=True)
        return
    if index_scienceopen_queries is None:
        print("[agent] ScienceOpen indexing unavailable: optional network dependencies are missing.", flush=True)
        return

    if scienceopen_default:
        if index_default_scienceopen_categories is None:
            print("[agent] Default ScienceOpen category indexing unavailable.", flush=True)
        else:
            print("[agent] Default ScienceOpen learning: indexing medical preprint queries…", flush=True)
            result = index_default_scienceopen_categories(
                dataset_path=DATASET_PATH,
                max_dataset_pairs=kb_pair_limit,
            )
            print(f"[agent] Default ScienceOpen index result: {result}", flush=True)
            _sync_persisted_knowledge_dataset(DATASET_PATH, target_conf, kb_pair_limit)

    clean_queries = _dedupe_preserve_order(scienceopen_queries or [])
    if clean_queries:
        print(f"[agent] Indexing {len(clean_queries)} extra ScienceOpen query/DOI(s) into the medical KB…", flush=True)
        result = index_scienceopen_queries(clean_queries, dataset_path=DATASET_PATH, max_dataset_pairs=kb_pair_limit)
        print(f"[agent] Extra ScienceOpen index result: {result}", flush=True)
        _sync_persisted_knowledge_dataset(DATASET_PATH, target_conf, kb_pair_limit)

    if load_kb_into_tensor_engine is not None:
        try:
            n = load_kb_into_tensor_engine()
            print(f"[agent] Tensor engine loaded {n} preprint chunk(s).", flush=True)
        except Exception as exc:
            print(f"[agent] Tensor engine load skipped: {exc}", flush=True)

    if kb_stats is not None:
        try:
            print(f"[agent] Persistent KB stats after ScienceOpen sync: {kb_stats()}", flush=True)
        except Exception as exc:
            print(f"[agent] KB stats unavailable: {exc}", flush=True)


def run_agent(
    target_conf: float = TARGET_CONF,
    initial_batch: int = INITIAL_BATCH,
    knowledge_batch: int = INITIAL_KNOWLEDGE_BATCH,
    vocab_growth: int = DEFAULT_VOCAB_GROWTH,
    max_iters: Optional[int] = None,
    fixed_topic: Optional[str] = None,
    scienceopen_queries: Optional[list[str]] = None,
    scienceopen_default: bool = True,
    scienceopen_enabled: bool = True,
    learn_all_categories: bool = True,
    kb_pair_limit: int = DEFAULT_KB_PAIR_LIMIT,
    reason_size: str = "1.5b",
) -> None:
    print(
        "\n"
        "Stella V medical-research trainer\n"
        f"Reasoner={reason_size}  Source=ScienceOpen preprints  Domain=medical research\n",
        flush=True,
    )

    tracker = ConfidenceTracker.load()
    _index_scienceopen_for_learning(
        scienceopen_enabled=scienceopen_enabled,
        scienceopen_default=scienceopen_default,
        scienceopen_queries=scienceopen_queries,
        kb_pair_limit=kb_pair_limit,
        target_conf=target_conf,
    )
    tracker.target_conf = target_conf
    _seed_dataset_from_bootstrap(DATASET_PATH, target_conf)

    if tracker.records:
        start_iter = tracker.records[-1].iteration + 1
        print(
            f"[agent] Resuming from iteration {start_iter}  "
            f"(best so far: {tracker.best_conf * 100:.1f}%)",
            flush=True,
        )
    else:
        start_iter = 1
        print("[agent] Starting fresh run.", flush=True)

    batch_size = initial_batch
    valid_topics = set(TOPIC_RING)
    forced_topic = fixed_topic if fixed_topic in valid_topics else None
    recent_topics: list[str] = []
    stop_iter = (max_iters + start_iter) if max_iters is not None else 10_000

    for iteration in range(start_iter, stop_iter):
        print(f"\n[agent] ══ Iteration {iteration} ══", flush=True)

        topics_this_iter = _topic_plan_for_iteration(
            forced_topic=forced_topic,
            iteration=iteration,
            tracker=tracker,
            learn_all_categories=learn_all_categories,
        )
        if learn_all_categories and not forced_topic:
            print(
                "[agent] All-category learning plan: " + ", ".join(topics_this_iter),
                flush=True,
            )
        elif forced_topic:
            batch_size = initial_batch

        if tracker.is_stagnant() and not forced_topic:
            batch_size = min(batch_size + STAGNATION_BOOST, MAX_BATCH)
            print(f"[agent] Stagnation detected — raising per-category batch to {batch_size}.", flush=True)

        last_total_samples = _append_to_dataset([], DATASET_PATH)

        for topic in topics_this_iter:
            print(f"\n[agent] ---- Category: {topic} ----", flush=True)
            recent_topics.append(topic)
            if len(recent_topics) > 12:
                recent_topics.pop(0)

            _ensure_vocabulary_and_usage_dataset(topic, iteration, target_conf, vocab_growth)
            _sync_persisted_knowledge_dataset(DATASET_PATH, target_conf, kb_pair_limit)

            if knowledge_batch > 0:
                print(
                    f"[agent] Generating {knowledge_batch} natural-language teacher QA pairs "
                    f"(topic={topic}, iter={iteration})…",
                    flush=True,
                )
                total_samples, knowledge_pairs = _append_knowledge_batch_and_warm_train(
                    batch_size=knowledge_batch,
                    topic=topic,
                    iteration=iteration,
                    target_conf=target_conf,
                )
                print(f"[agent] Got {len(knowledge_pairs)} valid natural-language teacher pairs.", flush=True)
            else:
                total_samples = _append_to_dataset([], DATASET_PATH)

            print(
                f"[agent] Generating {batch_size} compact teacher QA pairs "
                f"(topic={topic}, iter={iteration})…",
                flush=True,
            )

            total_samples, pairs = _append_teacher_batch_and_warm_train(
                batch_size=batch_size,
                topic=topic,
                iteration=iteration,
                target_conf=target_conf,
            )
            last_total_samples = total_samples
            print(f"[agent] Got {len(pairs)} valid compact teacher pairs.", flush=True)

            print("[agent] Training after this category…", flush=True)
            train_summary = _train_stellav_in_process(DATASET_PATH, target_conf)
            print(f"[agent] {train_summary}", flush=True)

            print("[agent] Evaluating StellaV outputs…", flush=True)
            avg_conf, match_rate, failures = _collect_stellav_failures(DATASET_PATH, MODEL_PATH)
            print(
                f"[agent] topic={topic} avg_conf={avg_conf:.4f}  match_rate={match_rate:.4f}  "
                f"failures={len(failures)}",
                flush=True,
            )

            if failures:
                print("[agent] Sample failure:", flush=True)
                f0 = failures[0]
                print(f"        Q: {f0.question}", flush=True)
                print(f"        Expected : {f0.expected}", flush=True)
                print(f"        Predicted: {f0.predicted}", flush=True)

            tracker.record(
                iteration=iteration,
                topic=topic,
                dataset_size=total_samples,
                avg_conf=avg_conf,
                match_rate=match_rate,
            )
            print(tracker.report(), flush=True)

            if avg_conf >= target_conf:
                print(
                    f"\n🎯  Target reached!  avg_conf = {avg_conf * 100:.1f}% "
                    f"≥ {target_conf * 100:.0f}%\n"
                    f"    Iteration       : {iteration}\n"
                    f"    Category        : {topic}\n"
                    f"    Total QA samples : {total_samples}\n"
                    f"    Best confidence  : {tracker.best_conf * 100:.1f}%\n"
                    f"\n    StellaV is ready.\n",
                    flush=True,
                )
                return

            if knowledge_batch > 0:
                knowledge_corrective_n = min(max(4, len(failures)), knowledge_batch)
                print(
                    f"[agent] Asking compact teacher for {knowledge_corrective_n} natural-language corrective q/a pair(s)…",
                    flush=True,
                )
                corrective_knowledge = _ask_teacher_knowledge_corrections(
                    failures=failures,
                    n=knowledge_corrective_n,
                    topics=recent_topics,
                    iteration=iteration,
                )

                if corrective_knowledge:
                    added_total, added = _append_unique_to_dataset(corrective_knowledge, DATASET_PATH)
                    print(
                        f"[agent] Added {added}/{len(corrective_knowledge)} natural-language corrective pair(s) "
                        f"(dataset now {added_total}).",
                        flush=True,
                    )

                    if added > 0:
                        print("[agent] Training StellaV on natural-language corrective pairs…", flush=True)
                        knowledge_summary = _train_stellav_in_process(DATASET_PATH, target_conf)
                        print(f"[agent] Natural-language corrective train summary: {knowledge_summary}", flush=True)

            corrective_n = min(len(failures) * 3 or 10, 30)
            print(
                f"[agent] Asking compact teacher for {corrective_n} corrective q/a pair(s) "
                f"from {len(failures)} StellaV failure(s)…",
                flush=True,
            )
            corrective = _ask_teacher_corrections(
                failures=failures,
                n=corrective_n,
                topics=recent_topics,
                iteration=iteration,
            )

            if corrective:
                added_total, added = _append_unique_to_dataset(corrective, DATASET_PATH)
                print(
                    f"[agent] Added {added}/{len(corrective)} corrective compact teacher pairs "
                    f"(dataset now {added_total}).",
                    flush=True,
                )

                if added > 0:
                    print("[agent] Training StellaV on corrective compact teacher pairs…", flush=True)
                    corrective_summary = _train_stellav_in_process(DATASET_PATH, target_conf)
                    print(f"[agent] Corrective train summary: {corrective_summary}", flush=True)

            time.sleep(SLEEP_BETWEEN)

        if learn_all_categories and not forced_topic:
            print(
                f"[agent] Completed all categories for iteration {iteration}; dataset now {last_total_samples} sample(s).",
                flush=True,
            )

    print(
        f"\n[agent] Reached max_iters={max_iters}.  "
        f"Best confidence: {tracker.best_conf * 100:.1f}%",
        flush=True,
    )


def _parse_extra_slug_args(slugs: list[str] | None, slug_file: str | None = None) -> list[str]:
    out: list[str] = []
    for raw in slugs or []:
        for part in str(raw).split(','):
            part = part.strip()
            if part:
                out.append(part)
    if slug_file:
        path = Path(slug_file)
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    out.append(line)
    return _dedupe_preserve_order(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="StellaV agentic trainer — sectioned append-only Ollama p/q/a teacher with vocabulary growth, natural-language knowledge pairs, and code QA pairs."
    )
    p.add_argument("--target-conf", type=float, default=TARGET_CONF)
    p.add_argument("--batch", type=int, default=INITIAL_BATCH)
    p.add_argument("--knowledge-batch", type=int, default=INITIAL_KNOWLEDGE_BATCH)
    p.add_argument(
        "--vocab-growth",
        type=int,
        default=DEFAULT_VOCAB_GROWTH,
        help="Approximate new tokens to request per active section per iteration. Default is 0 for safer startup.",
    )
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--topic", type=str, default="")
    p.add_argument("--scienceopen-query", action="append", default=[], help="Extra ScienceOpen query or DOI to index in addition to the default medical manifest.")
    p.add_argument("--scienceopen-query-file", type=str, default="", help="File with one extra ScienceOpen query or DOI per line.")
    p.add_argument("--no-scienceopen", action="store_true", help="Disable ScienceOpen indexing for this run.")
    p.add_argument("--no-scienceopen-default", action="store_true", help="Skip the default medical ScienceOpen manifest.")
    p.add_argument("--reason", type=str, default="1.5b", help="Reasoning model size: 1.5b or 3b.")
    p.add_argument("--single-topic", action="store_true", help="Only learn one topic instead of walking every category by default.")
    p.add_argument("--kb-pair-limit", type=int, default=DEFAULT_KB_PAIR_LIMIT)
    p.add_argument(
        "--no-ollama-check",
        action="store_true",
        help="Skip startup reachability check. Generation will still wait for Ollama.",
    )
    args = p.parse_args()

    if not args.no_ollama_check:
        if not agent_client.is_alive():
            print(
                f"[agent] Ollama is not reachable at {agent_client.OLLAMA_BASE}.\n"
                "        Run `python setup_ollama.py` first, or start Ollama manually with `ollama serve`.",
                file=sys.stderr,
            )
            sys.exit(1)

        models = agent_client.available_models()
        present = any(
            agent_client.MODEL_NAME == m or m.startswith(agent_client.MODEL_NAME + ":")
            for m in models
        )
        suffix = "" if present else "  (not listed by Ollama tags; generation may fail until pulled)"
        print(f"[agent] Ollama OK  •  model={agent_client.MODEL_NAME}{suffix}", flush=True)

    run_agent(
        target_conf=args.target_conf,
        initial_batch=args.batch,
        knowledge_batch=args.knowledge_batch,
        vocab_growth=args.vocab_growth,
        max_iters=args.max_iters,
        fixed_topic=args.topic or None,
        scienceopen_queries=_parse_extra_slug_args(args.scienceopen_query, args.scienceopen_query_file),
        scienceopen_default=not args.no_scienceopen_default,
        scienceopen_enabled=not args.no_scienceopen,
        reason_size=args.reason,
        learn_all_categories=not args.single_topic,
        kb_pair_limit=args.kb_pair_limit,
    )


if __name__ == "__main__":
    main()
