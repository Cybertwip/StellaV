from __future__ import annotations

"""Persistent Stella V medical knowledge indexed from ScienceOpen preprints."""

import hashlib
import json
import os
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from scienceopen import (
    DEFAULT_CATEGORY_QUERIES,
    default_scienceopen_queries,
    fetch_scienceopen_doi,
    paper_text,
    search_scienceopen,
    scienceopen_url,
    strip_jats,
)
from tensor_engine import get_engine

KB_PATH = Path(os.environ.get("STELLAV_KB_PATH", "stellav_knowledge_base.json"))
DEFAULT_DATASET_PATH = Path(os.environ.get("STELLAV_DATASET", "agent_dataset.jsonl"))

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+.\-]*|[0-9]+(?:\.[0-9]+)?|[^\w\s]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9_+.\-]+", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

TOPIC_SECTIONS = {
    "clinical_trials",
    "epidemiology",
    "oncology",
    "cardiology",
    "immunology",
    "infectious",
    "pharmacology",
    "genomics",
    "neurology",
    "public_health",
}
ALL_SECTIONS = ["language", "methods", *sorted(TOPIC_SECTIONS)]

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "methods": {
        "randomization", "blinding", "endpoint", "confounding", "bias", "cohort", "p-value",
        "hazard", "odds", "power", "sample", "protocol", "consort", "prisma",
    },
    "clinical_trials": {
        "trial", "rct", "randomized", "placebo", "arm", "endpoint", "phase", "blinding",
        "allocation", "consort", "protocol", "adverse",
    },
    "epidemiology": {
        "epidemiology", "incidence", "prevalence", "cohort", "case-control", "confounding",
        "risk", "odds", "population", "observational",
    },
    "oncology": {
        "cancer", "tumor", "tumour", "oncology", "chemotherapy", "immunotherapy", "checkpoint",
        "pd-1", "metastasis", "staging", "radiation",
    },
    "cardiology": {
        "heart", "cardiac", "cardiology", "atherosclerosis", "infarction", "stroke",
        "hypertension", "failure", "coronary", "lipid",
    },
    "immunology": {
        "immune", "immunology", "vaccine", "antibody", "antigen", "t-cell", "b-cell",
        "innate", "adaptive", "cytokine",
    },
    "infectious": {
        "infection", "infectious", "pathogen", "virus", "bacteria", "sepsis", "resistance",
        "antimicrobial", "antibiotic", "vaccine",
    },
    "pharmacology": {
        "drug", "pharmacology", "pharmacokinetics", "pharmacodynamics", "dose", "exposure",
        "adverse", "metabolism", "clearance", "therapeutic",
    },
    "genomics": {
        "gene", "genomic", "gwas", "crispr", "variant", "mutation", "precision", "sequencing",
        "polygenic", "transcript",
    },
    "neurology": {
        "neuro", "alzheimer", "parkinson", "stroke", "dementia", "seizure", "csf",
        "amyloid", "brain", "neuron",
    },
    "public_health": {
        "public", "screening", "prevention", "equity", "population", "program", "uptake",
        "policy", "vaccination", "health-system",
    },
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "have", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "use",
    "uses", "using", "what", "when", "where", "which", "why", "with", "without", "does", "do", "about",
}


def now_ts() -> int:
    return int(time.time())


def normalize_slug(value: str) -> str:
    raw = urllib.parse.unquote(str(value or "").strip())
    raw = raw.replace("https://doi.org/", "").replace("http://doi.org/", "")
    raw = raw.replace(" ", "_")
    return raw or "Untitled"


def normalize_space(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _new_state() -> dict[str, Any]:
    return {
        "version": 2,
        "kind": "stellav_medical_scienceopen_kb",
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "pages": {},
        "chunks": {},
        "chunk_order": [],
        "token_index": {},
        "stats": {"pages": 0, "chunks": 0, "tokens": 0, "deduped_chunks": 0},
    }


def load_kb(path: Path = KB_PATH) -> dict[str, Any]:
    if not path.exists():
        return _new_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _new_state()
    state = _new_state()
    if isinstance(data, dict):
        state.update(data)
    state.setdefault("pages", {})
    state.setdefault("chunks", {})
    state.setdefault("chunk_order", [])
    state.setdefault("token_index", {})
    state.setdefault("stats", {})
    for k, v in _new_state()["stats"].items():
        state["stats"].setdefault(k, v)
    return state


def save_kb(state: dict[str, Any], path: Path = KB_PATH) -> None:
    state["updated_at"] = now_ts()
    state["stats"] = {
        "pages": len(state.get("pages", {})),
        "chunks": len(state.get("chunks", {})),
        "tokens": len(state.get("token_index", {})),
        "deduped_chunks": int(state.get("stats", {}).get("deduped_chunks", 0)),
    }
    path.write_text(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8", "ignore")).hexdigest()[:24]


def iter_tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        tok = raw.strip().lower()
        if tok and tok != "\n":
            out.append(tok)
    return out


def word_tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "") if t.lower() not in STOP_WORDS and len(t) > 1]


def classify_categories(text: str, category_hint: Optional[str] = None) -> list[str]:
    tokens = set(word_tokens(text))
    lowered = (text or "").lower()
    scores: Counter[str] = Counter()
    for section, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in tokens or kw in lowered:
                scores[section] += 1
    if category_hint in ALL_SECTIONS:
        scores[category_hint] += 10
    chosen = [section for section, score in scores.most_common() if score > 0]
    if "language" not in chosen:
        chosen.insert(0, "language")
    return chosen[:6]


def dominant_topic(categories: Iterable[str]) -> str:
    for cat in categories:
        if cat in TOPIC_SECTIONS:
            return cat
    if "methods" in set(categories):
        return "methods"
    return "language"


def split_chunks(text: str, *, max_words: int = 140, min_words: int = 18) -> list[str]:
    text = normalize_space(strip_jats(text))
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        units.extend([s.strip() for s in SENTENCE_RE.split(para) if s.strip()])
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        count = len(unit.split())
        if current and current_words + count > max_words:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
            current = []
            current_words = 0
        current.append(unit)
        current_words += count
    if current:
        chunk = " ".join(current).strip()
        if len(chunk.split()) >= min_words or not chunks:
            chunks.append(chunk)
    return chunks


def compact_answer(text: str, *, max_words: int = 60) -> str:
    text = normalize_space(text).replace("\n", " ")
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    chosen = " ".join(sentences[:2]) if sentences else text
    words = chosen.split()
    if len(words) > max_words:
        chosen = " ".join(words[:max_words]).rstrip(",;:") + "."
    return chosen.strip()


def keyword_phrase(text: str, categories: Iterable[str]) -> str:
    tokens = word_tokens(text)
    counts = Counter(t for t in tokens if len(t) >= 3)
    for cat in categories:
        for kw in CATEGORY_KEYWORDS.get(cat, ()):
            if kw in counts or kw in (text or "").lower():
                return kw.replace("_", " ")
    if counts:
        return " ".join(t for t, _ in counts.most_common(2))
    return "preprint"


def _add_to_token_index(state: dict[str, Any], chunk_hash: str, tokens: Iterable[str]) -> None:
    index = state.setdefault("token_index", {})
    for tok in dict.fromkeys(tokens):
        if tok in STOP_WORDS or len(tok) < 2:
            continue
        bucket = index.setdefault(tok, [])
        if chunk_hash not in bucket:
            bucket.append(chunk_hash)


def teach_realtime_vocabulary(text: str, categories: Iterable[str], *, max_tokens_per_section: int = 96) -> dict[str, int]:
    try:
        from medical_domain_generator import learn_vocabulary_from_text
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for section in dict.fromkeys(categories):
        if section not in ALL_SECTIONS:
            continue
        result = learn_vocabulary_from_text(text, section_hint=section, max_tokens=max_tokens_per_section)
        for key, value in result.get("new_by_section", {}).items():
            counts[key] = counts.get(key, 0) + int(value)
    return counts


def index_text(
    *,
    title: str,
    slug: str,
    content_text: str,
    url: str = "",
    category_hint: Optional[str] = None,
    path: Path = KB_PATH,
    teach_vocab: bool = True,
    doi: str = "",
    source: str = "scienceopen",
) -> dict[str, Any]:
    state = load_kb(path)
    title = normalize_space(title) or normalize_slug(slug).replace("_", " ")
    slug = normalize_slug(slug or doi or title)
    text = normalize_space(content_text)
    page_hash = content_hash(text)
    page = state.setdefault("pages", {}).get(slug, {})
    page.setdefault("chunk_hashes", [])
    page.update({
        "title": title,
        "slug": slug,
        "doi": doi,
        "url": url or (scienceopen_url(doi) if doi else ""),
        "source": source,
        "source_hash": page_hash,
        "indexed_at": now_ts(),
        "word_count": len(text.split()),
    })

    added_chunks = 0
    deduped_chunks = 0
    new_vocab_counts: dict[str, int] = {}
    engine = get_engine()
    for i, chunk_text in enumerate(split_chunks(text), start=1):
        chunk_hash = content_hash(chunk_text)
        categories = classify_categories(chunk_text + "\n" + title, category_hint)
        topic = dominant_topic(categories)
        tokens = iter_tokens(chunk_text)
        if chunk_hash in state.setdefault("chunks", {}):
            deduped_chunks += 1
            if chunk_hash not in page["chunk_hashes"]:
                page["chunk_hashes"].append(chunk_hash)
            continue

        chunk = {
            "hash": chunk_hash,
            "chunk_id": f"{slug}#{i}",
            "slug": slug,
            "title": title,
            "doi": doi,
            "url": page["url"],
            "text": chunk_text,
            "answer": compact_answer(chunk_text),
            "categories": categories,
            "topic": topic,
            "keyword": keyword_phrase(chunk_text, categories),
            "tokens": sorted(set(t for t in tokens if len(t) > 1))[:256],
            "created_at": now_ts(),
            "source": source,
        }
        state["chunks"][chunk_hash] = chunk
        state.setdefault("chunk_order", []).append(chunk_hash)
        page["chunk_hashes"].append(chunk_hash)
        _add_to_token_index(state, chunk_hash, chunk["tokens"])
        engine.add(
            chunk_text,
            chunk_id=chunk_hash,
            title=title,
            doi=doi,
            url=page["url"],
            topic=topic,
            source=source,
            meta=chunk,
        )
        added_chunks += 1
        if teach_vocab:
            counts = teach_realtime_vocabulary(chunk_text, categories)
            for key, value in counts.items():
                new_vocab_counts[key] = new_vocab_counts.get(key, 0) + value

    state["pages"][slug] = page
    state.setdefault("stats", {})["deduped_chunks"] = int(state.get("stats", {}).get("deduped_chunks", 0)) + deduped_chunks
    save_kb(state, path)
    return {
        "slug": slug,
        "title": title,
        "doi": doi,
        "added_chunks": added_chunks,
        "deduped_chunks": deduped_chunks,
        "page_chunks": len(page.get("chunk_hashes", [])),
        "new_vocab": new_vocab_counts,
        "kb_path": str(path),
    }


def load_kb_into_tensor_engine(path: Path = KB_PATH) -> int:
    state = load_kb(path)
    engine = get_engine()
    n = 0
    for chunk in state.get("chunks", {}).values():
        engine.add(
            str(chunk.get("text") or ""),
            chunk_id=str(chunk.get("hash") or chunk.get("chunk_id") or n),
            title=str(chunk.get("title") or ""),
            doi=str(chunk.get("doi") or ""),
            url=str(chunk.get("url") or ""),
            topic=str(chunk.get("topic") or ""),
            source=str(chunk.get("source") or "scienceopen"),
            meta=chunk,
        )
        n += 1
    return n


def qa_pairs_from_kb(path: Path = KB_PATH, *, max_pairs: int = 1000) -> list[dict[str, str]]:
    state = load_kb(path)
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk_hash in state.get("chunk_order", []):
        chunk = state.get("chunks", {}).get(chunk_hash)
        if not chunk:
            continue
        title = chunk.get("title", "preprint")
        keyword = chunk.get("keyword", "research")
        answer = compact_answer(chunk.get("answer") or chunk.get("text", ""))
        doi = chunk.get("doi") or ""
        topic = chunk.get("topic", "language")
        questions = [
            f"what does {title} report about {keyword}?",
            f"summarize {keyword} from the ScienceOpen preprint {title}",
        ]
        if doi:
            questions.append(f"what is the ScienceOpen preprint {doi} about?")
        if topic not in {"language", "methods"}:
            questions.append(f"how is {keyword} relevant to {topic.replace('_', ' ')}?")
        for q in questions:
            key = (q.lower(), answer.lower())
            if key in seen or not answer:
                continue
            seen.add(key)
            pairs.append({"question": q, "answer": answer, "section": topic, "source": "scienceopen"})
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _existing_dataset_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            q = str(obj.get("question", "")).strip().lower()
            a = str(obj.get("answer", "")).strip().lower()
            if q and a:
                keys.add((q, a))
    except Exception:
        return keys
    return keys


def append_kb_pairs_to_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    kb_path: Path = KB_PATH,
    max_pairs: int = 1000,
) -> dict[str, int]:
    pairs = qa_pairs_from_kb(kb_path, max_pairs=max_pairs)
    existing = _existing_dataset_keys(dataset_path)
    added = 0
    if pairs:
        with open(dataset_path, "a", encoding="utf-8") as f:
            for p in pairs:
                q = str(p.get("question", "")).strip()
                a = str(p.get("answer", "")).strip()
                key = (q.lower(), a.lower())
                if not q or not a or key in existing:
                    continue
                existing.add(key)
                f.write(json.dumps({"question": q, "answer": a, "section": p.get("section", "language"), "source": "scienceopen"}, ensure_ascii=False) + "\n")
                added += 1
    return {"available_pairs": len(pairs), "added_pairs": added, "dataset_total": len(existing)}


def index_scienceopen_paper(paper: dict[str, Any], *, category_hint: Optional[str] = None, kb_path: Path = KB_PATH) -> dict[str, Any]:
    return index_text(
        title=str(paper.get("title") or "Untitled preprint"),
        slug=str(paper.get("doi") or paper.get("title") or "preprint"),
        content_text=paper_text(paper),
        url=str(paper.get("url") or ""),
        category_hint=category_hint,
        path=kb_path,
        teach_vocab=True,
        doi=str(paper.get("doi") or ""),
        source=str(paper.get("source") or "scienceopen"),
    )


def index_scienceopen_queries(
    queries: Iterable[str],
    *,
    rows_per_query: int = 6,
    category_hint: Optional[str] = None,
    kb_path: Path = KB_PATH,
    dataset_path: Optional[Path] = DEFAULT_DATASET_PATH,
    max_dataset_pairs: int = 1000,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw in queries:
        query = str(raw).strip()
        if not query:
            continue
        try:
            if query.lower().startswith("10.") or query.lower().startswith("https://doi.org/"):
                papers = [fetch_scienceopen_doi(query)]
            else:
                papers = search_scienceopen(query, rows=rows_per_query)
            for paper in papers:
                item = index_scienceopen_paper(paper, category_hint=category_hint, kb_path=kb_path)
                item["query"] = query
                results.append(item)
        except Exception as exc:
            errors.append(f"{query}: {exc}")
    dataset_stats = {"available_pairs": 0, "added_pairs": 0, "dataset_total": 0}
    if dataset_path is not None:
        dataset_stats = append_kb_pairs_to_dataset(dataset_path, kb_path=kb_path, max_pairs=max_dataset_pairs)
    load_kb_into_tensor_engine(kb_path)
    return {"indexed": results, "errors": errors, "dataset": dataset_stats, "kb_path": str(kb_path)}


def index_default_scienceopen_categories(
    *,
    categories: Optional[Iterable[str]] = None,
    rows_per_query: int = 4,
    kb_path: Path = KB_PATH,
    dataset_path: Optional[Path] = DEFAULT_DATASET_PATH,
    max_dataset_pairs: int = 1000,
) -> dict[str, Any]:
    manifest = default_scienceopen_queries(categories)
    indexed: list[dict[str, Any]] = []
    errors: list[str] = []
    for category, queries in manifest.items():
        for query in queries:
            try:
                papers = search_scienceopen(query, rows=rows_per_query)
                for paper in papers:
                    item = index_scienceopen_paper(paper, category_hint=category, kb_path=kb_path)
                    item["category"] = category
                    item["query"] = query
                    indexed.append(item)
            except Exception as exc:
                errors.append(f"{category}/{query}: {exc}")
    dataset_stats = {"available_pairs": 0, "added_pairs": 0, "dataset_total": 0}
    if dataset_path is not None:
        dataset_stats = append_kb_pairs_to_dataset(dataset_path, kb_path=kb_path, max_pairs=max_dataset_pairs)
    load_kb_into_tensor_engine(kb_path)
    return {
        "manifest": manifest,
        "indexed": indexed,
        "errors": errors,
        "dataset": dataset_stats,
        "kb_path": str(kb_path),
    }


def _score_chunk(question: str, chunk: dict[str, Any]) -> float:
    q_tokens = word_tokens(question)
    if not q_tokens:
        return 0.0
    q_set = set(q_tokens)
    chunk_tokens = set(chunk.get("tokens") or word_tokens(chunk.get("text", "")))
    title_tokens = set(word_tokens(chunk.get("title", "")))
    keyword_tokens = set(word_tokens(chunk.get("keyword", "")))
    if not chunk_tokens:
        return 0.0
    overlap = len(q_set & chunk_tokens)
    title_overlap = len(q_set & title_tokens)
    key_overlap = len(q_set & keyword_tokens)
    union = len(q_set | chunk_tokens)
    score = overlap / max(1, union)
    score += 0.15 * title_overlap
    score += 0.25 * key_overlap
    categories = set(chunk.get("categories", []))
    for cat, kws in CATEGORY_KEYWORDS.items():
        if cat in categories and (q_set & kws):
            score += 0.12
    return score


def answer_from_kb(question: str, *, path: Path = KB_PATH, max_words: int = 90) -> tuple[str, float, list[str]] | None:
    load_kb_into_tensor_engine(path)
    engine_hits = get_engine().search(question, k=3)
    if engine_hits and engine_hits[0].score > 0.05:
        hit = engine_hits[0]
        answer = compact_answer(hit.text, max_words=max_words)
        doi = hit.doi or "unspecified DOI"
        text = f"From ScienceOpen preprint {hit.title or 'untitled'} ({doi}): {answer} This is research information, not clinical advice."
        conf = min(0.94, 0.62 + max(0.0, hit.score))
        return text, conf, iter_tokens(text)

    state = load_kb(path)
    chunks = state.get("chunks", {})
    if not chunks:
        return None
    best: tuple[float, dict[str, Any]] | None = None
    for chunk in chunks.values():
        score = _score_chunk(question, chunk)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, chunk)
    if best is None or best[0] < 0.08:
        return None
    score, chunk = best
    answer = compact_answer(chunk.get("answer") or chunk.get("text", ""), max_words=max_words)
    title = str(chunk.get("title", "a ScienceOpen preprint")).strip()
    doi = str(chunk.get("doi") or "")
    doi_bit = f" ({doi})" if doi else ""
    text = f"From ScienceOpen preprint {title}{doi_bit}: {answer} This is research information, not clinical advice."
    conf = min(0.94, 0.62 + score)
    return text, conf, iter_tokens(text)


def kb_stats(path: Path = KB_PATH) -> dict[str, Any]:
    state = load_kb(path)
    by_topic: Counter[str] = Counter()
    for chunk in state.get("chunks", {}).values():
        by_topic[str(chunk.get("topic", "language"))] += 1
    return {
        "path": str(path),
        "pages": len(state.get("pages", {})),
        "chunks": len(state.get("chunks", {})),
        "tokens": len(state.get("token_index", {})),
        "deduped_chunks": int(state.get("stats", {}).get("deduped_chunks", 0)),
        "by_topic": dict(sorted(by_topic.items())),
        "source": "scienceopen",
        "tensor": get_engine().status(),
    }



