#!/usr/bin/env python3
"""Ollama-backed Stella V teacher protocol for medical-research vocabulary.

Compact p/q/a lines stay append-only. Game-engine sections are gone; sections are
language, methods, and medical-research topics used to ground ScienceOpen retrieval.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote, unquote

import agent_client
from bootstrap_data import BOOTSTRAP_VOCAB_SECTIONS, bootstrap_dataset_items, bootstrap_usage_items

TOPICS = [
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
]
BASE_VOCAB_SECTIONS = ["language", "methods"]
VOCAB_SECTIONS = BASE_VOCAB_SECTIONS + TOPICS

VOCAB_PATH = Path(os.environ.get("STELLAV_TEACHER_VOCAB", "agent_teacher_vocab.json"))
USAGE_BOOTSTRAP_PATH = Path(os.environ.get("STELLAV_USAGE_BOOTSTRAP", "agent_vocab_usage_bootstrap.jsonl"))
RETRY_INITIAL_WAIT = 3
RETRY_MAX_WAIT = 45
RETRY_BACKOFF = 1.7
TEACHER_CHUNK_SIZE = max(1, int(os.environ.get("STELLAV_TEACHER_CHUNK_SIZE", "8")))
USAGE_PAIRS_PER_SECTION = max(1, int(os.environ.get("STELLAV_USAGE_PAIRS_PER_SECTION", "3")))
KNOWLEDGE_PAIRS_PER_TOPIC = max(1, int(os.environ.get("STELLAV_KNOWLEDGE_PAIRS_PER_TOPIC", "3")))
MAX_TEACHER_ATTEMPTS = max(1, int(os.environ.get("STELLAV_TEACHER_MAX_ATTEMPTS", "2")))
MAX_EMPTY_ACCEPT_RETRIES = max(1, int(os.environ.get("STELLAV_EMPTY_ACCEPT_RETRIES", "2")))
PROMPT_SECTION_LIMIT = max(24, int(os.environ.get("STELLAV_PROMPT_SECTION_LIMIT", "48")))
PROMPT_RELATED_LIMIT = max(6, int(os.environ.get("STELLAV_PROMPT_RELATED_LIMIT", "8")))

SECTION_MIN_COUNTS = {
    "language": 40,
    "methods": 36,
    "clinical_trials": 24,
    "epidemiology": 24,
    "oncology": 24,
    "cardiology": 20,
    "immunology": 20,
    "infectious": 20,
    "pharmacology": 20,
    "genomics": 20,
    "neurology": 20,
    "public_health": 20,
}

_TOPIC_DIRECTIVES = {
    "clinical_trials": "randomization, blinding, endpoints, phases, allocation concealment, CONSORT, safety monitoring",
    "epidemiology": "incidence, prevalence, confounding, cohorts, case-control, causal inference, bias",
    "oncology": "tumors, staging, immunotherapy, biomarkers, chemotherapy, metastasis, precision oncology",
    "cardiology": "atherosclerosis, heart failure, myocardial infarction, blood pressure, lipids, stroke prevention",
    "immunology": "innate and adaptive immunity, vaccines, antibodies, T cells, cytokines, autoimmunity",
    "infectious": "pathogens, antimicrobial resistance, sepsis, source control, diagnostics, infection prevention",
    "pharmacology": "pharmacokinetics, pharmacodynamics, dose, exposure, therapeutic index, adverse events",
    "genomics": "GWAS, CRISPR, variants, precision medicine, polygenic scores, gene therapy",
    "neurology": "stroke, neurodegeneration, Alzheimer, Parkinson, biomarkers, clinical scales",
    "public_health": "screening, prevention, vaccination programs, health equity, implementation",
}

_SECTION_DIRECTIVES = {
    "language": "English research-communication tokens: question words, evidence verbs, hedge words, citation words; no game or engine vocabulary.",
    "methods": "Study-design and biostatistics tokens: randomization, blinding, power, bias, endpoint, hazard, confidence interval.",
}
_SECTION_DIRECTIVES.update(_TOPIC_DIRECTIVES)

_SECTION_HINTS = {
    "language": [
        "how", "what", "which", "when", "where", "why", "explain", "summarize", "cite", "evidence",
        "preprint", "scienceopen", "study", "trial", "patient", "disease", "risk", "outcome", "review",
        "question", "answer", "report", "suggests", "associated", "compared", "versus", "limit",
    ],
    "methods": [
        "randomization", "blinding", "endpoint", "confounding", "bias", "cohort", "control", "power",
        "sample", "interval", "hazard", "odds", "ratio", "protocol", "consort", "prisma", "p-value",
    ],
    "clinical_trials": [
        "trial", "rct", "randomized", "placebo", "phase", "arm", "allocation", "concealment", "adverse",
        "primary", "secondary", "interim", "futility", "consort",
    ],
    "epidemiology": [
        "incidence", "prevalence", "cohort", "case-control", "confounding", "selection", "bias",
        "risk", "population", "observational", "causal",
    ],
    "oncology": [
        "cancer", "tumor", "oncology", "checkpoint", "immunotherapy", "chemotherapy", "metastasis",
        "staging", "biomarker", "pd-1", "radiation",
    ],
    "cardiology": [
        "heart", "cardiac", "atherosclerosis", "infarction", "failure", "hypertension", "stroke",
        "coronary", "lipid", "ejection",
    ],
    "immunology": [
        "immune", "vaccine", "antibody", "antigen", "t-cell", "b-cell", "innate", "adaptive",
        "cytokine", "autoimmunity",
    ],
    "infectious": [
        "infection", "pathogen", "virus", "bacteria", "sepsis", "resistance", "antimicrobial",
        "antibiotic", "source", "stewardship",
    ],
    "pharmacology": [
        "drug", "dose", "exposure", "pharmacokinetics", "pharmacodynamics", "clearance", "metabolism",
        "adverse", "therapeutic", "index",
    ],
    "genomics": [
        "gene", "variant", "gwas", "crispr", "sequencing", "polygenic", "precision", "mutation",
        "transcript", "editing",
    ],
    "neurology": [
        "neuro", "alzheimer", "parkinson", "stroke", "dementia", "amyloid", "neuron", "csf",
        "seizure", "cognition",
    ],
    "public_health": [
        "screening", "prevention", "equity", "program", "uptake", "policy", "vaccination",
        "population", "implementation", "overdiagnosis",
    ],
}

_QA_LINE_RE = re.compile(r"^\s*([qa])\s*[:=|]\s*(.*?)\s*$", re.IGNORECASE)
_P_STANDARD_RE = re.compile(r"^\s*p\s*[:=|]\s*(\d+)\s*[:=|]\s*(.*?)\s*$", re.IGNORECASE)
_P_ANGLE_RE = re.compile(r"^\s*p\s*<\s*(\d+)\s*>\s*[:=|]\s*(.*?)\s*$", re.IGNORECASE)
_BOOTSTRAP_TOKEN_RE = re.compile(r"[a-z0-9_']+|[^\w\s]", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(game engine|rigid body|swapchain|gamepad|cgo|pygame|shader pipeline)\b",
    re.IGNORECASE,
)


class TeacherGenerationError(RuntimeError):
    pass


def _norm_topic(topic: Optional[str]) -> str:
    return topic if topic in TOPICS else random.choice(TOPICS)


def _norm_section(section: Optional[str]) -> str:
    return section if section in VOCAB_SECTIONS else "language"


def _normalize_token(token: str) -> str:
    token = unquote(str(token).strip())
    if token in {"\\n", "<nl>"}:
        return "\\n"
    return token.lower()


def _encode_token(token: str) -> str:
    return quote(token, safe="")


def _empty_state() -> dict[str, Any]:
    return {
        "version": 3,
        "append_only": True,
        "id_to_token": {},
        "token_to_id": {},
        "sections": {section: [] for section in VOCAB_SECTIONS},
    }


def _state_from_legacy(data: Any) -> dict[str, Any]:
    state = _empty_state()
    if not isinstance(data, dict):
        return state
    raw_items = data.get("id_to_token", data)
    if isinstance(raw_items, dict):
        seen: set[str] = set()
        for raw_id, raw_tok in sorted(raw_items.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999999):
            try:
                idx = int(raw_id)
            except Exception:
                continue
            if idx < 6:
                continue
            tok = _normalize_token(str(raw_tok))
            if not tok or tok in seen:
                continue
            state["id_to_token"][str(idx)] = tok
            state["token_to_id"][tok] = idx
            seen.add(tok)
    raw_sections = data.get("sections", {}) if isinstance(data, dict) else {}
    if isinstance(raw_sections, dict):
        for section, ids in raw_sections.items():
            if section not in state["sections"] or not isinstance(ids, list):
                continue
            for idx in ids:
                try:
                    iid = int(idx)
                except Exception:
                    continue
                if str(iid) in state["id_to_token"] and iid not in state["sections"][section]:
                    state["sections"][section].append(iid)
    if not any(state["sections"].values()):
        for sid, tok in state["id_to_token"].items():
            state["sections"][_classify_token_section(tok)].append(int(sid))
    return state


def _load_vocab_state(path: Path = VOCAB_PATH) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        return _state_from_legacy(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return _empty_state()


def _save_vocab_state(state: dict[str, Any], path: Path = VOCAB_PATH) -> None:
    clean = _empty_state()
    seen_tokens: set[str] = set()
    for raw_id, raw_tok in sorted(state.get("id_to_token", {}).items(), key=lambda kv: int(kv[0])):
        idx = int(raw_id)
        if idx < 6:
            continue
        tok = _normalize_token(str(raw_tok))
        if not tok or tok in seen_tokens:
            continue
        seen_tokens.add(tok)
        clean["id_to_token"][str(idx)] = tok
        clean["token_to_id"][tok] = idx
    sections = state.get("sections", {})
    for section in VOCAB_SECTIONS:
        added: set[int] = set()
        for raw_id in sections.get(section, []) if isinstance(sections, dict) else []:
            try:
                idx = int(raw_id)
            except Exception:
                continue
            if str(idx) in clean["id_to_token"] and idx not in added:
                clean["sections"][section].append(idx)
                added.add(idx)
    path.write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")


def _next_vocab_id(state: dict[str, Any]) -> int:
    ids = [int(k) for k in state.get("id_to_token", {}).keys() if str(k).isdigit()]
    return max([5, *ids]) + 1


def _section_token_count(state: dict[str, Any], section: str) -> int:
    section = _norm_section(section)
    ids = state.get("sections", {}).get(section, [])
    return len([i for i in ids if str(i) in state.get("id_to_token", {})])


def _total_vocab_count(state: dict[str, Any]) -> int:
    return len(state.get("id_to_token", {}))


def _iter_bootstrap_tokens(text: str) -> Iterator[str]:
    for raw in _BOOTSTRAP_TOKEN_RE.findall(text.lower()):
        token = _normalize_token(raw)
        if token:
            yield token


def _classify_token_section(tok: str) -> str:
    tok = _normalize_token(tok)
    for section, hints in _SECTION_HINTS.items():
        if tok in {_normalize_token(h) for h in hints}:
            return section
    return "language"


def _attach_token(state: dict[str, Any], token: str, section: str, proposed_id: Optional[int] = None) -> tuple[int, bool]:
    section = _norm_section(section)
    token = _normalize_token(token)
    if not token:
        return -1, False
    token_to_id = state.setdefault("token_to_id", {})
    id_to_token = state.setdefault("id_to_token", {})
    sections = state.setdefault("sections", {s: [] for s in VOCAB_SECTIONS})
    for s in VOCAB_SECTIONS:
        sections.setdefault(s, [])
    if token in token_to_id:
        actual_id = int(token_to_id[token])
        if actual_id not in sections[section]:
            sections[section].append(actual_id)
        return actual_id, False
    actual_id = _next_vocab_id(state)
    id_to_token[str(actual_id)] = token
    token_to_id[token] = actual_id
    sections[section].append(actual_id)
    return actual_id, True


def teacher_vocab_tokens(path: Path = VOCAB_PATH) -> list[str]:
    state = _load_vocab_state(path)
    return [state["id_to_token"][str(i)] for i in sorted(int(k) for k in state["id_to_token"].keys())]


def learn_vocabulary_from_text(
    text: str,
    *,
    section_hint: Optional[str] = None,
    max_tokens: int = 128,
    path: Path = VOCAB_PATH,
) -> dict[str, Any]:
    state = _load_vocab_state(path)
    section_hint = _norm_section(section_hint)
    new_by_section: dict[str, int] = {s: 0 for s in VOCAB_SECTIONS}
    attached_by_section: dict[str, int] = {s: 0 for s in VOCAB_SECTIONS}
    seen: set[str] = set()
    for raw in _BOOTSTRAP_TOKEN_RE.findall(text or ""):
        token = _normalize_token(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        if len(seen) > max_tokens:
            break
        classified = _classify_token_section(token)
        section = classified if classified != "language" else (section_hint if section_hint in VOCAB_SECTIONS else "language")
        before_ids = set(state.setdefault("sections", {}).setdefault(section, []))
        _actual_id, is_new = _attach_token(state, token, section)
        after_ids = set(state.setdefault("sections", {}).setdefault(section, []))
        if is_new:
            new_by_section[section] += 1
        if after_ids != before_ids:
            attached_by_section[section] += 1
    _save_vocab_state(state, path)
    return {
        "new_by_section": {k: v for k, v in new_by_section.items() if v},
        "attached_by_section": {k: v for k, v in attached_by_section.items() if v},
        "total_vocab": _total_vocab_count(state),
    }


def seed_builtin_teacher_vocabulary(path: Path = VOCAB_PATH) -> dict[str, Any]:
    state = _load_vocab_state(path)
    for section, tokens in BOOTSTRAP_VOCAB_SECTIONS.items():
        for token in tokens:
            _attach_token(state, token, section)
    for item in bootstrap_dataset_items():
        answer_section = _norm_section(item.get("section"))
        for token in _iter_bootstrap_tokens(item["question"]):
            _attach_token(state, token, "language")
        for token in _iter_bootstrap_tokens(item["answer"]):
            chosen = answer_section if answer_section != "language" else _classify_token_section(token)
            _attach_token(state, token, chosen)
    _save_vocab_state(state, path)
    return state


def _section_ids_for_prompt(state: dict[str, Any], section: str, *, max_ids: Optional[int] = None) -> list[int]:
    section = _norm_section(section)
    id_to_token = state.get("id_to_token", {})
    ids = [int(i) for i in state.get("sections", {}).get(section, []) if str(int(i)) in id_to_token]
    ids = sorted(dict.fromkeys(ids))
    if not max_ids or len(ids) <= max_ids:
        return ids
    hint_tokens = {_normalize_token(t) for t in _SECTION_HINTS.get(section, [])}
    preferred = [idx for idx in ids if _normalize_token(id_to_token.get(str(idx), "")) in hint_tokens][:max_ids]
    if len(preferred) >= max_ids:
        return preferred
    tail = [idx for idx in ids if idx not in set(preferred)]
    return preferred + tail[-(max_ids - len(preferred)):]


def _vocab_block(state: dict[str, Any], sections: Optional[list[str]] = None, *, include_headers: bool = True, max_ids_per_section: Optional[int] = None) -> str:
    id_to_token = state.get("id_to_token", {})
    lines: list[str] = []
    for section in sections or VOCAB_SECTIONS:
        if section not in VOCAB_SECTIONS:
            continue
        ids = _section_ids_for_prompt(state, section, max_ids=max_ids_per_section)
        if not ids:
            continue
        if include_headers:
            lines.append(f"# section={section}")
        for idx in ids:
            lines.append(f"p|{idx}|{_encode_token(id_to_token[str(idx)])}")
    return "\n".join(lines)


def _strip_fences(raw: str) -> str:
    raw = raw.strip().replace("\ufeff", "")
    raw = re.sub(r"^```(?:text|csv|json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_p_line(line: str) -> Optional[tuple[int, str]]:
    m = _P_STANDARD_RE.match(line) or _P_ANGLE_RE.match(line)
    if not m:
        return None
    try:
        idx = int(m.group(1))
    except Exception:
        return None
    tok = _normalize_token(m.group(2))
    if idx < 6 or not tok:
        return None
    return idx, tok


def _parse_id_csv(text: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return []
    return out


def _detokenize(tokens: list[str]) -> str:
    no_space_before = {".", ",", ";", ":", ")", "]", "}", ">", "?"}
    no_space_after = {"(", "[", "{", "<", "#", ":", ".", "'", '"'}
    text = ""
    prev = ""
    for tok in tokens:
        if tok == "\\n":
            text += "\n"
        elif not text or tok in no_space_before or prev in no_space_after:
            text += tok
        else:
            text += " " + tok
        prev = tok
    return re.sub(r"\s+", " ", text).strip()


def _decode_ids(ids: list[int], state: dict[str, Any], aliases: Optional[dict[int, int]] = None) -> str:
    aliases = aliases or {}
    id_to_token = state.get("id_to_token", {})
    tokens: list[str] = []
    for idx in ids:
        actual = aliases.get(idx, idx)
        tok = id_to_token.get(str(actual))
        if tok is None:
            return ""
        tokens.append(tok)
    return _detokenize(tokens)


def _looks_like_domain_question(q: str) -> bool:
    ql = q.lower()
    anchors = VOCAB_SECTIONS + [
        "preprint", "scienceopen", "trial", "patient", "disease", "cancer", "heart",
        "vaccine", "drug", "gene", "stroke", "risk", "study", "how", "what", "why",
        "endpoint", "confounding", "immunity", "infection", "screening",
    ]
    return any(t.replace("_", " ") in ql or t in ql for t in anchors)


def _is_valid_pair(q: str, a: str, phase: Optional[str] = None) -> bool:
    q = q.strip()
    a = a.strip()
    if len(q) < 3 or len(a) < 1:
        return False
    if _FORBIDDEN_RE.search(q) or _FORBIDDEN_RE.search(a):
        return False
    if not _looks_like_domain_question(q):
        return False
    return len(a.split()) <= 40 and a.count("\n") <= 3


def parse_pqa_response(
    raw: str,
    state: Optional[dict[str, Any]] = None,
    *,
    section: str = "language",
    phase: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, str]], int]:
    if state is None:
        state = _load_vocab_state()
    else:
        state = _state_from_legacy(state)
    raw = _strip_fences(raw)
    aliases: dict[int, int] = {}
    pending_q: list[int] | None = None
    pairs: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    new_tokens = 0
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        p = _parse_p_line(line)
        if p is not None:
            proposed_id, token = p
            actual_id, is_new = _attach_token(state, token, section, proposed_id)
            if actual_id >= 6:
                aliases[proposed_id] = actual_id
            if is_new:
                new_tokens += 1
            continue
        m = _QA_LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        payload = m.group(2).strip()
        if key == "q":
            pending_q = _parse_id_csv(payload)
            continue
        if key == "a" and pending_q is not None:
            a_ids = _parse_id_csv(payload)
            q_text = _decode_ids(pending_q, state, aliases)
            a_text = _decode_ids(a_ids, state, aliases)
            pending_q = None
            if not q_text or not a_text or not _is_valid_pair(q_text, a_text, phase=phase):
                continue
            dedupe = (q_text.lower(), a_text.lower())
            if dedupe in seen_pairs:
                continue
            seen_pairs.add(dedupe)
            pairs.append({"question": q_text, "answer": a_text})
    if new_tokens:
        _save_vocab_state(state)
    return state, pairs, new_tokens


def _wait_for_ollama(wait: int) -> bool:
    if agent_client.is_alive():
        return True
    print(f"[gen] Ollama not reachable at {agent_client.OLLAMA_BASE} — retrying in {wait}s…", flush=True)
    time.sleep(wait)
    return False


def _generate_teacher(prompt: str, label: str, *, needed_pairs: int = 0, vocab_request: bool = False) -> str:
    wait = RETRY_INITIAL_WAIT
    attempt = 0
    while True:
        attempt += 1
        if not _wait_for_ollama(wait):
            wait = min(int(wait * RETRY_BACKOFF), RETRY_MAX_WAIT)
            continue
        try:
            max_tokens = 2048 if vocab_request else max(768, min(3072, 90 * max(1, needed_pairs) + 512))
            print(f"[gen] {label} attempt {attempt}: requesting compact p/q/a from {agent_client.MODEL_NAME}…", flush=True)
            return agent_client.generate(prompt, temperature=0.05 if vocab_request else 0.12, max_tokens=max_tokens)
        except Exception as exc:
            print(f"[gen] {label} attempt {attempt}: error: {exc}", flush=True)
            if attempt >= MAX_TEACHER_ATTEMPTS:
                raise TeacherGenerationError(f"{label} failed after {attempt} attempt(s): {exc}") from exc
            time.sleep(wait)
            wait = min(int(wait * RETRY_BACKOFF), RETRY_MAX_WAIT)


def _build_section_vocab_prompt(section: str, iteration: int, state: dict[str, Any], *, target_count: Optional[int] = None) -> str:
    section = _norm_section(section)
    target = max(SECTION_MIN_COUNTS.get(section, 20), int(target_count or SECTION_MIN_COUNTS.get(section, 20)))
    have = _section_token_count(state, section)
    need = max(1, target - have)
    next_id = _next_vocab_id(state)
    existing_section = _vocab_block(state, [section], include_headers=True, max_ids_per_section=PROMPT_SECTION_LIMIT)
    hints = ", ".join(_encode_token(t) for t in _SECTION_HINTS.get(section, [])[:160])
    return f"""\
Teach Stella V ONLY the {section} medical-research vocabulary section.
Section meaning: {_SECTION_DIRECTIVES.get(section, section)}
Iteration: {iteration}. Current count: {have}. Reach at least {target}. Add at least {need} unique token(s).
Known tokens:
{existing_section or '(empty)'}
Suggested anchors: {hints}
Return ONLY p lines. Format: p|<integer_id>|<percent-encoded-token>
New ids start at or after {next_id}. No game, graphics, or engine tokens.
"""


def ensure_section_vocabulary(section: str, iteration: int = 1, *, target_count: Optional[int] = None) -> dict[str, Any]:
    section = _norm_section(section)
    state = _load_vocab_state()
    target = max(SECTION_MIN_COUNTS.get(section, 20), int(target_count or SECTION_MIN_COUNTS.get(section, 20)))
    empty_retries = 0
    while _section_token_count(state, section) < target:
        before = _section_token_count(state, section)
        prompt = _build_section_vocab_prompt(section, iteration, state, target_count=target)
        try:
            raw = _generate_teacher(prompt, f"vocab/{section}", vocab_request=True)
        except TeacherGenerationError as exc:
            print(f"[gen] vocab/{section}: {exc}; skipping further growth.", flush=True)
            break
        state, _pairs, new_count = parse_pqa_response(raw, state, section=section, phase=f"vocab_{section}")
        _save_vocab_state(state)
        after = _section_token_count(state, section)
        if new_count > 0 or after > before:
            empty_retries = 0
            continue
        empty_retries += 1
        if empty_retries >= MAX_EMPTY_ACCEPT_RETRIES:
            break
        time.sleep(RETRY_INITIAL_WAIT)
    return state


def expand_teacher_vocabulary(iteration: int = 1, *, sections: Optional[list[str]] = None, extra_tokens_per_section: int = 6) -> dict[str, Any]:
    state = seed_builtin_teacher_vocabulary()
    chosen = [s for s in dict.fromkeys(_norm_section(s) for s in (sections or VOCAB_SECTIONS)) if s in VOCAB_SECTIONS]
    if extra_tokens_per_section <= 0:
        return state
    for section in chosen:
        current = _section_token_count(state, section)
        state = ensure_section_vocabulary(section, iteration, target_count=current + extra_tokens_per_section)
    return state


def ensure_all_category_vocabularies(iteration: int = 1) -> dict[str, Any]:
    state = seed_builtin_teacher_vocabulary()
    for section in VOCAB_SECTIONS:
        state = ensure_section_vocabulary(section, iteration)
    print(f"[gen] Vocabulary ready: total={_total_vocab_count(state)} token id(s).", flush=True)
    return state


def ensure_teacher_vocabulary(topic: Optional[str] = None, iteration: int = 1) -> dict[str, Any]:
    topic = _norm_topic(topic)
    state = seed_builtin_teacher_vocabulary()
    for section in BASE_VOCAB_SECTIONS + [topic]:
        state = ensure_section_vocabulary(section, iteration)
    return state


def seed_builtin_usage_bootstrap(path: Path = USAGE_BOOTSTRAP_PATH) -> list[dict[str, str]]:
    saved: list[dict[str, str]] = []
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    saved.append(json.loads(line))
        except Exception:
            saved = []
    seen = {
        (str(item.get("phase", "")).lower(), str(item.get("question", "")).strip().lower(), str(item.get("answer", "")).strip().lower())
        for item in saved
    }
    merged = list(saved)
    changed = False
    for item in bootstrap_usage_items():
        key = (str(item.get("phase", "")).lower(), str(item.get("question", "")).strip().lower(), str(item.get("answer", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append({"phase": str(item["phase"]), "question": str(item["question"]), "answer": str(item["answer"])})
        changed = True
    if changed or not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return merged


def _usage_phase_prompt(section: str, n: int, iteration: int) -> str:
    state = _load_vocab_state()
    section = _norm_section(section)
    prompt_sections = ["language", "methods", section] if section not in BASE_VOCAB_SECTIONS else ["language", section]
    vocab_text = _vocab_block(state, list(dict.fromkeys(prompt_sections)), include_headers=True, max_ids_per_section=PROMPT_SECTION_LIMIT)
    goal = f"Teach {section} medical-research usage. Answers should be short English research sentences, not code and not clinical advice."
    return f"""\
Generate exactly {n} vocabulary-usage training pair(s) for section={section}.
{goal}
Iteration: {iteration}.
Use only these vocabulary sections:
{vocab_text}
Return ONLY:
q|<csv token ids>
a|<csv token ids>
Produce exactly {n} q lines and exactly {n} a lines. No markdown.
"""


def ensure_vocabulary_usage_bootstrap(topic: Optional[str] = None, iteration: int = 1, *, pairs_per_section: int = USAGE_PAIRS_PER_SECTION) -> list[dict[str, str]]:
    seed_builtin_teacher_vocabulary()
    ensure_all_category_vocabularies(iteration)
    saved = seed_builtin_usage_bootstrap()
    by_phase: dict[str, list[dict[str, str]]] = {f"usage_{s}": [] for s in VOCAB_SECTIONS}
    for item in saved:
        phase = str(item.get("phase", ""))
        if phase in by_phase:
            by_phase[phase].append(item)
    changed = False
    for section in VOCAB_SECTIONS:
        phase = f"usage_{section}"
        empty_retries = 0
        while len(by_phase[phase]) < pairs_per_section:
            request_n = min(pairs_per_section - len(by_phase[phase]), TEACHER_CHUNK_SIZE)
            try:
                raw = _generate_teacher(_usage_phase_prompt(section, request_n, iteration), phase, needed_pairs=request_n)
            except TeacherGenerationError as exc:
                print(f"[gen] {phase}: {exc}; keeping current bootstrap items.", flush=True)
                break
            state, pairs, _new = parse_pqa_response(raw, _load_vocab_state(), section=section, phase=phase)
            _save_vocab_state(state)
            if not pairs:
                empty_retries += 1
                if empty_retries >= MAX_EMPTY_ACCEPT_RETRIES:
                    break
                time.sleep(RETRY_INITIAL_WAIT)
                continue
            empty_retries = 0
            for pair in pairs:
                item = {"phase": phase, "question": pair["question"], "answer": pair["answer"]}
                key = (item["question"].lower(), item["answer"].lower())
                if all((p["question"].lower(), p["answer"].lower()) != key for p in by_phase[phase]):
                    by_phase[phase].append(item)
                    changed = True
    merged: list[dict[str, str]] = []
    for phase in by_phase:
        merged.extend(by_phase[phase][:pairs_per_section])
    if changed or not USAGE_BOOTSTRAP_PATH.exists():
        with open(USAGE_BOOTSTRAP_PATH, "w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return merged


def _build_knowledge_prompt(n: int, topic: str, iteration: int, *, failed_questions: Optional[list[str]] = None, accepted_questions: Optional[list[str]] = None) -> str:
    topic = _norm_topic(topic)
    ensure_teacher_vocabulary(topic, iteration)
    state = _load_vocab_state()
    vocab_text = _vocab_block(state, ["language", "methods", topic], include_headers=True, max_ids_per_section=PROMPT_SECTION_LIMIT)
    failures = "\nWeak questions to clarify:\n" + "\n".join(f"- {q}" for q in (failed_questions or [])[:8]) + "\n" if failed_questions else ""
    avoid = "\nDo not repeat these questions:\n" + "\n".join(f"- {q}" for q in (accepted_questions or [])[-12:]) + "\n" if accepted_questions else ""
    return f"""\
Generate exactly {n} Stella V knowledge QA pair(s) for topic={topic}.
Topic meaning: {_TOPIC_DIRECTIVES[topic]}.
Iteration: {iteration}.
{failures}{avoid}
Use these vocabulary sections:
{vocab_text}
Return ONLY compact q/a lines:
q|<csv token ids>
a|<csv token ids>
Answers must be short natural English about medical research, not clinical advice, not game engines, not code.
Produce exactly {n} q lines and exactly {n} a lines.
"""


def iter_knowledge_batches(n: int = KNOWLEDGE_PAIRS_PER_TOPIC, topic: Optional[str] = None, iteration: int = 1, failed_questions: Optional[list[str]] = None) -> Iterator[list[dict[str, str]]]:
    topic = _norm_topic(topic)
    ensure_teacher_vocabulary(topic, iteration)
    collected_count = 0
    seen_questions: set[str] = set()
    accepted_questions: list[str] = []
    round_num = 0
    empty_retries = 0
    while collected_count < n:
        round_num += 1
        request_n = min(n - collected_count, TEACHER_CHUNK_SIZE)
        prompt = _build_knowledge_prompt(request_n, topic, iteration, failed_questions=failed_questions if round_num == 1 else None, accepted_questions=accepted_questions)
        try:
            raw = _generate_teacher(prompt, f"knowledge/{topic}/chunk{round_num}", needed_pairs=request_n)
        except TeacherGenerationError as exc:
            print(f"[gen] knowledge/{topic}/chunk{round_num}: {exc}; stopping.", flush=True)
            break
        state, batch, _new = parse_pqa_response(raw, _load_vocab_state(), section=topic, phase=f"knowledge_{topic}")
        _save_vocab_state(state)
        fresh = []
        for pair in batch:
            qkey = pair["question"].strip().lower()
            if qkey in seen_questions:
                continue
            seen_questions.add(qkey)
            accepted_questions.append(pair["question"])
            fresh.append(pair)
        if not fresh:
            empty_retries += 1
            if empty_retries >= MAX_EMPTY_ACCEPT_RETRIES:
                break
            time.sleep(RETRY_INITIAL_WAIT)
            continue
        empty_retries = 0
        collected_count += len(fresh)
        yield fresh


def generate_knowledge_batch(n: int = KNOWLEDGE_PAIRS_PER_TOPIC, topic: Optional[str] = None, iteration: int = 1, failed_questions: Optional[list[str]] = None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for chunk in iter_knowledge_batches(n=n, topic=topic, iteration=iteration, failed_questions=failed_questions):
        out.extend(chunk)
        if len(out) >= n:
            break
    return out[:n]


def iter_medical_batches(n: int = 20, topic: Optional[str] = None, iteration: int = 1, failed_questions: Optional[list[str]] = None) -> Iterator[list[dict[str, str]]]:
    yield from iter_knowledge_batches(n=n, topic=topic, iteration=iteration, failed_questions=failed_questions)


def generate_medical_batch(n: int = 20, topic: Optional[str] = None, iteration: int = 1, failed_questions: Optional[list[str]] = None) -> list[dict[str, str]]:
    return generate_knowledge_batch(n=n, topic=topic, iteration=iteration, failed_questions=failed_questions)


# Compatibility aliases while callers are updated.
iter_game_batches = iter_medical_batches
generate_game_batch = generate_medical_batch
