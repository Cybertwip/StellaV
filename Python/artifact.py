#!/usr/bin/env python3
"""Detect script/file requests and turn retrieved ScienceOpen evidence into artifacts.

Stella V does not write files. When OpenCode (or another client) sends a write
tool, we draft the file from preprint passages and return a tool call.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Sequence

from tensor_engine import RetrievedChunk

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
FILENAME_RE = re.compile(r"(?i)\b(?:[\w.-]+/)*[\w.-]+\.(py|r|js|ts|go|rs|ipynb|sh|rb|jl|txt)\b")
FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)

_ARTIFACT_VERBS = (
    "write", "create", "generate", "make", "draft", "scaffold", "emit", "produce",
)
_ARTIFACT_NOUNS = (
    "script", "python", "chempy", "notebook", "module", "program", "source file", ".py", "code",
)
_STOP = {
    "write", "writing", "create", "creating", "generate", "generating", "make", "making",
    "draft", "scaffold", "python", "chempy", "script", "scripts", "file", "files", "code",
    "program", "now", "please", "based", "research", "the", "a", "an", "to", "for", "begin",
    "beginning", "start", "starting", "do", "that", "this", "with", "using", "use", "want",
    "wants", "user", "preprint", "passage", "passages", "therefore", "answer", "from",
    "found", "in", "of", "and", "or", "on", "it", "is", "be", "can", "should", "able",
    "just", "so", "those", "these", "i", "will", "get", "numpy", "pandas", "scipy",
    "jupyter", "notebook",
}
_MEDICAL_HINTS = (
    "cancer", "vaccine", "mrna", "rna", "immun", "oncolog", "tumor", "tumour",
    "antigen", "peptide", "trial", "therap", "patient", "clinic", "epidem",
)


def looks_like_artifact_request(text: str) -> bool:
    q = (text or "").lower()
    if not q.strip():
        return False
    has_verb = any(v in q for v in _ARTIFACT_VERBS)
    has_noun = any(n in q for n in _ARTIFACT_NOUNS)
    if has_verb and has_noun:
        return True
    if "write the script" in q or "script now" in q:
        return True
    return bool(FILENAME_RE.search(text or "") and has_verb)


def research_query_from(user_text: str, transcript: str = "") -> str:
    cleaned = _strip_boilerplate(user_text)
    if len(cleaned.split()) < 2:
        cleaned = _strip_boilerplate(transcript)
    if len(cleaned.split()) < 2:
        cleaned = _keep_medical(user_text + " " + transcript)
    doi_match = DOI_RE.search((transcript or "") + " " + (user_text or ""))
    if doi_match:
        cleaned = (cleaned + " " + doi_match.group(0).rstrip(".,;")).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned or (user_text or "").strip()


def _strip_boilerplate(text: str) -> str:
    kept: list[str] = []
    for tok in (text or "").lower().split():
        tok = tok.strip(".,;:!?\"'`")
        if not tok or tok in _STOP:
            continue
        if FILENAME_RE.match(tok):
            continue
        kept.append(tok)
    return " ".join(kept)


def _keep_medical(text: str) -> str:
    kept: list[str] = []
    seen: set[str] = set()
    for tok in (text or "").lower().split():
        tok = tok.strip(".,;:!?\"'`")
        if tok in seen:
            continue
        if any(h in tok for h in _MEDICAL_HINTS):
            kept.append(tok)
            seen.add(tok)
    return " ".join(kept)


def infer_artifact_path(user_text: str) -> str:
    named = FILENAME_RE.search(user_text or "")
    if named:
        return named.group(0)
    lower = (user_text or "").lower()
    stem = _slug_stem(research_query_from(user_text, "")) or "research_prototype"
    if ("vaccine" in lower or "therap" in lower) and "research" not in stem:
        stem += "_research"
    if "notebook" in lower or "ipynb" in lower:
        return stem + ".ipynb"
    if "chempy" in lower or "python" in lower or "script" in lower:
        return stem + ".py"
    return stem + ".py"


def _slug_stem(text: str) -> str:
    parts: list[str] = []
    for field in (text or "").lower().split():
        tok = "".join(ch for ch in field if ch.isalnum())
        if len(tok) < 2:
            continue
        parts.append(tok)
        if len(parts) >= 4:
            break
    stem = "_".join(parts)
    return stem[:48]


def extract_fenced_code(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    if looks_like_source(text):
        return text
    return ""


def looks_like_source(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    if q.startswith("#!/") or q.startswith('"""') or q.startswith("'''"):
        return True
    hits = 0
    for i, line in enumerate(q.splitlines()):
        if i > 40:
            break
        trim = line.strip()
        if trim.startswith("import ") or trim.startswith("from "):
            hits += 2
        elif trim.startswith("def ") or trim.startswith("class "):
            hits += 2
        elif trim.startswith("function ") or trim.startswith("package "):
            hits += 2
        elif trim.startswith("#") and i < 8:
            hits += 1
    return hits >= 2


def find_write_tool(tools: Sequence[dict[str, Any]] | None) -> dict[str, Any] | None:
    prefer = ("write", "write_file", "writefile", "create_file", "createfile")
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip().lower()
        if name:
            by_name[name] = tool
    for name in prefer:
        if name in by_name:
            return by_name[name]
    for name, tool in by_name.items():
        if "todo" in name:
            continue
        if "write" in name or "create_file" in name:
            return tool
    return None


def write_tool_arg_keys(tool: dict[str, Any]) -> tuple[str, str]:
    path_key, content_key = "file_path", "content"
    fn = tool.get("function") if isinstance(tool, dict) else None
    params = (fn or {}).get("parameters") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}
    props = params.get("properties") if isinstance(params, dict) else None
    required = params.get("required") if isinstance(params, dict) else None
    if not isinstance(props, dict):
        return path_key, content_key
    path_key = _first_schema_key(props, required, (
        "file_path", "filePath", "path", "filename", "file", "target_file", "targetFile",
    )) or path_key
    content_key = _first_schema_key(props, required, (
        "content", "contents", "text", "body", "file_text", "fileText",
    )) or content_key
    return path_key, content_key


def _first_schema_key(props: dict[str, Any], required: Any, aliases: tuple[str, ...]) -> str:
    lower = {str(k).lower(): str(k) for k in props}
    if isinstance(required, list):
        for req in required:
            orig = lower.get(str(req).lower())
            if orig and any(str(req).lower() == alias.lower() for alias in aliases):
                return orig
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return ""


def synthesize_write_call(tool: dict[str, Any], path: str, content: str) -> dict[str, Any]:
    fn = tool.get("function") if isinstance(tool, dict) else {}
    name = str((fn or {}).get("name") or "write")
    path_key, content_key = write_tool_arg_keys(tool)
    return {
        "id": "call_" + uuid.uuid4().hex[:12],
        "type": "function",
        "index": 0,
        "function": {
            "name": name,
            "arguments": json.dumps({path_key: path, content_key: content}),
        },
    }


def select_artifact_evidence(
    user_text: str,
    hits: Sequence[RetrievedChunk],
    transcript: str = "",
) -> RetrievedChunk | None:
    blob = f"{user_text} {transcript}"
    dois = [m.rstrip(".,;") for m in DOI_RE.findall(blob)]
    for doi in dois:
        for hit in hits:
            if hit.doi and (doi in hit.doi or hit.doi in doi):
                return hit
    query = research_query_from(user_text, transcript).lower()
    terms = query.split()
    best: RetrievedChunk | None = None
    best_score = -1.0
    for hit in hits:
        text_blob = f"{hit.title} {hit.text} {hit.doi}".lower()
        score = float(hit.score)
        for term in terms:
            if len(term) >= 4 and term in text_blob:
                score += 0.5
        if (hit.source or "").lower() == "scienceopen" and hit.doi:
            score += 0.1
        if score > best_score:
            best_score = score
            best = hit
    if best is not None:
        return best
    if dois:
        return RetrievedChunk(
            text=" ".join(blob.split())[:700],
            score=0.0,
            title="ScienceOpen preprint",
            doi=dois[0],
            url=f"https://www.scienceopen.com/hosted-document?doi={dois[0]}",
            source="scienceopen",
        )
    return None


def artifact_assistant_note(path: str, hits: Sequence[RetrievedChunk]) -> str:
    hit = hits[0] if hits else None
    if hit and (hit.title or hit.doi):
        doi = hit.doi or "DOI n/a"
        title = " ".join((hit.title or "").split())[:140] or "ScienceOpen preprint"
        return (
            f"Retrieved ScienceOpen preprint {title} ({doi}). "
            f"Drafting {path} from that evidence for OpenCode to write. "
            "Computational research only, not a therapy."
        )
    return (
        f"Drafting {path} from retrieved research evidence for OpenCode to write. "
        "Computational research only, not a therapy."
    )


def build_code_prompt(question: str, hits: Sequence[RetrievedChunk], *, local_draft: str = "") -> str:
    from reasoner import format_passages

    draft = ""
    if (local_draft or "").strip():
        draft = f"\nLocal tensor-memory draft (optional):\n{local_draft.strip()[:280]}\n"
    return (
        "You are Stella V. The user asked for a file or script. Stella retrieves "
        "ScienceOpen preprints and pushes that evidence; you must WRITE THE REQUESTED "
        "PROGRAM, not summarize the papers.\n"
        "Rules:\n"
        "1. Output only the complete file in one markdown code fence.\n"
        "2. Ground comments, constants, and model structure in the passages. Cite the DOI in a module docstring.\n"
        "3. This is a computational research prototype. Not a real vaccine, not manufacturing, not diagnosis or treatment.\n"
        "4. If the user asked for Chempy, use the chempy library (Reaction / ReactionSystem, optional ODE kinetics) with placeholder rates.\n"
        "5. Use placeholder sequences and toy kinetics only. No wet-lab protocol.\n"
        "6. If evidence is thin, still write a clearly labeled prototype and say so in comments.\n\n"
        f"User request:\n{question.strip()}\n"
        f"{draft}\n"
        f"ScienceOpen preprint passages:\n{format_passages(hits, limit=4)}\n\n"
        "Write the file now. Output only code."
    )


def fallback_research_script(user_text: str, hits: Sequence[RetrievedChunk], transcript: str = "") -> str:
    title, doi, url, passage = "unindexed ScienceOpen preprint", "n/a", "", "No matching preprint was in tensor memory."
    ev = select_artifact_evidence(user_text, hits, transcript)
    if ev is not None:
        title = " ".join((ev.title or "").split())[:180] or title
        doi = ev.doi or doi
        url = ev.url or ""
        passage = " ".join((ev.text or "").split())[:700] or passage
    lower = (user_text or "").lower()
    if "chempy" in lower or ("python" in lower and any(k in lower for k in ("vaccine", "kinetic", "mrna"))):
        return _chempy_stub(title, doi, url, passage)
    return _python_stub(title, doi, url, passage)


def _python_stub(title: str, doi: str, url: str, passage: str) -> str:
    return f'''#!/usr/bin/env python3
"""Computational research prototype grounded in a ScienceOpen preprint.

Title: {title}
DOI: {doi}
URL: {url}

Passage (unreviewed preprint):
    {passage}

This is NOT a vaccine, NOT a wet-lab protocol, and NOT clinical advice.
Stella V only retrieved evidence; this file is a research sketch for local experimentation.
"""

from __future__ import annotations

EVIDENCE = {{
    "title": {title!r},
    "doi": {doi!r},
    "url": {url!r},
    "unreviewed": True,
}}


def main() -> None:
    print("ScienceOpen DOI:", EVIDENCE["doi"])
    print("Title:", EVIDENCE["title"])
    print("Unreviewed preprint. Computational sketch only.")


if __name__ == "__main__":
    main()
'''


def _chempy_stub(title: str, doi: str, url: str, passage: str) -> str:
    return f'''#!/usr/bin/env python3
"""Toy Chempy kinetics for a personalized mRNA cancer-vaccine *research sketch*.

Grounded in ScienceOpen preprint:
    {title}
    DOI: {doi}
    URL: {url}

Passage (unreviewed):
    {passage}

This is NOT a vaccine, NOT manufacturing instructions, and NOT medical advice.
Species and rate constants are placeholders for a computational prototype:
    mRNA -> antigen protein -> immune-activation marker
"""

from __future__ import annotations

DOI = {doi!r}

try:
    from chempy import Reaction, ReactionSystem
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install chempy to run this research sketch: pip install chempy") from exc


def build_system() -> ReactionSystem:
    # Placeholder first-order rates (1/hour). Not fitted to data.
    reactions = [
        Reaction({{"mRNA": 1}}, {{"mRNA": 1, "antigen": 1}}, param=0.12, name="translation"),
        Reaction({{"mRNA": 1}}, {{}}, param=0.05, name="mrna_decay"),
        Reaction({{"antigen": 1}}, {{"signal": 1}}, param=0.03, name="presentation"),
        Reaction({{"antigen": 1}}, {{}}, param=0.02, name="antigen_clearance"),
        Reaction({{"signal": 1}}, {{}}, param=0.01, name="signal_decay"),
    ]
    return ReactionSystem(reactions, "mRNA antigen signal")


def main() -> None:
    rsys = build_system()
    print("ScienceOpen DOI:", DOI)
    print("Unreviewed preprint. Toy ReactionSystem only.")
    print(rsys)
    try:
        import numpy as np
        from chempy.kinetics.ode import get_odesys

        odesys, _extra = get_odesys(rsys)
        tout = np.linspace(0.0, 48.0, 97)
        result, _info = odesys.integrate(tout, {{"mRNA": 1.0, "antigen": 0.0, "signal": 0.0}})
        print("t_hours\\tmRNA\\tantigen\\tsignal")
        for t, row in zip(result.x, result.yout):
            print(f"{{float(t):.2f}}\\t{{row[0]:.6f}}\\t{{row[1]:.6f}}\\t{{row[2]:.6f}}")
    except Exception as exc:
        print("ODE integrate skipped:", exc)


if __name__ == "__main__":
    main()
'''
