#!/usr/bin/env python3
"""ScienceOpen preprint client for Stella V.

ScienceOpen registers works under Crossref prefix 10.14293 and hosts them at
scienceopen.com. Preprints are Crossref type ``posted-content``. Abstracts are
fetched from Crossref (JATS) and, when needed, Europe PMC.
"""

from __future__ import annotations

import html
import os
import re
import time
import urllib.parse
from typing import Any, Iterable, Optional

import requests

SCIENCEOPEN_PREFIX = "10.14293"
SCIENCEOPEN_BASE = "https://www.scienceopen.com"
CROSSREF_WORKS = "https://api.crossref.org/works"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
USER_AGENT = os.environ.get(
    "STELLAV_USER_AGENT",
    "StellaV/1.0 (medical-research assistant; mailto:stellav@localhost)",
)
REQUEST_PAUSE = float(os.environ.get("STELLAV_SCIENCEOPEN_PAUSE", "0.2"))

_JATS_P_CLOSE = re.compile(r"</jats:p>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

MEDICAL_QUERY_HINTS = (
    "medicine",
    "clinical",
    "patient",
    "trial",
    "disease",
    "therapy",
    "treatment",
    "cancer",
    "cardio",
    "immun",
    "infect",
    "pharma",
    "genom",
    "neuro",
    "epidem",
    "public health",
    "diagnos",
    "biomarker",
    "vaccine",
    "hospital",
    "oncolog",
    "pathophys",
)


def normalize_space(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_jats(text: str) -> str:
    text = _JATS_P_CLOSE.sub("\n\n", str(text or ""))
    text = _TAG_RE.sub(" ", text)
    return normalize_space(html.unescape(text))


def scienceopen_url(doi: str) -> str:
    doi = str(doi or "").strip().removeprefix("https://doi.org/")
    if not doi:
        return SCIENCEOPEN_BASE
    return f"{SCIENCEOPEN_BASE}/hosted-document?doi={urllib.parse.quote(doi, safe='')}"


def is_scienceopen_doi(doi: str) -> bool:
    return str(doi or "").lower().startswith(SCIENCEOPEN_PREFIX)


def looks_medical(title: str, abstract: str = "") -> bool:
    blob = f"{title} {abstract}".lower()
    return any(hint in blob for hint in MEDICAL_QUERY_HINTS)


def _crossref_get(params: dict[str, Any]) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    resp = requests.get(CROSSREF_WORKS, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _work_to_preprint(item: dict[str, Any]) -> dict[str, Any]:
    title_raw = item.get("title") or []
    title = title_raw[0] if isinstance(title_raw, list) and title_raw else str(title_raw or "Untitled")
    authors = []
    for author in item.get("author") or []:
        given = str(author.get("given") or "").strip()
        family = str(author.get("family") or "").strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    issued = item.get("posted") or item.get("issued") or item.get("created") or {}
    date_parts = (issued.get("date-parts") or [[]])[0]
    year = int(date_parts[0]) if date_parts else None
    doi = str(item.get("DOI") or "").strip()
    abstract = strip_jats(str(item.get("abstract") or ""))
    subjects = [str(s) for s in (item.get("subject") or []) if s]
    return {
        "doi": doi,
        "title": normalize_space(title),
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "subjects": subjects,
        "type": str(item.get("type") or ""),
        "url": scienceopen_url(doi),
        "crossref_url": str(item.get("URL") or ""),
        "source": "scienceopen",
        "is_preprint": str(item.get("type") or "") == "posted-content" or "preprint" in str(item.get("subtype") or "").lower(),
        "cited_by": int(item.get("is-referenced-by-count") or 0),
    }


def search_scienceopen(
    query: str,
    *,
    rows: int = 12,
    preprints_only: bool = True,
    medical_only: bool = True,
    has_abstract: bool = True,
) -> list[dict[str, Any]]:
    """Search ScienceOpen-registered works via Crossref prefix 10.14293."""
    query = normalize_space(query)
    if not query:
        return []
    filters = [f"prefix:{SCIENCEOPEN_PREFIX}"]
    if preprints_only:
        filters.append("type:posted-content")
    if has_abstract:
        filters.append("has-abstract:true")
    params = {
        "query": query,
        "filter": ",".join(filters),
        "rows": max(1, min(int(rows), 50)),
        "select": "DOI,title,author,abstract,type,URL,issued,created,is-referenced-by-count",
        "mailto": "stellav@localhost",
    }
    try:
        data = _crossref_get(params)
    except requests.HTTPError:
        params.pop("select", None)
        try:
            data = _crossref_get(params)
        except Exception:
            return search_europepmc_scienceopen(query, rows=rows)
    items = ((data.get("message") or {}).get("items")) or []
    out: list[dict[str, Any]] = []
    for item in items:
        paper = _work_to_preprint(item)
        if not paper["doi"]:
            continue
        if medical_only and not looks_medical(paper["title"], paper["abstract"]):
            continue
        if has_abstract and not paper["abstract"]:
            continue
        out.append(paper)
    time.sleep(REQUEST_PAUSE)
    if not out:
        return search_europepmc_scienceopen(query, rows=rows)
    return out


def fetch_scienceopen_doi(doi: str) -> dict[str, Any]:
    doi = str(doi or "").strip().removeprefix("https://doi.org/")
    if not doi:
        raise ValueError("DOI is required")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = f"{CROSSREF_WORKS}/{urllib.parse.quote(doi)}"
    resp = requests.get(url, headers=headers, params={"mailto": "stellav@localhost"}, timeout=30)
    resp.raise_for_status()
    item = (resp.json().get("message") or {})
    paper = _work_to_preprint(item)
    if not paper["abstract"]:
        paper["abstract"] = fetch_europepmc_abstract(doi)
    return paper


def fetch_europepmc_abstract(doi: str) -> str:
    doi = str(doi or "").strip()
    if not doi:
        return ""
    params = {
        "query": f'DOI:"{doi}"',
        "format": "json",
        "pageSize": 1,
        "resultType": "core",
    }
    try:
        resp = requests.get(EUROPEPMC_SEARCH, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        results = (((resp.json().get("resultList") or {}).get("result")) or [])
        if not results:
            return ""
        return normalize_space(str(results[0].get("abstractText") or ""))
    except Exception:
        return ""


def search_europepmc_scienceopen(query: str, *, rows: int = 8) -> list[dict[str, Any]]:
    """Secondary discovery through Europe PMC's ScienceOpen Preprints records."""
    query = normalize_space(query)
    if not query:
        return []
    params = {
        "query": f'(PUBLISHER:"ScienceOpen Preprints" OR PUBLISHER:"ScienceOpen") AND ({query})',
        "format": "json",
        "pageSize": max(1, min(int(rows), 25)),
        "resultType": "core",
    }
    resp = requests.get(EUROPEPMC_SEARCH, params=params, headers={"User-Agent": USER_AGENT}, timeout=25)
    resp.raise_for_status()
    results = (((resp.json().get("resultList") or {}).get("result")) or [])
    out: list[dict[str, Any]] = []
    for item in results:
        doi = str(item.get("doi") or "").strip()
        title = normalize_space(str(item.get("title") or "Untitled"))
        abstract = normalize_space(str(item.get("abstractText") or ""))
        if not doi:
            continue
        out.append({
            "doi": doi,
            "title": title,
            "authors": [part.strip() for part in str(item.get("authorString") or "").split(",") if part.strip()],
            "year": int(item["pubYear"]) if str(item.get("pubYear") or "").isdigit() else None,
            "abstract": abstract,
            "subjects": [],
            "type": str(item.get("pubType") or "preprint"),
            "url": scienceopen_url(doi) if is_scienceopen_doi(doi) else f"https://doi.org/{doi}",
            "crossref_url": f"https://doi.org/{doi}",
            "source": "scienceopen-europepmc",
            "is_preprint": "preprint" in str(item.get("pubType") or "").lower(),
            "cited_by": int(item.get("citedByCount") or 0),
        })
    time.sleep(REQUEST_PAUSE)
    return out


DEFAULT_CATEGORY_QUERIES: dict[str, list[str]] = {
    "clinical_trials": [
        "randomized controlled trial methodology",
        "clinical trial endpoint blinding",
        "adaptive clinical trial design",
    ],
    "epidemiology": [
        "epidemiology confounding bias",
        "incidence prevalence cohort study",
        "causal inference observational study",
    ],
    "oncology": [
        "cancer immunotherapy checkpoint",
        "tumor microenvironment oncology",
        "cancer biomarker precision oncology",
    ],
    "cardiology": [
        "heart failure clinical research",
        "atherosclerosis cardiovascular risk",
        "myocardial infarction outcomes",
    ],
    "immunology": [
        "vaccine immune response",
        "autoimmunity T cell B cell",
        "innate adaptive immunity",
    ],
    "infectious": [
        "antimicrobial resistance infection",
        "viral pathogenesis clinical",
        "sepsis infectious disease",
    ],
    "pharmacology": [
        "pharmacokinetics pharmacodynamics",
        "drug safety adverse event",
        "therapeutic index clinical pharmacology",
    ],
    "genomics": [
        "GWAS precision medicine",
        "CRISPR gene therapy clinical",
        "genomic biomarker companion diagnostic",
    ],
    "neurology": [
        "alzheimer biomarker neurodegeneration",
        "stroke clinical neurology",
        "parkinson disease research",
    ],
    "public_health": [
        "public health screening prevention",
        "health equity epidemiology",
        "vaccination program effectiveness",
    ],
}


def default_scienceopen_queries(categories: Optional[Iterable[str]] = None) -> dict[str, list[str]]:
    if categories is None:
        chosen = list(DEFAULT_CATEGORY_QUERIES)
    else:
        chosen = [c for c in categories if c in DEFAULT_CATEGORY_QUERIES]
    return {cat: list(DEFAULT_CATEGORY_QUERIES[cat]) for cat in chosen}


def paper_text(paper: dict[str, Any]) -> str:
    authors = ", ".join(paper.get("authors") or [])
    year = paper.get("year") or "n.d."
    bits = [
        str(paper.get("title") or ""),
        f"Authors: {authors}" if authors else "",
        f"Year: {year}",
        f"DOI: {paper.get('doi') or ''}",
        str(paper.get("abstract") or ""),
    ]
    return normalize_space("\n".join(bit for bit in bits if bit))
