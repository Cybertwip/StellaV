from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9_']+|[^\w\s]", re.IGNORECASE)
SECTION_KEYS = (
    "language",
    "methods",
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
)

DISCLAIMER = (
    "Stella V summarizes medical research and ScienceOpen preprints. "
    "This is not diagnosis, treatment, or personal medical advice."
)

GENERAL_TOPIC_OVERVIEW = (
    "I know medical-research topics: clinical trials, epidemiology, oncology, "
    "cardiology, immunology, infectious disease, pharmacology, genomics, "
    "neurology, and public health. I retrieve ScienceOpen preprints with a "
    "tensor engine and reason over them with a 1.5B or 3B model."
)


@dataclass(frozen=True)
class ConceptKnowledge:
    name: str
    explanation: str


@dataclass(frozen=True)
class TopicKnowledge:
    key: str
    title: str
    section: str
    aliases: Sequence[str]
    overview: str
    why_it_matters: str
    best_practices: Sequence[str]
    pitfalls: Sequence[str]
    concepts: Sequence[ConceptKnowledge]


TOPICS: Sequence[TopicKnowledge] = (
    TopicKnowledge(
        key="clinical_trials",
        title="clinical trials",
        section="clinical_trials",
        aliases=("clinical trial", "rct", "randomized trial", "blinding", "endpoint"),
        overview="Clinical trials test interventions in humans using protocol-defined populations, randomization, endpoints, and safety monitoring.",
        why_it_matters="Trial design determines whether an observed effect can be attributed to the intervention rather than bias, chance, or confounding.",
        best_practices=(
            "Pre-specify primary endpoints, analysis sets, and stopping rules.",
            "Use randomization and allocation concealment to limit selection bias.",
            "Report according to CONSORT and separate efficacy from safety interpretation.",
        ),
        pitfalls=(
            "Changing the primary endpoint after unblinded looks inflates false-positive risk.",
            "Underpowered trials can miss clinically important effects.",
            "Poor inclusion criteria can make results hard to generalize.",
        ),
        concepts=(
            ConceptKnowledge("randomization", "Randomization assigns participants to arms by chance so measured and unmeasured confounders are balanced in expectation."),
            ConceptKnowledge("blinding", "Blinding keeps participants, clinicians, or assessors unaware of assignment to reduce performance and detection bias."),
            ConceptKnowledge("primary endpoint", "The primary endpoint is the pre-specified outcome that the trial is mainly powered to evaluate."),
        ),
    ),
    TopicKnowledge(
        key="epidemiology",
        title="epidemiology",
        section="epidemiology",
        aliases=("epidemiology", "incidence", "prevalence", "cohort", "case control", "confounding"),
        overview="Epidemiology studies the distribution and determinants of disease in populations using observational and interventional designs.",
        why_it_matters="Population evidence is how researchers estimate risk, identify causes, and judge whether an association is likely causal.",
        best_practices=(
            "State the target population, exposure window, and outcome definition.",
            "Address confounding with design and analysis rather than after-the-fact storytelling.",
            "Report absolute and relative measures when both are meaningful.",
        ),
        pitfalls=(
            "Confounding by indication can make a treatment look harmful or protective for the wrong reason.",
            "Selection into a study can distort incidence and risk ratios.",
            "Reverse causation is common when disease changes the measured exposure.",
        ),
        concepts=(
            ConceptKnowledge("incidence", "Incidence is the rate of new cases in a population over a defined time period."),
            ConceptKnowledge("prevalence", "Prevalence is the proportion of a population that has the condition at a given time."),
            ConceptKnowledge("confounding", "Confounding occurs when a third factor associated with both exposure and outcome distorts the estimated effect."),
        ),
    ),
    TopicKnowledge(
        key="oncology",
        title="oncology",
        section="oncology",
        aliases=("oncology", "cancer", "tumor", "immunotherapy", "checkpoint", "chemotherapy"),
        overview="Oncology research studies tumor biology, staging, systemic therapy, radiotherapy, surgery, and immune-based treatments.",
        why_it_matters="Cancer outcomes depend on tumor type, stage, molecular features, and whether therapy hits a validated biological target.",
        best_practices=(
            "Separate histology, stage, and molecular subtype when comparing treatments.",
            "Use validated biomarkers before claiming a precision-oncology benefit.",
            "Interpret surrogate endpoints such as response rate in light of survival and toxicity.",
        ),
        pitfalls=(
            "Tumor shrinkage is not always a surrogate for longer survival.",
            "Single-arm early studies overstate benefit without a control.",
            "Immune-related adverse events can be delayed and easy to miss.",
        ),
        concepts=(
            ConceptKnowledge("checkpoint inhibitor", "Checkpoint inhibitors block inhibitory immune receptors such as PD-1 or CTLA-4 so T cells can attack tumor cells."),
            ConceptKnowledge("biomarker", "A biomarker is a measurable feature used to diagnose disease, predict outcome, or select therapy."),
            ConceptKnowledge("staging", "Staging describes anatomic extent of cancer and is a major determinant of prognosis and treatment intent."),
        ),
    ),
    TopicKnowledge(
        key="cardiology",
        title="cardiology",
        section="cardiology",
        aliases=("cardiology", "heart failure", "atherosclerosis", "myocardial infarction", "blood pressure"),
        overview="Cardiology research covers atherosclerosis, ischemic heart disease, heart failure, arrhythmia, and prevention of cardiovascular events.",
        why_it_matters="Cardiovascular disease is a leading cause of death, so small risk reductions in trials can matter at population scale.",
        best_practices=(
            "Report both relative and absolute risk reductions for preventive therapies.",
            "Account for competing risks in older populations.",
            "Distinguish HFrEF from HFpEF when discussing heart-failure treatments.",
        ),
        pitfalls=(
            "Surrogate lipid or blood-pressure changes do not automatically prove outcome benefit.",
            "Trial populations may be younger and less comorbid than usual care.",
            "Adherence and adverse effects can erase an apparent efficacy gain.",
        ),
        concepts=(
            ConceptKnowledge("atherosclerosis", "Atherosclerosis is lipid-driven arterial plaque formation that can cause ischemia, infarction, and stroke."),
            ConceptKnowledge("heart failure", "Heart failure is a clinical syndrome in which the heart cannot meet metabolic demand without elevated filling pressures."),
            ConceptKnowledge("myocardial infarction", "Myocardial infarction is ischemic necrosis of heart muscle, usually from coronary plaque rupture or occlusion."),
        ),
    ),
    TopicKnowledge(
        key="immunology",
        title="immunology",
        section="immunology",
        aliases=("immunology", "vaccine", "antibody", "t cell", "innate immunity", "adaptive immunity"),
        overview="Immunology studies innate and adaptive defense, vaccination, autoimmunity, and immune-mediated tissue injury.",
        why_it_matters="Immune mechanisms explain infection control, vaccine effect, autoimmunity, and many modern biologics.",
        best_practices=(
            "Specify antigen, adjuvant, schedule, and correlate of protection when discussing vaccines.",
            "Separate cellular and humoral readouts.",
            "Look for off-target inflammation when evaluating immune therapies.",
        ),
        pitfalls=(
            "Antibody titers are not always correlates of protection.",
            "Mouse immunology often fails to translate to human disease.",
            "Immunosuppression benefit must be weighed against infection and malignancy risk.",
        ),
        concepts=(
            ConceptKnowledge("innate immunity", "Innate immunity is the rapid, relatively nonspecific first line of host defense, including barriers, complement, and myeloid cells."),
            ConceptKnowledge("adaptive immunity", "Adaptive immunity is antigen-specific T and B cell immunity that can form immunological memory."),
            ConceptKnowledge("vaccine", "A vaccine presents antigen in a controlled way so the host develops protective memory without the full disease."),
        ),
    ),
    TopicKnowledge(
        key="infectious",
        title="infectious disease",
        section="infectious",
        aliases=("infectious", "infection", "antimicrobial resistance", "sepsis", "pathogen", "virus"),
        overview="Infectious-disease research studies pathogens, host response, diagnostics, antimicrobials, and infection control.",
        why_it_matters="Pathogen evolution, resistance, and delayed diagnosis still drive large morbidity and mortality.",
        best_practices=(
            "Use stewardship: right drug, dose, duration, and de-escalation.",
            "Confirm infection with microbiology when it changes therapy.",
            "Report resistance context, not just a drug name.",
        ),
        pitfalls=(
            "Treating colonization as infection drives resistance.",
            "Sepsis bundles without source control do not replace treating the focus.",
            "In vitro susceptibility may not predict clinical success at the infection site.",
        ),
        concepts=(
            ConceptKnowledge("antimicrobial resistance", "Antimicrobial resistance is the ability of a microbe to survive a drug that would normally inhibit or kill it."),
            ConceptKnowledge("sepsis", "Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection."),
            ConceptKnowledge("source control", "Source control means draining, removing, or otherwise eliminating the focus of infection."),
        ),
    ),
    TopicKnowledge(
        key="pharmacology",
        title="pharmacology",
        section="pharmacology",
        aliases=("pharmacology", "pharmacokinetics", "pharmacodynamics", "dose", "adverse event", "therapeutic index"),
        overview="Pharmacology studies how drugs are absorbed, distributed, metabolized, and excreted, and how they act on targets.",
        why_it_matters="Dose, exposure, and safety margins decide whether a biologically active compound is a usable medicine.",
        best_practices=(
            "Link PK exposure to PD effect instead of dose alone.",
            "Look for CYP, transporter, and renal/hepatic adjustments.",
            "Collect adverse events with the same rigor as efficacy.",
        ),
        pitfalls=(
            "A wide in-vitro potency gap can collapse in vivo because of clearance or protein binding.",
            "Drug-drug interactions are easy to miss in narrow-index agents.",
            "Off-label dose extrapolation from adults to children is unsafe without PK data.",
        ),
        concepts=(
            ConceptKnowledge("pharmacokinetics", "Pharmacokinetics describes concentration over time: absorption, distribution, metabolism, and excretion."),
            ConceptKnowledge("pharmacodynamics", "Pharmacodynamics describes what the drug does to the body as a function of exposure."),
            ConceptKnowledge("therapeutic index", "The therapeutic index is the margin between efficacious and toxic exposure."),
        ),
    ),
    TopicKnowledge(
        key="genomics",
        title="genomics",
        section="genomics",
        aliases=("genomics", "gwas", "crispr", "precision medicine", "variant", "gene therapy"),
        overview="Genomics research maps variants, expression, and gene function to disease risk and treatment response.",
        why_it_matters="Genetic evidence can identify targets, stratify patients, and support gene-directed therapies.",
        best_practices=(
            "Require replication and multiple-testing control in GWAS.",
            "Report ancestry and variant interpretation guidelines such as ACMG when relevant.",
            "Treat polygenic scores as probabilistic, not diagnostic.",
        ),
        pitfalls=(
            "Association is not mechanism; a GWAS hit may be a marker not a cause.",
            "Euro-centric cohorts reduce portability of polygenic scores.",
            "Off-target editing is a central safety issue for CRISPR therapeutics.",
        ),
        concepts=(
            ConceptKnowledge("GWAS", "A genome-wide association study tests many common variants for association with a trait, with strict multiple-testing control."),
            ConceptKnowledge("CRISPR", "CRISPR systems can cut or edit DNA at programmed sites and are being developed as research tools and therapeutics."),
            ConceptKnowledge("precision medicine", "Precision medicine uses biological markers, often genomic, to match an intervention to a defined subgroup."),
        ),
    ),
    TopicKnowledge(
        key="neurology",
        title="neurology",
        section="neurology",
        aliases=("neurology", "alzheimer", "parkinson", "stroke", "neurodegeneration", "biomarker"),
        overview="Neurology research studies stroke, neurodegeneration, epilepsy, neuroinflammation, and nervous-system repair.",
        why_it_matters="The brain's limited regeneration and long disease prodromes make early biomarkers and prevention especially important.",
        best_practices=(
            "Use standardized clinical scales plus imaging or fluid biomarkers when available.",
            "Separate disease modification from symptomatic benefit.",
            "Account for long follow-up and competing mortality in dementia trials.",
        ),
        pitfalls=(
            "Amyloid lowering is not automatically clinical benefit.",
            "Small motor-score changes may not be meaningful to patients.",
            "Hospital stroke trials can miss pre-hospital delays that dominate outcome.",
        ),
        concepts=(
            ConceptKnowledge("neurodegeneration", "Neurodegeneration is progressive loss of neurons and their networks, as in Alzheimer and Parkinson disease."),
            ConceptKnowledge("stroke", "Stroke is acute focal brain injury from ischemia or hemorrhage."),
            ConceptKnowledge("fluid biomarker", "A fluid biomarker is a measurable analyte in CSF, blood, or other fluid used for diagnosis or monitoring."),
        ),
    ),
    TopicKnowledge(
        key="public_health",
        title="public health",
        section="public_health",
        aliases=("public health", "screening", "prevention", "health equity", "vaccination program"),
        overview="Public-health research evaluates prevention, screening, vaccination programs, health systems, and population outcomes.",
        why_it_matters="Population interventions can prevent more disease than late treatment, but they must be balanced against harm, cost, and equity.",
        best_practices=(
            "Use screening criteria: important disease, acceptable test, effective follow-up, and net benefit.",
            "Report uptake and equity, not only efficacy in trial volunteers.",
            "Evaluate implementation as carefully as the biological intervention.",
        ),
        pitfalls=(
            "Overdiagnosis can make screening look beneficial when it mainly finds indolent disease.",
            "Programs that miss marginalized groups widen health gaps.",
            "Short-term process metrics can hide weak outcome impact.",
        ),
        concepts=(
            ConceptKnowledge("screening", "Screening tests asymptomatic people to find disease earlier, which only helps if earlier treatment improves outcomes."),
            ConceptKnowledge("prevention", "Prevention reduces incidence or severity before advanced disease, spanning vaccines, risk-factor control, and policy."),
            ConceptKnowledge("health equity", "Health equity means reducing unfair, avoidable differences in health access and outcomes across groups."),
        ),
    ),
)

TOPIC_BY_KEY: Dict[str, TopicKnowledge] = {topic.key: topic for topic in TOPICS}

_CODE_INTENT_RE = re.compile(r"\b(write|show|example|sample|snippet|implement|code|starter|demo)\b", re.IGNORECASE)
_BEST_PRACTICE_RE = re.compile(r"\b(best practice|best practices|recommend|recommended|tips)\b", re.IGNORECASE)
_PITFALL_RE = re.compile(r"\b(pitfall|pitfalls|avoid|mistake|common bug|common bugs|wrong)\b", re.IGNORECASE)
_WHY_RE = re.compile(r"\bwhy\b|\bimportant\b|\bimportance\b|\bmatter\b", re.IGNORECASE)
_CAPABILITY_RE = re.compile(r"\bwhat can you do\b|\bcapabilities\b|\bhelp with\b", re.IGNORECASE)
_TOPICS_RE = re.compile(r"\bwhat topics\b|\bknowledge base\b|\bwhat do you know\b", re.IGNORECASE)
_GREETING_SET = {"hello", "hi", "hi stellav", "hey", "hey stellav", "hi stella", "hello stella"}


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _detect_topic(question: str) -> TopicKnowledge | None:
    text = question.lower()
    best: TopicKnowledge | None = None
    best_score = 0
    for topic in TOPICS:
        score = 0
        for alias in (topic.key, topic.title, *topic.aliases):
            alias_text = alias.lower()
            if alias_text in text:
                score += (5 if alias_text == topic.key else 3) + len(alias_text.split())
        for concept in topic.concepts:
            if concept.name.lower() in text:
                score += 6 + len(concept.name.split())
        if score > best_score:
            best = topic
            best_score = score
    return best


def _join_list(prefix: str, items: Sequence[str]) -> str:
    clean_items = [item.strip().rstrip(".") for item in items if item.strip()]
    if not clean_items:
        return prefix.rstrip()
    if len(clean_items) == 1:
        return f"{prefix}{clean_items[0]}."
    if len(clean_items) == 2:
        return f"{prefix}{clean_items[0]} and {clean_items[1]}."
    return f"{prefix}{', '.join(clean_items[:-1])}, and {clean_items[-1]}."


def _detect_concept(question: str) -> tuple[TopicKnowledge, ConceptKnowledge] | None:
    text = question.lower()
    matches: list[tuple[int, TopicKnowledge, ConceptKnowledge]] = []
    for topic in TOPICS:
        for concept in topic.concepts:
            concept_text = concept.name.lower()
            if concept_text in text:
                matches.append((len(concept_text), topic, concept))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    _length, topic, concept = matches[0]
    return topic, concept


def semantic_response(question: str) -> tuple[str, float] | None:
    text = question.strip()
    lowered = text.lower()
    if not text:
        return None

    if "who are you" in lowered:
        return (
            "I am Stella V, a local medical-research assistant. I retrieve ScienceOpen preprints "
            "with a GPU tensor engine and reason over them with a selectable 1.5B or 3B model. "
            + DISCLAIMER,
            0.95,
        )
    if _CAPABILITY_RE.search(lowered):
        return (
            "I can search ScienceOpen preprints, rank them with tensor retrieval, and reason about "
            "clinical trials, epidemiology, oncology, cardiology, immunology, infection, pharmacology, "
            "genomics, neurology, and public health. " + DISCLAIMER,
            0.94,
        )
    if _TOPICS_RE.search(lowered):
        return GENERAL_TOPIC_OVERVIEW, 0.94
    if lowered in _GREETING_SET:
        return "Hello. I am Stella V, a medical-research assistant.", 0.95
    if "not medical advice" in lowered or "disclaimer" in lowered:
        return DISCLAIMER, 0.96
    if "scienceopen" in lowered:
        return (
            "ScienceOpen hosts and indexes research, including preprints with Crossref prefix 10.14293. "
            "Stella V searches those preprints, embeds them as tensors, and reasons over the ranked abstracts.",
            0.93,
        )
    if "preprint" in lowered:
        return (
            "A preprint is a complete manuscript shared before journal peer review. It speeds communication "
            "but should be read as unreviewed evidence, especially in medicine.",
            0.93,
        )
    if "tensor" in lowered or "reasoning model" in lowered or "1.5b" in lowered or "3b" in lowered:
        return (
            "Stella V keeps a linear tensor memory for fast retrieval and a selectable 1.5B or 3B reasoning "
            "model that uses that memory as a backend to pick and explain ScienceOpen evidence.",
            0.92,
        )

    concept_match = _detect_concept(lowered)
    topic = concept_match[0] if concept_match else _detect_topic(lowered)
    if topic is None:
        return None
    if _CODE_INTENT_RE.search(lowered):
        return (
            f"I do not generate game or engine code. For {topic.title}, I summarize research methods and preprint evidence. "
            + DISCLAIMER,
            0.84,
        )
    if concept_match is not None:
        return concept_match[1].explanation, 0.93
    if _BEST_PRACTICE_RE.search(lowered):
        return _join_list(f"Good practices in {topic.title} are ", topic.best_practices), 0.91
    if _PITFALL_RE.search(lowered):
        return _join_list(f"Common pitfalls in {topic.title} are ", topic.pitfalls), 0.91
    if _WHY_RE.search(lowered):
        return topic.why_it_matters, 0.91
    return topic.overview, 0.9


def _base_general_items() -> List[dict[str, str]]:
    return [
        {"section": "language", "question": "Who are you?", "answer": "I am Stella V, a local medical-research assistant that uses ScienceOpen preprints and a 1.5B or 3B reasoner."},
        {"section": "language", "question": "What can you do?", "answer": "I retrieve ScienceOpen medical preprints with a tensor engine and reason over the ranked evidence."},
        {"section": "language", "question": "How should you answer?", "answer": "I should answer from retrieved preprints and established methods, cite DOIs, and avoid clinical advice."},
        {"section": "language", "question": "What topics do you know?", "answer": GENERAL_TOPIC_OVERVIEW},
        {"section": "language", "question": "Hello", "answer": "Hello. I am Stella V, a medical-research assistant."},
        {"section": "language", "question": "Are you ready?", "answer": "Yes. I can search ScienceOpen preprints and reason about medical research."},
        {"section": "language", "question": "Is this medical advice?", "answer": DISCLAIMER},
        {"section": "language", "question": "What is a preprint?", "answer": "A preprint is a manuscript shared before journal peer review and should be treated as unreviewed evidence."},
        {"section": "language", "question": "What is ScienceOpen?", "answer": "ScienceOpen is a research discovery and publishing platform; Stella V uses it as the preprint source."},
        {"section": "methods", "question": "What is a primary endpoint?", "answer": "The primary endpoint is the pre-specified outcome a study is mainly designed to evaluate."},
        {"section": "methods", "question": "What is confounding?", "answer": "Confounding is distortion of an effect estimate by a third factor linked to both exposure and outcome."},
    ]


def _topic_dataset_items(topic: TopicKnowledge) -> List[dict[str, str]]:
    section = topic.section if topic.section in SECTION_KEYS else "language"
    items: List[dict[str, str]] = [
        {"section": section, "question": f"What is {topic.title}?", "answer": topic.overview},
        {"section": section, "question": f"Explain {topic.title} briefly.", "answer": topic.overview},
        {"section": section, "question": f"Tell me about {topic.title}.", "answer": topic.overview},
        {"section": section, "question": f"Why is {topic.title} important?", "answer": topic.why_it_matters},
        {"section": section, "question": f"Why does {topic.title} matter?", "answer": topic.why_it_matters},
        {"section": section, "question": f"What are best practices for {topic.title}?", "answer": _join_list(f"Good practices in {topic.title} are ", topic.best_practices)},
        {"section": section, "question": f"What should I avoid in {topic.title}?", "answer": _join_list(f"Common pitfalls in {topic.title} are ", topic.pitfalls)},
        {"section": section, "question": f"What are common pitfalls in {topic.title}?", "answer": _join_list(f"Common pitfalls in {topic.title} are ", topic.pitfalls)},
    ]
    for concept in topic.concepts:
        items.extend(
            [
                {"section": section, "question": f"What is {concept.name}?", "answer": concept.explanation},
                {"section": section, "question": f"Explain {concept.name}.", "answer": concept.explanation},
                {"section": section, "question": f"Why use {concept.name}?", "answer": concept.explanation},
            ]
        )
    return items


def generate_dataset_items() -> List[dict[str, str]]:
    items = _base_general_items()
    seen: set[tuple[str, str]] = {
        (item["question"].strip().lower(), item["answer"].strip().lower()) for item in items
    }
    for topic in TOPICS:
        for item in _topic_dataset_items(topic):
            key = (item["question"].strip().lower(), item["answer"].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def build_usage_items() -> List[dict[str, str]]:
    items = [
        {"phase": "usage_language", "section": "language", "question": "What is a preprint?", "answer": "A preprint is a manuscript shared before journal peer review."},
        {"phase": "usage_methods", "section": "methods", "question": "What is randomization?", "answer": "Randomization assigns participants by chance to balance confounders."},
        {"phase": "usage_clinical_trials", "section": "clinical_trials", "question": "What is blinding?", "answer": "Blinding hides treatment assignment to reduce bias."},
        {"phase": "usage_epidemiology", "section": "epidemiology", "question": "What is incidence?", "answer": "Incidence is the rate of new cases over time."},
        {"phase": "usage_oncology", "section": "oncology", "question": "What is a checkpoint inhibitor?", "answer": "A checkpoint inhibitor blocks inhibitory immune receptors so T cells can attack tumors."},
        {"phase": "usage_cardiology", "section": "cardiology", "question": "What is atherosclerosis?", "answer": "Atherosclerosis is lipid-driven arterial plaque that can cause ischemia."},
        {"phase": "usage_immunology", "section": "immunology", "question": "What does a vaccine do?", "answer": "A vaccine presents antigen so the host forms protective memory."},
        {"phase": "usage_infectious", "section": "infectious", "question": "What is antimicrobial resistance?", "answer": "Antimicrobial resistance is microbial survival despite a normally active drug."},
        {"phase": "usage_pharmacology", "section": "pharmacology", "question": "What is pharmacokinetics?", "answer": "Pharmacokinetics describes concentration over time after a dose."},
        {"phase": "usage_genomics", "section": "genomics", "question": "What is a GWAS?", "answer": "A GWAS tests genome-wide variant associations with a trait."},
        {"phase": "usage_neurology", "section": "neurology", "question": "What is neurodegeneration?", "answer": "Neurodegeneration is progressive loss of neurons and their networks."},
        {"phase": "usage_public_health", "section": "public_health", "question": "What is screening?", "answer": "Screening tests asymptomatic people to find disease earlier."},
    ]
    seen: set[tuple[str, str, str]] = set()
    out: List[dict[str, str]] = []
    for item in items:
        key = (item["phase"].lower(), item["question"].lower(), item["answer"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_vocab_sections() -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {section: [] for section in SECTION_KEYS}
    sections["language"].extend(
        [
            "hello", "yes", "no", "please", "stellav", "preprint", "scienceopen", "evidence",
            "study", "trial", "patient", "disease", "treatment", "risk", "outcome", "review",
            "question", "answer", "explain", "summarize", "cite", "doi", "research", "medical",
        ]
    )
    sections["methods"].extend(
        [
            "randomization", "blinding", "endpoint", "confounding", "bias", "cohort", "control",
            "power", "sample", "p-value", "confidence", "interval", "hazard", "odds", "ratio",
        ]
    )
    for item in generate_dataset_items():
        section = item["section"] if item["section"] in SECTION_KEYS else "language"
        sections["language"].extend(tokenize(item["question"]))
        sections[section].extend(tokenize(item["answer"]))
    for topic in TOPICS:
        section = topic.section if topic.section in SECTION_KEYS else "language"
        sections["language"].extend(tokenize(topic.title))
        for alias in topic.aliases:
            sections["language"].extend(tokenize(alias))
        sections[section].extend(tokenize(topic.overview))
        sections[section].extend(tokenize(topic.why_it_matters))
        for sentence in (*topic.best_practices, *topic.pitfalls):
            sections[section].extend(tokenize(sentence))
        for concept in topic.concepts:
            sections[section].extend(tokenize(concept.name))
            sections[section].extend(tokenize(concept.explanation))
    clean: Dict[str, List[str]] = {}
    for section, tokens in sections.items():
        seen: set[str] = set()
        unique: List[str] = []
        for token in tokens:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        clean[section] = unique
    return clean
