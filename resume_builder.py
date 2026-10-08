"""
resume_builder.py

Tier 2B: Dynamic Resume & Content Engine (AutoApply AI)

For a given job (must be status='SCORED'), this module:
  1. Sends the full master_resume.json content + job info to Gemini, asking it
     to select the most relevant 3-4 projects and experiences, reorder their
     bullets so the most job-relevant ones lead, and optionally rephrase
     bullets into a tighter "X by Y using Z" format.
  2. VALIDATES every rephrased bullet against the ORIGINAL bullet text: if the
     rephrase introduces a number or a known tech/skill term that wasn't in
     the original bullet, the rephrase is rejected and the original verbatim
     text is used instead. This is the zero-hallucination guardrail -- the
     LLM can select, reorder, and lightly polish phrasing, but cannot invent
     facts that make it into your actual resume.
  3. Injects the validated content into templates/resume_template.typ and
     compiles it to a PDF via the Typst CLI.

Usage:
    python resume_builder.py <job_id>
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from database import Database, DatabaseError, JobRecord

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MASTER_RESUME_PATH = os.getenv("MASTER_RESUME_PATH", "./master_resume.json")
OUTPUT_DIR = os.getenv("RESUME_OUTPUT_DIR", "./tailored_resumes")
TEMPLATE_PATH = os.getenv("RESUME_TEMPLATE_PATH", "./templates/resume_template.typ")
TYPST_BINARY = os.getenv("TYPST_BINARY", "typst")

REQUEST_DELAY_SECONDS = float(os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "5"))
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2

MAX_PROJECTS = 4
MAX_EXPERIENCES = 4


class ResumeBuilderError(Exception):
    """Raised for resume-builder-level failures the caller should handle explicitly."""


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class BulletCandidate:
    original_index: int
    original_text: str
    proposed_text: str
    final_text: str = ""  # set after validation
    was_rewritten: bool = False
    rejected_reason: Optional[str] = None


@dataclass
class SectionSelection:
    kind: str  # "project" or "experience"
    name: str
    bullets: list[BulletCandidate] = field(default_factory=list)


# ----------------------------------------------------------------------
# Loading + vocabulary building
# ----------------------------------------------------------------------

def load_master_resume(path: str = MASTER_RESUME_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ResumeBuilderError(f"Failed to load master resume at {path}: {e}") from e


def build_vocabulary(resume: dict) -> set[str]:
    """
    Build the set of 'known' tech/skill terms from the ENTIRE resume. Used by
    the hallucination guardrail: a rephrased bullet is allowed to reference
    any term already established somewhere in the candidate's real resume,
    but not a term invented from nothing.
    """
    vocab: set[str] = set()
    for items in resume.get("skills", {}).values():
        if isinstance(items, list):
            vocab.update(items)
    for proj in resume.get("projects", []):
        vocab.update(proj.get("technologies", []))
    return {v.strip() for v in vocab if v and v.strip()}


# ----------------------------------------------------------------------
# Hallucination guardrail
# ----------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d+\.?\d*")
# Matches capitalized/technical-looking tokens: "FastAPI", "C#", "Node.js", "REST", etc.
# Used to catch ANY new technical term introduced in a rephrase, not just ones
# that happen to appear elsewhere in the resume.
_TECH_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9+.#/]*\b")

# Common words that start capitalized but aren't technologies -- excluded so
# they don't trigger false-positive rejections (e.g. the first word of a
# sentence, or common resume verbs/nouns).
_COMMON_WORDS_IGNORE = {
    "the", "a", "an", "i", "designed", "built", "developed", "implemented",
    "created", "led", "managed", "delivered", "achieved", "improved",
    "reduced", "increased", "accomplished", "using", "by", "with", "for",
    "and", "or", "to", "of", "in", "on", "at",
}


def validate_rephrase(original: str, candidate: str, vocab: set[str]) -> tuple[bool, Optional[str]]:
    """
    Returns (is_valid, rejection_reason).

    A rephrase is REJECTED if:
      - it contains a number not present in the original bullet text, OR
      - it contains a known vocabulary term (from the candidate's real resume)
        that was NOT present in the original bullet text.

    This allows honest reformatting/reordering of an existing fact, but blocks
    the model from inventing a new metric or claiming an unrelated technology
    was used in that specific bullet.
    """
    orig_numbers = set(_NUMBER_PATTERN.findall(original))
    cand_numbers = set(_NUMBER_PATTERN.findall(candidate))
    new_numbers = cand_numbers - orig_numbers
    if new_numbers:
        return False, f"introduced number(s) not in original: {new_numbers}"

    orig_lower = original.lower()
    cand_lower = candidate.lower()

    # Check 1: any known vocab term (from elsewhere in the candidate's real
    # resume) that wasn't in THIS bullet -- catches cross-contamination
    # between projects/experiences.
    for term in vocab:
        term_lower = term.lower()
        if len(term_lower) < 3:
            continue
        if term_lower in cand_lower and term_lower not in orig_lower:
            return False, f"introduced term not in original bullet: '{term}'"

    # Check 2: any capitalized/technical-looking token in the rephrase that
    # doesn't appear anywhere in the original bullet -- catches totally
    # fabricated technologies that aren't even elsewhere in the resume
    # (e.g. inventing "Kubernetes" out of nowhere).
    orig_tokens = {t.lower() for t in _TECH_TOKEN_PATTERN.findall(original)}
    cand_tokens = _TECH_TOKEN_PATTERN.findall(candidate)
    for token in cand_tokens:
        token_lower = token.lower()
        if token_lower in _COMMON_WORDS_IGNORE:
            continue
        if token_lower not in orig_tokens and token_lower not in orig_lower:
            return False, f"introduced unrecognized term not in original bullet: '{token}'"

    return True, None


# ----------------------------------------------------------------------
# Gemini content selection
# ----------------------------------------------------------------------

SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selections": {
            "type": "array",
            "description": "Ordered list of experience/project sections to include, most relevant first",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["project", "experience"]},
                    "name": {"type": "string", "description": "Must exactly match a provided project/company name"},
                    "bullets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_index": {"type": "integer"},
                                "text": {
                                    "type": "string",
                                    "description": "Either the original bullet verbatim, or a tightly rephrased X-by-Y-using-Z version. Never invent new facts, numbers, or technologies not already in the original bullet.",
                                },
                            },
                            "required": ["original_index", "text"],
                        },
                    },
                },
                "required": ["kind", "name", "bullets"],
            },
        },
    },
    "required": ["selections"],
}

SYSTEM_PROMPT = f"""You are an expert resume writer helping a candidate tailor their resume for a \
specific job application. You will be given the candidate's full resume (all projects and work \
experience, each bullet labeled with an index) and a job posting.

Your task:
1. Select the {MAX_PROJECTS} most relevant PROJECTS and up to {MAX_EXPERIENCES} most relevant WORK \
EXPERIENCES for this specific job, based on tech-stack and role overlap.
2. For EACH selected section, include ALL of its original bullets (not just one or two) -- reorder \
them so the most relevant ones to this job appear first, but do not drop bullets just because they \
seem less relevant. A typical experience/project has 3-5 bullets and all of them should normally \
be included, just reordered.
3. Optionally rephrase bullets into a tighter "Accomplished X, measured by Y, by doing Z" format.

CRITICAL RULES:
- NEVER invent a new technology, number, metric, or fact that isn't already in the original bullet.
- If you rephrase a bullet, only reorder/reword what's already there -- do not add anything new.
- If in doubt, return the original bullet text unchanged rather than risk adding something new.
- Every bullet you return must reference an original_index from the provided list.
- Do not fabricate section names -- 'name' must exactly match a project or company name provided.
- Do not omit bullets from a selected section unless there are more than 5 -- in that case keep the \
5 most relevant ones.

Respond ONLY with the requested JSON structure."""


def build_selection_prompt(resume: dict, job: JobRecord) -> tuple[str, dict[tuple[str, int], str]]:
    """
    Build the prompt text and a lookup table mapping (section_name, original_index)
    -> original bullet text, used later for validation.
    """
    lookup: dict[tuple[str, int], str] = {}
    lines = []

    lines.append("PROJECTS:")
    for proj in resume.get("projects", []):
        name = proj.get("name", "")
        techs = ", ".join(proj.get("technologies", []))
        lines.append(f"\n[PROJECT: {name}] (technologies: {techs})")
        for idx, bullet in enumerate(proj.get("highlights", [])):
            lookup[(name, idx)] = bullet
            lines.append(f"  [{idx}] {bullet}")

    lines.append("\n\nWORK EXPERIENCE:")
    for exp in resume.get("experience", []):
        name = exp.get("company", "")
        position = exp.get("position", "")
        lines.append(f"\n[EXPERIENCE: {name}] ({position})")
        for idx, bullet in enumerate(exp.get("highlights", [])):
            lookup[(name, idx)] = bullet
            lines.append(f"  [{idx}] {bullet}")

    resume_text = "\n".join(lines)

    prompt = (
        f"CANDIDATE'S FULL RESUME CONTENT:\n{resume_text}\n\n"
        f"---\n\n"
        f"JOB POSTING:\n"
        f"Title: {job.title}\n"
        f"Company: {job.company}\n"
        f"Description:\n{job.description or job.fit_reasoning or '(no description available)'}"
    )
    return prompt, lookup


def select_content(
    client: genai.Client, resume: dict, job: JobRecord, vocab: set[str]
) -> list[SectionSelection]:
    """
    Call Gemini to select/reorder/rephrase resume content for this job, then
    validate every rephrased bullet against the hallucination guardrail.
    """
    prompt, lookup = build_selection_prompt(resume, job)

    last_error: Optional[Exception] = None
    result = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=SELECTION_SCHEMA,
                    temperature=0.3,
                ),
            )
            result = json.loads(response.text)
            time.sleep(REQUEST_DELAY_SECONDS)
            break
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d: content selection call failed: %s", attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if result is None:
        raise ResumeBuilderError(f"Failed to get content selection after {MAX_RETRIES + 1} attempts: {last_error}")

    sections: list[SectionSelection] = []
    for sel in result.get("selections", []):
        kind = sel.get("kind")
        name = sel.get("name")
        raw_bullets = sel.get("bullets", [])

        section = SectionSelection(kind=kind, name=name)
        for b in raw_bullets:
            idx = b.get("original_index")
            proposed = b.get("text", "")

            original = lookup.get((name, idx))
            if original is None:
                logger.warning(
                    "Model referenced unknown section/index (%s, %s) -- skipping this bullet",
                    name, idx,
                )
                continue

            candidate = BulletCandidate(original_index=idx, original_text=original, proposed_text=proposed)

            if proposed.strip() == original.strip():
                candidate.final_text = original
                candidate.was_rewritten = False
            else:
                is_valid, reason = validate_rephrase(original, proposed, vocab)
                if is_valid:
                    candidate.final_text = proposed
                    candidate.was_rewritten = True
                else:
                    candidate.final_text = original
                    candidate.was_rewritten = False
                    candidate.rejected_reason = reason
                    logger.info(
                        "Rejected rephrase for '%s' bullet %d (%s) -- using original text instead",
                        name, idx, reason,
                    )

            section.bullets.append(candidate)

        if section.bullets:
            sections.append(section)

    return sections


# ----------------------------------------------------------------------
# Typst rendering
# ----------------------------------------------------------------------

def escape_typst(text: str) -> str:
    """Escape characters with special meaning in Typst markup."""
    special_chars = ["\\", "#", "$", "*", "_", "@", "`", "<", ">"]
    for ch in special_chars:
        text = text.replace(ch, "\\" + ch)
    return text


# Matches numbers/metrics worth bolding: "10+", "99%", "3.5", "20+", etc.
_BOLD_NUMBER_PATTERN = re.compile(r"\d+\.?\d*\+?%?")


def render_bullet_content(text: str, vocab: set[str]) -> str:
    """
    Build a Typst CONTENT block (not a plain string) for one bullet, with
    numbers/metrics and known tech terms wrapped in #strong[...] so they
    render bold -- mirroring the \\textbf{} highlighting in the LaTeX
    version this template is modeled on.

    Returns a string like: [Deployed a #strong[RESTful API] using #strong[FastAPI] ...]
    """
    spans: list[tuple[int, int]] = []

    for m in _BOLD_NUMBER_PATTERN.finditer(text):
        spans.append((m.start(), m.end()))

    for term in vocab:
        if len(term) < 2:
            continue
        # Word-boundary match only -- a plain substring search would wrongly
        # match e.g. vocab term "Zed" inside the unrelated word "quantized".
        try:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        except re.error:
            continue  # skip any term that produces an invalid regex (unlikely, but don't crash on it)
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))

    # Merge overlapping/adjacent spans so we don't produce nested/broken #strong[] calls
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    parts = []
    pos = 0
    for s, e in merged:
        if s > pos:
            parts.append(escape_typst(text[pos:s]))
        parts.append(f"#strong[{escape_typst(text[s:e])}]")
        pos = e
    if pos < len(text):
        parts.append(escape_typst(text[pos:]))

    return "[" + "".join(parts) + "]"


def render_education_entries(resume: dict, vocab: set[str]) -> str:
    blocks = []
    for edu in resume.get("education", []):
        institution = escape_typst(edu.get("institution", ""))
        credential = escape_typst(edu.get("credential", ""))
        details = edu.get("details", "")
        dates = escape_typst(f"{edu.get('start_date', '')} -- {edu.get('end_date', '')}")
        bullets_typst = "(" + render_bullet_content(details, vocab) + ",)" if details else "()"
        blocks.append(f'#entry("{institution}", "{credential}", "{dates}", "", {bullets_typst})')
    return "\n".join(blocks)


def render_selected_entries(resume: dict, sections: list[SectionSelection], kind: str, vocab: set[str]) -> str:
    """Render either project or experience entries based on the selection."""
    blocks = []
    source_list = resume.get("projects" if kind == "project" else "experience", [])
    by_name = {
        (item.get("name") if kind == "project" else item.get("company")): item
        for item in source_list
    }

    for section in sections:
        if section.kind != kind:
            continue
        item = by_name.get(section.name)
        if not item:
            continue

        bullets_typst = "(" + ", ".join(render_bullet_content(b.final_text, vocab) for b in section.bullets) + ",)"

        if kind == "project":
            title = escape_typst(item.get("name", ""))
            subtitle = escape_typst(", ".join(item.get("technologies", [])))
            dates = escape_typst(item.get("date", ""))
            location = ""
        else:
            title = escape_typst(item.get("position", ""))
            subtitle = escape_typst(item.get("company", ""))
            dates = escape_typst(f"{item.get('start_date', '')} -- {item.get('end_date', '')}")
            location = escape_typst(item.get("location", ""))

        blocks.append(f'#entry("{title}", "{subtitle}", "{dates}", "{location}", {bullets_typst})')

    return "\n".join(blocks)


def render_skills_block(resume: dict) -> str:
    lines = []
    label_map = {
        "languages": "Languages",
        "frameworks": "Frameworks",
        "developer_tools": "Developer Tools",
        "libraries_and_packages": "Libraries",
        "ml_ai_tools": "ML/AI Tools",
    }
    for key, label in label_map.items():
        items = resume.get("skills", {}).get(key, [])
        if items:
            escaped = escape_typst(", ".join(items))
            lines.append(f'#text(weight: "bold")[{label}:] {escaped}')
    # Blank line between entries creates a real paragraph break in Typst
    # (a trailing "\\\\" does NOT reliably render as a line break here).
    return "\n\n".join(lines)


def render_contact_line(personal: dict) -> str:
    parts = [personal.get("email", ""), personal.get("phone", ""), personal.get("linkedin", ""), personal.get("github", "")]
    parts = [escape_typst(p) for p in parts if p]
    return " | ".join(parts)


def render_typst_source(resume: dict, sections: list[SectionSelection]) -> str:
    template_file = Path(TEMPLATE_PATH)
    if not template_file.exists():
        raise ResumeBuilderError(f"Typst template not found at {template_file.resolve()}")

    template = template_file.read_text(encoding="utf-8")
    personal = resume.get("personal_info", {})

    filled = template
    filled = filled.replace("{{NAME}}", escape_typst(personal.get("name", "")))
    filled = filled.replace("{{CONTACT_LINE}}", render_contact_line(personal))
    vocab = build_vocabulary(resume)
    filled = filled.replace("{{EDUCATION_ENTRIES}}", render_education_entries(resume, vocab))
    filled = filled.replace("{{EXPERIENCE_ENTRIES}}", render_selected_entries(resume, sections, "experience", vocab))
    filled = filled.replace("{{PROJECT_ENTRIES}}", render_selected_entries(resume, sections, "project", vocab))
    filled = filled.replace("{{SKILLS_BLOCK}}", render_skills_block(resume))
    return filled


def compile_typst_to_pdf(typst_source: str, output_pdf_path: str) -> None:
    """Write the Typst source to a temp file and compile it to PDF via the Typst CLI."""
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    typ_path = output_path.with_suffix(".typ")

    typ_path.write_text(typst_source, encoding="utf-8")

    try:
        result = subprocess.run(
            [TYPST_BINARY, "compile", str(typ_path), str(output_path)],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as e:
        raise ResumeBuilderError(
            f"Typst binary '{TYPST_BINARY}' not found. Make sure Typst is installed and on your PATH."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ResumeBuilderError("Typst compilation timed out after 30 seconds") from e

    if result.returncode != 0:
        raise ResumeBuilderError(f"Typst compilation failed:\n{result.stderr}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def build_resume_for_job(job_id: int) -> str:
    """
    Build a tailored PDF resume for the given job. Returns the output PDF path.
    Raises ResumeBuilderError on failure.
    """
    if not GEMINI_API_KEY:
        raise ResumeBuilderError("GEMINI_API_KEY is not set in .env")

    db = Database()
    jobs = db.get_jobs_by_status("SCORED")
    job = next((j for j in jobs if j.id == job_id), None)
    if job is None:
        raise ResumeBuilderError(f"Job {job_id} not found or not in SCORED status")

    resume = load_master_resume()
    vocab = build_vocabulary(resume)
    client = genai.Client(api_key=GEMINI_API_KEY)

    logger.info("Selecting tailored content for job %d ('%s' @ %s)...", job.id, job.title, job.company)
    sections = select_content(client, resume, job, vocab)

    rewritten_count = sum(1 for s in sections for b in s.bullets if b.was_rewritten)
    rejected_count = sum(1 for s in sections for b in s.bullets if b.rejected_reason)
    logger.info(
        "Content selected: %d sections, %d bullets rephrased, %d rephrase(s) rejected by guardrail",
        len(sections), rewritten_count, rejected_count,
    )

    typst_source = render_typst_source(resume, sections)

    safe_company = re.sub(r"[^\w\-]", "_", job.company)[:40]
    safe_title = re.sub(r"[^\w\-]", "_", job.title)[:40]
    output_pdf_path = os.path.join(OUTPUT_DIR, f"{safe_company}_{safe_title}_{job.id}.pdf")

    logger.info("Compiling Typst -> PDF at %s...", output_pdf_path)
    compile_typst_to_pdf(typst_source, output_pdf_path)

    try:
        db.create_application(job.id, tailored_pdf_path=output_pdf_path)
        db.update_status(job.id, "TAILORED")
    except DatabaseError as e:
        logger.warning("PDF built successfully but failed to update database: %s", e)

    logger.info("Resume built successfully: %s", output_pdf_path)
    return output_pdf_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python resume_builder.py <job_id>")
        sys.exit(1)

    try:
        job_id_arg = int(sys.argv[1])
        path = build_resume_for_job(job_id_arg)
        print(f"Success: {path}")
    except (ResumeBuilderError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)