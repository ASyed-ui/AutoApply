"""
evaluator.py

Tier 2: AI Fit Scoring Engine (AutoApply AI)

For every job in the DB with status='DISCOVERED':
  1. If it has no stored description (e.g. GitHub-sourced listings only carry
     title/company/location/url), fetch the actual job posting page and
     extract readable text from it.
  2. Send the job info + a condensed version of master_resume.json to Gemini,
     asking for a 0-10 fit score and short reasoning.
  3. Store the score/reasoning. Jobs scoring >= FIT_SCORE_THRESHOLD move to
     status 'SCORED' (eligible for resume tailoring); everything else is
     marked 'SKIPPED' so it never reaches Tier 3.

Usage:
    python evaluator.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

from database import Database, DatabaseError, JobRecord

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FIT_SCORE_THRESHOLD = float(os.getenv("FIT_SCORE_THRESHOLD", "7.5"))
MASTER_RESUME_PATH = os.getenv("MASTER_RESUME_PATH", "./master_resume.json")

PAGE_FETCH_TIMEOUT_SECONDS = 15
MAX_DESCRIPTION_CHARS = 6000  # keep prompts small/cheap; full JDs rarely need more than this
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2

# Proactive delay between every Gemini call (success or fail), to stay under
# per-minute rate limits rather than only reacting after a 429. Free-tier
# lightweight models are often capped at 10-15 requests/minute, so ~5s
# between calls keeps a steady pace without tripping the limiter.
REQUEST_DELAY_SECONDS = float(os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "5"))

_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)")


def _extract_retry_delay(error_text: str) -> Optional[float]:
    """
    Google's 429 responses often include an explicit retryDelay (e.g. '23s')
    telling you exactly how long to wait. Honor that instead of guessing with
    a fixed backoff -- guessing too short just repeats the same failure.
    """
    match = _RETRY_DELAY_PATTERN.search(error_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


# Safety cap: stop after this many jobs in a single run, regardless of how
# many are left unscored. Keeps a run from silently burning through an
# entire daily API quota in one go -- rerun the script again once quota
# resets (daily, midnight Pacific Time) or on your next session.
MAX_JOBS_PER_RUN = int(os.getenv("EVALUATOR_MAX_JOBS_PER_RUN", "300"))

REQUEST_HEADERS = {
    # Many ATS/career pages block requests with no browser-like UA
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


class EvaluatorError(Exception):
    """Raised for evaluator-level failures the caller should handle explicitly."""


class DailyQuotaExhausted(Exception):
    """
    Raised when Gemini reports a PER-DAY quota violation (as opposed to a
    per-minute rate limit, which is worth retrying/backing off on). There is
    no point retrying or continuing to the next job -- every subsequent call
    will fail identically until the quota resets, so the whole run should
    stop immediately rather than spend minutes cycling through failures.
    """


# ----------------------------------------------------------------------
# Resume profile loading
# ----------------------------------------------------------------------

def load_resume_profile(path: str = MASTER_RESUME_PATH) -> str:
    """
    Condense master_resume.json into a compact text block for the fit-scoring
    prompt. We don't need every field -- just enough for the LLM to judge
    tech-stack and experience-level match.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            resume = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise EvaluatorError(f"Failed to load master resume at {path}: {e}") from e

    skills = resume.get("skills", {})
    all_skills = []
    for category, items in skills.items():
        if isinstance(items, list):
            all_skills.extend(items)

    experience_lines = []
    for exp in resume.get("experience", []):
        experience_lines.append(
            f"- {exp.get('position')} at {exp.get('company')} "
            f"({exp.get('start_date')}–{exp.get('end_date')})"
        )

    project_lines = []
    for proj in resume.get("projects", []):
        techs = ", ".join(proj.get("technologies", []))
        project_lines.append(f"- {proj.get('name')} ({techs})")

    education_lines = []
    for edu in resume.get("education", []):
        education_lines.append(f"- {edu.get('credential')} at {edu.get('institution')}")

    profile = (
        f"Skills: {', '.join(all_skills)}\n\n"
        f"Experience:\n" + "\n".join(experience_lines) + "\n\n"
        f"Projects:\n" + "\n".join(project_lines) + "\n\n"
        f"Education:\n" + "\n".join(education_lines)
    )
    return profile


# ----------------------------------------------------------------------
# Job description fetching (fallback for GitHub-sourced listings)
# ----------------------------------------------------------------------

def fetch_job_description(url: str) -> Optional[str]:
    """
    Best-effort fetch + extraction of a job posting's text content.

    Returns None if the fetch fails, times out, or yields unusably little
    text (e.g. a JS-rendered SPA that returns an empty shell). Callers should
    fall back to scoring off title/company/location alone in that case.
    """
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=PAGE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch job page %s: %s", url, e)
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())  # collapse whitespace
    except Exception as e:
        logger.warning("Failed to parse HTML from %s: %s", url, e)
        return None

    if len(text) < 200:
        # Almost certainly a JS-rendered page that returned an empty shell
        logger.info("Extracted text too short (%d chars) from %s, treating as unavailable", len(text), url)
        return None

    return text[:MAX_DESCRIPTION_CHARS]


# ----------------------------------------------------------------------
# Gemini fit scoring
# ----------------------------------------------------------------------

FIT_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "fit_score": {
            "type": "number",
            "description": "Fit score from 0 to 10, can include one decimal place",
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentence explanation of the score",
        },
    },
    "required": ["fit_score", "reasoning"],
}

SYSTEM_PROMPT = """You are an expert technical recruiter helping a candidate evaluate \
whether a job posting is a strong match for their background. You will be given the \
candidate's resume profile and a job posting. Score the fit from 0 to 10 based on:
- Overlap between the candidate's tech stack/skills and the job's requirements
- Whether the seniority level (internship/co-op/entry-level/experienced) matches the \
candidate's current stage (student, seeking internships/co-ops)
- Whether the role is actually software-engineering-related (penalize heavily if the \
role is clearly non-technical, e.g. finance, sales, hardware-only, or unrelated fields, \
even if it superficially mentions "engineer" or "technical")

Be strict: a 7.5+ score should be reserved for roles that are a clear, strong match. \
Respond ONLY with the requested JSON structure, nothing else."""


def build_job_prompt(resume_profile: str, job: JobRecord, description: Optional[str]) -> str:
    description_block = description or "(No detailed description available -- score based on title/company/location only, and be more conservative in that case.)"
    return (
        f"CANDIDATE RESUME PROFILE:\n{resume_profile}\n\n"
        f"---\n\n"
        f"JOB POSTING:\n"
        f"Title: {job.title}\n"
        f"Company: {job.company}\n"
        f"Location: {job.location or 'Not specified'}\n"
        f"Description:\n{description_block}"
    )


def score_job_fit(client: genai.Client, resume_profile: str, job: JobRecord, description: Optional[str]) -> tuple[float, str]:
    """
    Call Gemini to score fit for a single job. Retries on transient failures.

    Returns:
        (fit_score, reasoning)

    Raises:
        EvaluatorError if all retries are exhausted.
    """
    prompt = build_job_prompt(resume_profile, job, description)

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=FIT_SCORE_SCHEMA,
                    temperature=0.2,
                ),
            )
            result = json.loads(response.text)
            fit_score = float(result["fit_score"])
            reasoning = str(result["reasoning"]).strip()
            fit_score = max(0.0, min(10.0, fit_score))  # clamp to valid range
            time.sleep(REQUEST_DELAY_SECONDS)  # proactive throttle to stay under per-minute limits
            return fit_score, reasoning

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # Malformed response from the model -- retrying rarely helps here,
            # but we try once more in case it was a one-off formatting slip.
            last_error = e
            logger.warning("Attempt %d: malformed Gemini response for job %d: %s", attempt, job.id, e)
        except Exception as e:
            error_text = str(e)
            if "PerDay" in error_text or "RequestsPerDay" in error_text:
                # Daily quota exhausted -- every remaining job will fail the
                # same way until it resets. Stop immediately rather than
                # retrying this job or moving to the next one.
                raise DailyQuotaExhausted(
                    "Gemini's daily request quota is exhausted for this API key/model. "
                    "It resets at midnight Pacific Time. Either wait for reset, switch "
                    "GEMINI_MODEL to a higher-quota model, or reduce EVALUATOR_MAX_JOBS_PER_RUN in .env."
                ) from e
            # Otherwise: transient network/per-minute-rate-limit/server error -- worth backing off and retrying.
            last_error = e
            logger.warning("Attempt %d: Gemini call failed for job %d: %s", attempt, job.id, e)

            # Honor Google's own suggested wait time for per-minute rate limits
            # rather than guessing with a fixed backoff.
            suggested_delay = _extract_retry_delay(error_text)
            if suggested_delay is not None:
                logger.info("Waiting %.1fs as instructed by Gemini's retryDelay before retrying...", suggested_delay)
                time.sleep(suggested_delay + 0.5)  # small buffer past the exact boundary
                continue  # skip the generic backoff below, we already waited the right amount

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise EvaluatorError(f"Failed to score job {job.id} after {MAX_RETRIES + 1} attempts: {last_error}")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def run() -> None:
    if not GEMINI_API_KEY:
        raise EvaluatorError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(get one at https://aistudio.google.com/apikey)"
        )

    db = Database()
    db.init_schema()

    resume_profile = load_resume_profile()
    client = genai.Client(api_key=GEMINI_API_KEY)

    jobs = db.get_unscored_jobs()
    logger.info("Found %d unscored jobs", len(jobs))

    if len(jobs) > MAX_JOBS_PER_RUN:
        logger.info(
            "Capping this run to %d jobs (out of %d unscored). Rerun the script "
            "again later to continue with the rest -- already-scored jobs are "
            "never reprocessed.",
            MAX_JOBS_PER_RUN, len(jobs),
        )
        jobs = jobs[:MAX_JOBS_PER_RUN]

    scored_count = 0
    skipped_count = 0
    error_count = 0

    for job in jobs:
        try:
            description = job.description
            if not description:
                logger.info("Job %d ('%s' @ %s) has no description, fetching page...", job.id, job.title, job.company)
                description = fetch_job_description(job.apply_url)

            fit_score, reasoning = score_job_fit(client, resume_profile, job, description)
            db.update_fit_score(job.id, fit_score, reasoning)

            if fit_score >= FIT_SCORE_THRESHOLD:
                scored_count += 1
                logger.info("SCORED %.1f - %s @ %s", fit_score, job.title, job.company)
            else:
                db.update_status(job.id, "SKIPPED")
                skipped_count += 1
                logger.info("SKIPPED %.1f - %s @ %s", fit_score, job.title, job.company)

        except DailyQuotaExhausted as e:
            logger.error("STOPPING RUN: %s", e)
            logger.info(
                "Progress so far this run: %d scored, %d skipped, %d errors. "
                "Remaining unscored jobs are untouched and will be picked up "
                "on the next run.",
                scored_count, skipped_count, error_count,
            )
            break
        except EvaluatorError as e:
            logger.error("Failed to evaluate job %d: %s", job.id, e)
            error_count += 1
        except DatabaseError as e:
            logger.error("Database error for job %d: %s", job.id, e)
            error_count += 1
        except Exception as e:
            logger.error("Unexpected error evaluating job %d: %s", job.id, e, exc_info=True)
            error_count += 1

    logger.info(
        "Evaluation complete: %d scored >=%.1f, %d skipped, %d errors (out of %d total)",
        scored_count, FIT_SCORE_THRESHOLD, skipped_count, error_count, len(jobs),
    )


if __name__ == "__main__":
    run()