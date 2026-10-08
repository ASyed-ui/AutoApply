"""
browser_bot.py

Tier 3: Browser Automation & Human-in-the-Loop Execution (AutoApply AI)

For a given job (must be status='TAILORED', i.e. resume_builder.py already ran):
  1. Launches a HEADFUL (visible) Chromium browser via Playwright.
  2. Navigates to the job's apply_url.
  3. Detects the ATS platform (greenhouse/lever/workday/other) and runs the
     matching best-effort autofill adapter: name, email, phone, resume upload.
  4. NEVER clicks Submit. The script pauses and hands control to you --
     you review the filled form, solve any CAPTCHA, fix anything the
     autofill got wrong, and click Submit yourself.
  5. Only after you confirm in the terminal that you submitted does the
     script mark the job as APPLIED in the database.

This script will NEVER auto-submit an application under any circumstances --
that boundary is intentional per the project's human-in-the-loop design.

Usage:
    python browser_bot.py <job_id>
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

from database import Database, DatabaseError, JobRecord
from resume_builder import load_master_resume

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FIELD_TIMEOUT_MS = 5000  # per-field lookup timeout -- fields that don't exist on a given form are skipped, not fatal
NAV_TIMEOUT_MS = 30000
SESSION_STATE_PATH = os.getenv("BROWSER_SESSION_STATE_PATH", "./browser_session.json")

# Domains that are job LISTING pages, not application forms. JobSpy sometimes
# only has a listing URL (no direct employer application link available) --
# in that case there's nothing to autofill, and the person needs to click
# "Apply" themselves to get redirected to the real form.
LISTING_PAGE_DOMAINS = [
    "linkedin.com/jobs/view",
    "indeed.com/viewjob",
    "indeed.com/rc/clk",
    "glassdoor.com/job-listing",
    "ziprecruiter.com/jobs",
]


def is_listing_page(url: str) -> bool:
    return any(domain in url for domain in LISTING_PAGE_DOMAINS)


class BrowserBotError(Exception):
    """Raised for browser-automation-level failures the caller should handle explicitly."""


# ----------------------------------------------------------------------
# Best-effort field fillers
# ----------------------------------------------------------------------
# Every fill_* call is wrapped so a missing/renamed field on a given
# employer's form never crashes the whole run -- it just gets logged and
# skipped, leaving that field for the human to fill in during review.

def try_fill(page: Page, selectors: list[str], value: str, field_name: str) -> bool:
    """Try a list of candidate selectors in order; fill the first one that exists."""
    if not value:
        return False
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.fill(value, timeout=FIELD_TIMEOUT_MS)
                logger.info("Filled '%s' via selector: %s", field_name, selector)
                return True
        except PlaywrightTimeoutError:
            continue
        except Exception as e:
            logger.debug("Selector %s failed for %s: %s", selector, field_name, e)
            continue
    logger.warning("Could not find a field for '%s' -- you'll need to fill this in manually", field_name)
    return False


def try_upload(page: Page, selectors: list[str], file_path: str, field_name: str) -> bool:
    if not file_path or not Path(file_path).exists():
        logger.warning("Resume file not found at %s -- skipping upload", file_path)
        return False
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.set_input_files(file_path, timeout=FIELD_TIMEOUT_MS)
                logger.info("Uploaded '%s' via selector: %s", field_name, selector)
                return True
        except Exception as e:
            logger.debug("Upload selector %s failed for %s: %s", selector, field_name, e)
            continue
    logger.warning("Could not find an upload field for '%s' -- you'll need to attach this manually", field_name)
    return False


# ----------------------------------------------------------------------
# ATS-specific adapters
# ----------------------------------------------------------------------

def fill_greenhouse(page: Page, personal: dict, resume_pdf_path: Optional[str]) -> None:
    """
    Greenhouse forms (boards.greenhouse.io) use fairly consistent field
    naming across employers: input[name="job_application[first_name]"] etc.
    """
    name = personal.get("name", "")
    first_name, _, last_name = name.partition(" ")

    try_fill(page, [
        'input[name="job_application[first_name]"]', '#first_name', 'input[autocomplete="given-name"]',
    ], first_name, "first name")

    try_fill(page, [
        'input[name="job_application[last_name]"]', '#last_name', 'input[autocomplete="family-name"]',
    ], last_name, "last name")

    try_fill(page, [
        'input[name="job_application[email]"]', '#email', 'input[type="email"]',
    ], personal.get("email", ""), "email")

    try_fill(page, [
        'input[name="job_application[phone]"]', '#phone', 'input[type="tel"]',
    ], personal.get("phone", ""), "phone")

    try_fill(page, [
        'input[name="job_application[urls][0]"]', 'input[id*="linkedin" i]',
    ], personal.get("linkedin", ""), "LinkedIn")

    if resume_pdf_path:
        try_upload(page, [
            'input[name="job_application[resume]"]', '#resume', 'input[type="file"]',
        ], resume_pdf_path, "resume")


def fill_lever(page: Page, personal: dict, resume_pdf_path: Optional[str]) -> None:
    """Lever forms (jobs.lever.co) typically use a single 'name' field, not split first/last."""
    try_fill(page, [
        'input[name="name"]', '#name-input',
    ], personal.get("name", ""), "full name")

    try_fill(page, [
        'input[name="email"]', '#email-input', 'input[type="email"]',
    ], personal.get("email", ""), "email")

    try_fill(page, [
        'input[name="phone"]', '#phone-input', 'input[type="tel"]',
    ], personal.get("phone", ""), "phone")

    try_fill(page, [
        'input[name="urls[LinkedIn]"]',
    ], personal.get("linkedin", ""), "LinkedIn")

    if resume_pdf_path:
        try_upload(page, [
            'input[name="resume"]', 'input[type="file"]',
        ], resume_pdf_path, "resume")


def fill_workday(page: Page, personal: dict, resume_pdf_path: Optional[str]) -> None:
    """
    Workday (*.myworkdayjobs.com) is intentionally NOT auto-filled.

    Workday forms require account creation, multi-step wizards, and often
    email verification codes -- meaningfully more complex than a single-page
    form and prone to silently breaking employer-to-employer. Rather than
    risk a half-broken autofill on a platform with this much variance, this
    adapter just navigates there and hands off to you immediately.
    """
    logger.info(
        "Workday detected -- autofill is not attempted here (Workday's multi-step "
        "account-creation flow varies too much between employers to fill reliably). "
        "The browser has navigated to the posting; please fill this one manually."
    )


def fill_generic(page: Page, personal: dict, resume_pdf_path: Optional[str]) -> None:
    """Best-effort fallback for unrecognized ATS platforms using common field patterns."""
    name = personal.get("name", "")
    first_name, _, last_name = name.partition(" ")

    try_fill(page, ['input[name*="first" i]', 'input[id*="first" i]'], first_name, "first name")
    try_fill(page, ['input[name*="last" i]', 'input[id*="last" i]'], last_name, "last name")
    try_fill(page, ['input[type="email"]', 'input[name*="email" i]'], personal.get("email", ""), "email")
    try_fill(page, ['input[type="tel"]', 'input[name*="phone" i]'], personal.get("phone", ""), "phone")

    if resume_pdf_path:
        try_upload(page, ['input[type="file"]'], resume_pdf_path, "resume")


ADAPTERS = {
    "greenhouse": fill_greenhouse,
    "lever": fill_lever,
    "workday": fill_workday,
}


# ----------------------------------------------------------------------
# Human-in-the-loop pause
# ----------------------------------------------------------------------

def play_alert_sound() -> None:
    """Best-effort system chime so the person notices the form is ready for review."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.MessageBeep()
        else:
            print("\a", end="", flush=True)  # ASCII bell, works in most terminals
    except Exception:
        pass  # Sound is a nice-to-have; never let it block the actual workflow


def wait_for_human_review(job: JobRecord) -> bool:
    """
    Pause execution and hand off to the person. Returns True if they confirm
    they submitted the application, False if they chose to skip it.
    """
    play_alert_sound()
    print("\n" + "=" * 70)
    print(f"  READY FOR REVIEW: {job.title} @ {job.company}")
    print("=" * 70)
    print("  The form has been auto-filled as best as possible. Please:")
    print("  1. Review every field in the browser window")
    print("  2. Fix anything the autofill missed or got wrong")
    print("  3. Solve any CAPTCHA if present")
    print("  4. Click Submit yourself -- this script will NEVER submit for you")
    print("=" * 70)

    while True:
        response = input("\nDid you submit the application? [y/n/skip]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no", "skip"):
            return False
        print("Please answer 'y' (submitted), 'n' (did not submit), or 'skip'.")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def run_application(job_id: int) -> None:
    db = Database()
    jobs = db.get_jobs_by_status("TAILORED")
    job = next((j for j in jobs if j.id == job_id), None)
    if job is None:
        raise BrowserBotError(f"Job {job_id} not found or not in TAILORED status (run resume_builder.py first)")

    resume = load_master_resume()
    personal = resume.get("personal_info", {})

    # Look up the tailored PDF path for this job from the applications table
    tailored_pdf_path = _get_tailored_pdf_path(db, job_id)

    adapter = ADAPTERS.get(job.ats_platform, fill_generic)
    logger.info(
        "Opening browser for job %d ('%s' @ %s) -- detected ATS: %s",
        job.id, job.title, job.company, job.ats_platform or "unknown (using generic fallback)",
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])

        # Reuse a saved login session if one exists (set up via login_setup.py),
        # so LinkedIn-native "Apply" flows don't hit a sign-in wall every run.
        session_state = SESSION_STATE_PATH if Path(SESSION_STATE_PATH).exists() else None
        if session_state:
            logger.info("Loading saved browser session from %s", SESSION_STATE_PATH)
        else:
            logger.info(
                "No saved session found at %s -- LinkedIn-native apply flows may require "
                "signing in manually. Run 'python login_setup.py' once to avoid this.",
                SESSION_STATE_PATH,
            )
        context = browser.new_context(no_viewport=True, storage_state=session_state)
        page = context.new_page()

        try:
            page.goto(job.apply_url, timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError as e:
            browser.close()
            raise BrowserBotError(f"Failed to load {job.apply_url}: page took too long to respond") from e

        if is_listing_page(page.url):
            logger.warning(
                "This looks like a job LISTING page, not an application form -- "
                "there's nothing here to autofill. Click 'Apply' on the page yourself; "
                "if it redirects to the employer's real application form, you can fill "
                "that manually, or copy the resulting URL and update apply_url in the "
                "database to re-run autofill against it next time."
            )
        else:
            try:
                adapter(page, personal, tailored_pdf_path)
            except Exception as e:
                # Autofill problems should never prevent the person from reviewing
                # and manually completing the application themselves.
                logger.warning("Autofill encountered an error (form still open for manual completion): %s", e)

        submitted = wait_for_human_review(job)
        browser.close()

    if submitted:
        try:
            db.update_status(job.id, "APPLIED")
            logger.info("Job %d marked as APPLIED", job.id)
        except DatabaseError as e:
            logger.error("Failed to update job status: %s", e)
    else:
        logger.info("Job %d left as TAILORED -- you can run browser_bot.py again later to retry", job.id)


def _get_tailored_pdf_path(db: Database, job_id: int) -> Optional[str]:
    """Look up the most recent tailored_pdf_path for this job from the applications table."""
    try:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT tailored_pdf_path FROM applications WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            return row["tailored_pdf_path"] if row else None
    except Exception as e:
        logger.warning("Could not look up tailored resume path: %s", e)
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python browser_bot.py <job_id>")
        sys.exit(1)

    try:
        run_application(int(sys.argv[1]))
    except (BrowserBotError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)