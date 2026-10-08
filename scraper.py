"""
scraper.py

Tier 1: Job Ingestion & Discovery Engine (AutoApply AI)

Uses python-jobspy to scrape LinkedIn, Indeed, Glassdoor, and ZipRecruiter
for SWE internship postings, then inserts deduplicated results into the
local SQLite database via database.py.

Usage:
    python scraper.py

Configuration is pulled from .env (search terms, location, results per site).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from jobspy import scrape_jobs

from database import Database, DatabaseError

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------------------------------------
# Configuration (override via .env if desired)
# ----------------------------------------------------------------------
SEARCH_TERM = os.getenv("JOB_SEARCH_TERM", "software engineering intern")
LOCATION = os.getenv("JOB_LOCATION", "Toronto, ON, Canada")
RESULTS_PER_SITE = int(os.getenv("JOB_RESULTS_PER_SITE", "40"))
HOURS_OLD = int(os.getenv("JOB_HOURS_OLD", "72"))  # only fetch postings from the last N hours
SITES = ["linkedin", "indeed", "glassdoor", "zip_recruiter"]

# Map JobSpy's job_url domains to our ats_platform categories where detectable.
ATS_HINTS = {
    "greenhouse.io": "greenhouse",
    "lever.co": "lever",
    "myworkdayjobs.com": "workday",
}


def detect_ats_platform(url: Optional[str]) -> Optional[str]:
    """Best-effort detection of which ATS a job application URL belongs to."""
    if not url:
        return None
    for domain, platform in ATS_HINTS.items():
        if domain in url:
            return platform
    return "other"


def scrape_all_sites() -> pd.DataFrame:
    """
    Run JobSpy across all configured sites for the configured search term/location.

    Each site is scraped independently so that one site failing (rate limit,
    layout change, network blip) doesn't discard results already fetched from
    the others. Results from all successful sites are concatenated.

    Returns:
        A combined pandas DataFrame of raw job postings. Empty DataFrame if
        every site failed.
    """
    logger.info(
        "Scraping sites=%s term='%s' location='%s' results_per_site=%d hours_old=%d",
        SITES, SEARCH_TERM, LOCATION, RESULTS_PER_SITE, HOURS_OLD,
    )

    frames: list[pd.DataFrame] = []
    for site in SITES:
        try:
            site_df = scrape_jobs(
                site_name=[site],
                search_term=SEARCH_TERM,
                location=LOCATION,
                results_wanted=RESULTS_PER_SITE,
                hours_old=HOURS_OLD,
                country_indeed="Canada",
            )
            logger.info("%s: %d postings", site, len(site_df))
            if not site_df.empty:
                frames.append(site_df)
        except Exception as e:
            # Each site is isolated: one failing (rate limit, blocked host,
            # layout change) must not discard results already gathered from others.
            logger.error("%s scrape failed, skipping this site: %s", site, e)

    if not frames:
        logger.warning("All sites failed or returned no results this run.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined total: %d raw postings across %d successful site(s)", len(combined), len(frames))
    return combined


def store_jobs(jobs_df: pd.DataFrame, db: Database) -> tuple[int, int]:
    """
    Insert scraped jobs into the database, skipping duplicates.

    Returns:
        (new_count, duplicate_count)
    """
    new_count = 0
    duplicate_count = 0
    error_count = 0

    for _, row in jobs_df.iterrows():
        try:
            title = _clean(row.get("title"))
            company = _clean(row.get("company"))
            # Prefer the direct employer application URL when JobSpy provides one
            # (job_url_direct) -- job_url is often just the LinkedIn/Indeed LISTING
            # page, which has no application form to fill. Falling back to job_url
            # means browser_bot.py will just show the listing page in that case.
            apply_url = _clean(row.get("job_url_direct")) or _clean(row.get("job_url"))
            location = _clean(row.get("location"))
            description = _clean(row.get("description"))
            source = _clean(row.get("site"))

            if not title or not company or not apply_url:
                logger.warning("Skipping row with missing required fields: %s", row.to_dict())
                error_count += 1
                continue

            ats_platform = detect_ats_platform(apply_url)

            inserted = db.insert_job(
                title=title,
                company=company,
                apply_url=apply_url,
                location=location,
                description=description,
                ats_platform=ats_platform,
                source=source,
            )
            if inserted:
                new_count += 1
            else:
                duplicate_count += 1

        except DatabaseError as e:
            logger.error("Failed to store job row: %s", e)
            error_count += 1
        except Exception as e:
            logger.error("Unexpected error processing job row: %s", e, exc_info=True)
            error_count += 1

    if error_count:
        logger.warning("%d rows failed to process during this run", error_count)

    return new_count, duplicate_count


def _clean(value) -> Optional[str]:
    """Normalize pandas NaN/None to Python None, strip strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def run() -> None:
    """Entry point: scrape all sites, dedupe, store, report summary."""
    db = Database()
    db.init_schema()

    jobs_df = scrape_all_sites()
    if jobs_df.empty:
        logger.info("No postings retrieved this run (site failure or no matches). Exiting.")
        return

    new_count, duplicate_count = store_jobs(jobs_df, db)
    logger.info(
        "Scrape complete: %d new jobs stored, %d duplicates skipped, %d total processed",
        new_count, duplicate_count, len(jobs_df),
    )


if __name__ == "__main__":
    run()