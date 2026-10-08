"""
github_scraper.py

Tier 1 (extension): GitHub Repo Ingestion for AutoApply AI

Pulls structured internship listings from community-maintained GitHub repos
that publish a machine-readable `listings.json` (the format used by
SimplifyJobs/PittCSC-style repos), rather than scraping their README tables.

Format reference (per SimplifyJobs CONTRIBUTING.md):
    {
        "company_name": str,
        "title": str,
        "url": str,
        "locations": [str, ...],
        "category": str,          # e.g. "Software Engineering", "Software"
        "active": bool,           # true if application is currently open
        "terms": [str, ...],      # e.g. ["Summer 2026"]
        "date_posted": int,       # unix timestamp
        "date_updated": int,
        ...
    }

Usage:
    python github_scraper.py
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

from database import Database, DatabaseError

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Raw listings.json URLs for repos that follow the Simplify/PittCSC schema.
# Add more repos here as new ones appear each internship cycle — the repo
# name changes yearly (Summer2027-Internships is the current cycle as of
# 2026; older ones get archived).
GITHUB_REPO_URLS = [
    url.strip()
    for url in os.getenv(
        "GITHUB_LISTINGS_URLS",
        "https://raw.githubusercontent.com/SimplifyJobs/Summer2027-Internships/dev/.github/scripts/listings.json,"
        "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/.github/scripts/listings.json",
    ).split(",")
    if url.strip()
]

# Category strings used across schema versions for SWE-relevant roles.
# Both "Software" and "Software Engineering" appear depending on when the
# entry was added, since the repo's category taxonomy changed over time.
SWE_CATEGORIES = {"Software", "Software Engineering"}

REQUEST_TIMEOUT_SECONDS = 30


def fetch_listings(url: str) -> list[dict]:
    """
    Download and parse a listings.json file from a given repo URL.

    Returns an empty list on any failure (network, non-200, malformed JSON)
    rather than raising, so one bad repo URL doesn't take down the others.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            logger.error("Unexpected listings.json shape at %s (expected a list)", url)
            return []
        logger.info("%s: %d total listings fetched", url, len(data))
        return data
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return []
    except ValueError as e:  # JSON decode error
        logger.error("Failed to parse JSON from %s: %s", url, e)
        return []


def filter_active_swe(listings: list[dict]) -> list[dict]:
    """Keep only postings that are currently open AND tagged as software/SWE roles."""
    filtered = [
        entry for entry in listings
        if entry.get("active") is True and entry.get("category") in SWE_CATEGORIES
    ]
    logger.info("Filtered to %d active SWE listings", len(filtered))
    return filtered


def store_listings(listings: list[dict], db: Database, source_label: str) -> tuple[int, int]:
    """
    Insert filtered GitHub listings into the database, skipping duplicates.

    Returns:
        (new_count, duplicate_count)
    """
    new_count = 0
    duplicate_count = 0
    error_count = 0

    for entry in listings:
        try:
            title = _clean(entry.get("title"))
            company = _clean(entry.get("company_name"))
            apply_url = _clean(entry.get("url"))
            locations = entry.get("locations") or []
            location = ", ".join(locations) if locations else None

            if not title or not company or not apply_url:
                logger.warning("Skipping listing with missing required fields: %s", entry.get("id"))
                error_count += 1
                continue

            inserted = db.insert_job(
                title=title,
                company=company,
                apply_url=apply_url,
                location=location,
                description=None,  # GitHub listings don't include full JDs; evaluator can fetch if needed
                ats_platform=_detect_ats_platform(apply_url),
                source=source_label,
            )
            if inserted:
                new_count += 1
            else:
                duplicate_count += 1

        except DatabaseError as e:
            logger.error("Failed to store listing '%s': %s", entry.get("id"), e)
            error_count += 1
        except Exception as e:
            logger.error("Unexpected error processing listing: %s", e, exc_info=True)
            error_count += 1

    if error_count:
        logger.warning("%d listings failed to process from %s", error_count, source_label)

    return new_count, duplicate_count


def _detect_ats_platform(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    hints = {
        "greenhouse.io": "greenhouse",
        "lever.co": "lever",
        "myworkdayjobs.com": "workday",
    }
    for domain, platform in hints.items():
        if domain in url:
            return platform
    return "other"


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def run() -> None:
    """Entry point: fetch all configured repos, filter, dedupe, store, report."""
    db = Database()
    db.init_schema()

    total_new = 0
    total_dup = 0

    for url in GITHUB_REPO_URLS:
        # Use the repo name as the source label (e.g. "Summer2027-Internships")
        source_label = url.split("/")[4] if len(url.split("/")) > 4 else "github_repo"

        raw_listings = fetch_listings(url)
        if not raw_listings:
            continue

        active_swe = filter_active_swe(raw_listings)
        new_count, dup_count = store_listings(active_swe, db, source_label)
        total_new += new_count
        total_dup += dup_count
        logger.info("%s: %d new, %d duplicates", source_label, new_count, dup_count)

    logger.info("GitHub scrape complete: %d new jobs stored, %d duplicates skipped", total_new, total_dup)


if __name__ == "__main__":
    run()