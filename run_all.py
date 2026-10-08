"""
run_all.py

The single daily command for AutoApply AI (Phase 1 + Phase 2):

    1. scraper.py         -> JobSpy (LinkedIn, Indeed, Glassdoor, ZipRecruiter)
    2. github_scraper.py  -> Simplify/PittCSC-style GitHub listings.json repos
    3. evaluator.py       -> Gemini fit-scoring on everything newly discovered

Each stage is isolated: if one throws an unexpected error, the next stage
still runs. A clear summary is printed at the end so you can see at a glance
what happened without scrolling through the full log.

Usage:
    python run_all.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    logger.info("=" * 60)
    logger.info("AutoApply AI -- daily run starting")
    logger.info("=" * 60)

    summary = {
        "scraper": "not run",
        "github_scraper": "not run",
        "evaluator": "not run",
    }

    logger.info("--- Stage 1/3: Job board scrape (JobSpy) ---")
    try:
        import scraper
        scraper.run()
        summary["scraper"] = "OK"
    except Exception as e:
        logger.error("scraper.py failed unexpectedly: %s", e, exc_info=True)
        summary["scraper"] = f"FAILED: {e}"

    logger.info("--- Stage 2/3: GitHub repo scrape (Simplify/PittCSC listings) ---")
    try:
        import github_scraper
        github_scraper.run()
        summary["github_scraper"] = "OK"
    except Exception as e:
        logger.error("github_scraper.py failed unexpectedly: %s", e, exc_info=True)
        summary["github_scraper"] = f"FAILED: {e}"

    logger.info("--- Stage 3/3: Gemini fit scoring ---")
    try:
        import evaluator
        evaluator.run()
        summary["evaluator"] = "OK"
    except Exception as e:
        logger.error("evaluator.py failed unexpectedly: %s", e, exc_info=True)
        summary["evaluator"] = f"FAILED: {e}"

    logger.info("=" * 60)
    logger.info("Daily run complete. Summary:")
    for stage, result in summary.items():
        logger.info("  %-16s %s", stage, result)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()