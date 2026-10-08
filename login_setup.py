"""
login_setup.py

One-time setup: opens a real browser window for you to manually log into
LinkedIn (and any other site you want persisted, like a company's own
career portal). Once you log in and press Enter back in the terminal, your
session (cookies, local storage) is saved to disk.

browser_bot.py then loads this saved session on every run, so it opens
already logged in instead of hitting a sign-in wall on LinkedIn-native
"Apply" flows.

Run this again any time your session expires (LinkedIn logs you out
periodically) or you want to add another site's login.

Usage:
    python login_setup.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SESSION_STATE_PATH = os.getenv("BROWSER_SESSION_STATE_PATH", "./browser_session.json")
START_URL = "https://www.linkedin.com/login"


def run() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])

        # Load any existing session so you don't have to log in again if it's still valid
        existing_state = SESSION_STATE_PATH if Path(SESSION_STATE_PATH).exists() else None
        if existing_state:
            logger.info("Found existing session at %s -- loading it first", SESSION_STATE_PATH)
        context = browser.new_context(no_viewport=True, storage_state=existing_state)
        page = context.new_page()
        page.goto(START_URL)

        print("\n" + "=" * 70)
        print("  LOGIN SETUP")
        print("=" * 70)
        print("  A browser window has opened to LinkedIn's login page.")
        print("  1. Log in normally (solve any CAPTCHA/2FA as needed)")
        print("  2. Once you're fully logged in and see your feed, come back here")
        print("  3. You can also navigate to any other site (e.g. a company career")
        print("     portal) and log in there too -- all sessions in this browser")
        print("     window will be saved together")
        print("=" * 70)
        input("\nPress Enter once you're logged in (and ready to save the session): ")

        context.storage_state(path=SESSION_STATE_PATH)
        logger.info("Session saved to %s", SESSION_STATE_PATH)
        browser.close()

    print(f"\nDone. browser_bot.py will now reuse this session automatically.")
    print(f"Re-run this script any time your login expires.")


if __name__ == "__main__":
    run()