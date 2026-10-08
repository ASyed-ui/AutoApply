"""
database.py

SQLite persistence layer for AutoApply AI.

Handles:
- Schema initialization
- Job insertion with deduplication (via hash of company + title + location)
- Fit-score updates
- Status transitions
- Basic query helpers used by the Streamlit dashboard later

Usage:
    from database import Database

    db = Database()
    db.init_schema()
    was_new = db.insert_job(
        title="Software Engineering Intern",
        company="Acme Corp",
        location="Toronto, ON",
        apply_url="https://boards.greenhouse.io/acme/jobs/123",
        description="...",
        source="linkedin",
    )
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_DB_PATH = "autoapply.db"
SCHEMA_PATH = "schema.sql"

VALID_STATUSES = {"DISCOVERED", "SCORED", "TAILORED", "APPLIED", "SKIPPED", "REJECTED"}


class DatabaseError(Exception):
    """Raised for any database-layer failure that callers should handle explicitly."""


@dataclass
class JobRecord:
    id: int
    job_hash: str
    title: str
    company: str
    location: Optional[str]
    apply_url: str
    description: Optional[str]
    ats_platform: Optional[str]
    source: Optional[str]
    fit_score: Optional[float]
    fit_reasoning: Optional[str]
    date_discovered: str
    date_updated: str
    status: str


def make_job_hash(company: str, title: str, location: str | None) -> str:
    """
    Build a stable dedup hash from company + title + location.

    Normalizes case and whitespace so trivially different scrapes of the
    same posting (e.g. differing capitalization) still collide correctly.
    """
    normalized = f"{company.strip().lower()}|{title.strip().lower()}|{(location or '').strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Database:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, schema_path: str = SCHEMA_PATH):
        self.db_path = db_path
        self.schema_path = schema_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context-managed connection with row factory and FK enforcement on."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables/indexes if they don't already exist. Safe to call repeatedly."""
        schema_file = Path(self.schema_path)
        if not schema_file.exists():
            raise DatabaseError(f"Schema file not found at {schema_file.resolve()}")

        sql = schema_file.read_text(encoding="utf-8")
        try:
            with self._connect() as conn:
                conn.executescript(sql)
            logger.info("Database schema initialized at %s", self.db_path)
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize schema: {e}") from e

    # ------------------------------------------------------------------
    # Job operations
    # ------------------------------------------------------------------

    def insert_job(
        self,
        title: str,
        company: str,
        apply_url: str,
        location: Optional[str] = None,
        description: Optional[str] = None,
        ats_platform: Optional[str] = None,
        source: Optional[str] = None,
    ) -> bool:
        """
        Insert a job if it isn't already present (deduplicated by hash).

        Returns:
            True if a new row was inserted, False if it was a duplicate.
        """
        if not title or not company or not apply_url:
            raise DatabaseError("title, company, and apply_url are required to insert a job")

        job_hash = make_job_hash(company, title, location)

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                        (job_hash, title, company, location, apply_url,
                         description, ats_platform, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_hash, title, company, location, apply_url,
                     description, ats_platform, source),
                )
                inserted = cursor.rowcount == 1
                if inserted:
                    logger.info("Inserted new job: %s @ %s", title, company)
                else:
                    logger.debug("Duplicate job skipped: %s @ %s", title, company)
                return inserted
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert job '{title}' @ '{company}': {e}") from e

    def get_job_by_hash(self, job_hash: str) -> Optional[JobRecord]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_hash = ?", (job_hash,)
                ).fetchone()
                return JobRecord(**dict(row)) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to fetch job by hash: {e}") from e

    def get_jobs_by_status(self, status: str) -> list[JobRecord]:
        if status not in VALID_STATUSES:
            raise DatabaseError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY date_discovered DESC", (status,)
                ).fetchall()
                return [JobRecord(**dict(r)) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to fetch jobs with status '{status}': {e}") from e

    def update_fit_score(self, job_id: int, fit_score: float, fit_reasoning: str) -> None:
        """Store the AI-generated fit score/reasoning and bump status to SCORED."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET fit_score = ?, fit_reasoning = ?, status = 'SCORED',
                        date_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (fit_score, fit_reasoning, job_id),
                )
                if cursor.rowcount == 0:
                    raise DatabaseError(f"No job found with id {job_id}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update fit score for job {job_id}: {e}") from e

    def update_status(self, job_id: int, status: str) -> None:
        if status not in VALID_STATUSES:
            raise DatabaseError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, date_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status, job_id),
                )
                if cursor.rowcount == 0:
                    raise DatabaseError(f"No job found with id {job_id}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update status for job {job_id}: {e}") from e

    def get_unscored_jobs(self) -> list[JobRecord]:
        """Convenience helper for evaluator.py: jobs discovered but not yet fit-scored."""
        return self.get_jobs_by_status("DISCOVERED")

    # ------------------------------------------------------------------
    # Application operations
    # ------------------------------------------------------------------

    def create_application(self, job_id: int, tailored_pdf_path: Optional[str] = None) -> int:
        """Create an application record tied to a job. Returns the new application id."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO applications (job_id, tailored_pdf_path)
                    VALUES (?, ?)
                    """,
                    (job_id, tailored_pdf_path),
                )
                return cursor.lastrowid
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to create application for job {job_id}: {e}") from e

    def mark_applied(self, application_id: int) -> None:
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    UPDATE applications
                    SET applied_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (application_id,),
                )
                if cursor.rowcount == 0:
                    raise DatabaseError(f"No application found with id {application_id}")
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to mark application {application_id} as applied: {e}") from e


if __name__ == "__main__":
    # Quick manual smoke test: `python database.py`
    db = Database()
    db.init_schema()

    created = db.insert_job(
        title="Software Engineering Intern",
        company="Acme Corp",
        apply_url="https://boards.greenhouse.io/acme/jobs/123",
        location="Toronto, ON",
        description="Looking for a Python/FastAPI intern...",
        ats_platform="greenhouse",
        source="linkedin",
    )
    print(f"New job inserted: {created}")

    duplicate = db.insert_job(
        title="Software Engineering Intern",
        company="Acme Corp",
        apply_url="https://boards.greenhouse.io/acme/jobs/123-different-url",
        location="Toronto, ON",
    )
    print(f"Duplicate correctly skipped: {not duplicate}")

    discovered = db.get_jobs_by_status("DISCOVERED")
    print(f"Jobs with status DISCOVERED: {len(discovered)}")