-- AutoApply AI Database Schema
-- SQLite database storing discovered jobs and their application lifecycle

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_hash TEXT UNIQUE NOT NULL,       -- hash of company_name + job_title + location, used for dedup
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    apply_url TEXT NOT NULL,
    description TEXT,                    -- full job description, needed later for fit scoring
    ats_platform TEXT,                   -- 'greenhouse', 'lever', 'workday', 'other'
    source TEXT,                         -- 'linkedin', 'indeed', 'glassdoor', 'zip_recruiter', 'github_repo'
    fit_score REAL,
    fit_reasoning TEXT,
    date_discovered DATETIME DEFAULT CURRENT_TIMESTAMP,
    date_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'DISCOVERED'     -- 'DISCOVERED', 'SCORED', 'TAILORED', 'APPLIED', 'SKIPPED', 'REJECTED'
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_fit_score ON jobs(fit_score);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    tailored_pdf_path TEXT,
    custom_cover_letter TEXT,
    applied_at DATETIME,
    outreach_contact_name TEXT,
    outreach_contact_email TEXT,
    draft_email_id TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id);