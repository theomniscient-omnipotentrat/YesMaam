-- schema.sql — SQLite schema for the attendance system.
-- Applied once by initialize_database() using executescript().

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Students ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS students (
    student_id   TEXT        PRIMARY KEY,
    name         TEXT        NOT NULL,
    email        TEXT,
    registered_at TIMESTAMP  DEFAULT CURRENT_TIMESTAMP
);

-- ── Attendance ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT     NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    date        DATE     NOT NULL,
    time        TIME     NOT NULL,
    status      TEXT     NOT NULL DEFAULT 'Present',
    UNIQUE(student_id, date)          -- one record per student per day
);

-- ── Index for O(log n) history lookups ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_attendance_student_date
    ON attendance(student_id, date);

CREATE INDEX IF NOT EXISTS idx_attendance_date
    ON attendance(date);
