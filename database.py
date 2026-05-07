"""
database.py — Unified SQLite database: students + embeddings + attendance.

Thread safety
-------------
All writes are serialised through _write_lock (threading.Lock).
WAL journal mode allows concurrent reads alongside a write.
Each call opens its own connection and closes it on exit (connection-per-call
pattern avoids "database is locked" errors in multi-threaded use).

Schema
------
students   (id, name, enrolled_at, image_count, embedding BLOB)
attendance (id, student_id, student_name, date, time, status)

Indexes
-------
idx_student_date ON attendance(student_id, date)  -- duplicate check
idx_date         ON attendance(date)              -- per-date queries
"""

import logging
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional

import numpy as np

import config
import utils

logger = logging.getLogger("attendance_system.database")

_write_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────
#  Connection context manager
# ─────────────────────────────────────────────────────────────────

@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a WAL-mode SQLite connection that commits on success,
    rolls back on exception, and always closes.
    """
    con = sqlite3.connect(config.DATABASE_FILE, check_same_thread=False, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────
#  Schema init
# ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables and indexes. Safe to call multiple times (idempotent)."""
    import os
    os.makedirs(os.path.dirname(config.DATABASE_FILE) or ".", exist_ok=True)

    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                enrolled_at TEXT NOT NULL,
                image_count INTEGER NOT NULL DEFAULT 0,
                embedding   BLOB
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id   TEXT NOT NULL,
                student_name TEXT NOT NULL,
                date         TEXT NOT NULL,
                time         TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'Present'
            )""")
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_student_date
            ON attendance(student_id, date)""")
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_date
            ON attendance(date)""")

    logger.info("Database ready: %s", config.DATABASE_FILE)


# ─────────────────────────────────────────────────────────────────
#  Embedding serialisation
# ─────────────────────────────────────────────────────────────────

def _emb_to_blob(emb: np.ndarray) -> bytes:
    return pickle.dumps(emb.astype(np.float32))


def _blob_to_emb(blob: bytes) -> np.ndarray:
    return pickle.loads(blob).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  Student CRUD
# ─────────────────────────────────────────────────────────────────

def add_student(
    student_id: str,
    name: str,
    image_count: int = 0,
    embedding: Optional[np.ndarray] = None,
) -> bool:
    """Insert student. Returns False if ID already exists."""
    blob = _emb_to_blob(embedding) if embedding is not None else None
    now  = utils.current_datetime()
    try:
        with _write_lock:
            with _conn() as con:
                con.execute(
                    "INSERT INTO students (id, name, enrolled_at, image_count, embedding)"
                    " VALUES (?,?,?,?,?)",
                    (student_id, name, now, image_count, blob),
                )
        logger.info("Student added: %s — %s", student_id, name)
        return True
    except sqlite3.IntegrityError:
        logger.warning("Student '%s' already exists.", student_id)
        return False


def update_student_embedding(
    student_id: str,
    embedding: np.ndarray,
    image_count: int = 0,
) -> None:
    """Update embedding (and image count) for an existing student."""
    blob = _emb_to_blob(embedding)
    with _write_lock:
        with _conn() as con:
            con.execute(
                "UPDATE students SET embedding=?, image_count=? WHERE id=?",
                (blob, image_count, student_id),
            )
    logger.debug("Embedding updated: %s", student_id)


def get_student(student_id: str) -> Optional[dict]:
    """Return student dict (embedding decoded) or None."""
    with _conn() as con:
        row = con.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("embedding"):
        d["embedding"] = _blob_to_emb(d["embedding"])
    return d


def student_exists(student_id: str) -> bool:
    with _conn() as con:
        row = con.execute("SELECT 1 FROM students WHERE id=?", (student_id,)).fetchone()
    return row is not None


def get_all_students() -> List[dict]:
    """Return all student records with embeddings decoded."""
    with _conn() as con:
        rows = con.execute("SELECT * FROM students ORDER BY enrolled_at DESC").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("embedding"):
            d["embedding"] = _blob_to_emb(d["embedding"])
        result.append(d)
    return result


def get_all_embeddings() -> List[tuple]:
    """
    Return [(student_id, student_name, embedding_ndarray), …].
    Excludes students with no embedding.
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT id, name, embedding FROM students WHERE embedding IS NOT NULL"
        ).fetchall()
    return [
        (row["id"], row["name"], _blob_to_emb(row["embedding"]))
        for row in rows
    ]


def delete_student(student_id: str) -> bool:
    """Delete student record. Returns True if a row was deleted."""
    with _write_lock:
        with _conn() as con:
            cur = con.execute("DELETE FROM students WHERE id=?", (student_id,))
    deleted = cur.rowcount > 0
    if deleted:
        logger.info("Student %s deleted.", student_id)
    return deleted


def get_student_count() -> int:
    with _conn() as con:
        row = con.execute("SELECT COUNT(*) FROM students").fetchone()
    return row[0] if row else 0


# ─────────────────────────────────────────────────────────────────
#  Attendance
# ─────────────────────────────────────────────────────────────────

def mark_attendance(student_id: str, student_name: str) -> bool:
    """
    Mark student present today (once only per day).
    Thread-safe.  Returns True when a new record was written.
    """
    today = utils.current_date()
    now   = utils.current_time()

    # Fast duplicate check (read — no lock needed in WAL mode)
    with _conn() as con:
        dup = con.execute(
            "SELECT 1 FROM attendance WHERE student_id=? AND date=?",
            (student_id, today),
        ).fetchone()
    if dup:
        logger.debug("Duplicate skip: %s on %s", student_id, today)
        return False

    with _write_lock:
        # Re-check inside lock to close the TOCTOU window
        with _conn() as con:
            dup2 = con.execute(
                "SELECT 1 FROM attendance WHERE student_id=? AND date=?",
                (student_id, today),
            ).fetchone()
            if dup2:
                return False
            con.execute(
                "INSERT INTO attendance (student_id, student_name, date, time, status)"
                " VALUES (?,?,?,?,'Present')",
                (student_id, student_name, today, now),
            )
    logger.info("Attendance marked: %s (%s) at %s %s", student_id, student_name, today, now)
    return True


def get_attendance_by_date(date: Optional[str] = None) -> List[dict]:
    """Return all attendance records for a date (default: today)."""
    date = date or utils.current_date()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM attendance WHERE date=? ORDER BY time",
            (date,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_attendance_by_student(student_id: str) -> List[dict]:
    """Return all attendance history for one student, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM attendance WHERE student_id=? ORDER BY date DESC, time DESC",
            (student_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_students_attendance() -> List[dict]:
    """Aggregated (student_id, student_name, total_present) across all time."""
    with _conn() as con:
        rows = con.execute(
            """SELECT student_id, student_name, COUNT(*) AS total_present
               FROM attendance GROUP BY student_id ORDER BY student_name"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_attendance_between(start: str, end: str) -> List[dict]:
    """Return all records between two ISO dates inclusive."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM attendance WHERE date>=? AND date<=? ORDER BY date, time",
            (start, end),
        ).fetchall()
    return [dict(r) for r in rows]


def get_present_count_between(student_id: str, start: str, end: str) -> int:
    """Count attendance records for one student in a date range."""
    with _conn() as con:
        row = con.execute(
            "SELECT COUNT(*) FROM attendance WHERE student_id=? AND date>=? AND date<=?",
            (student_id, start, end),
        ).fetchone()
    return row[0] if row else 0


def is_marked_today(student_id: str) -> bool:
    return bool(get_attendance_by_date() and any(
        r["student_id"] == student_id for r in get_attendance_by_date()
    ))


# ─────────────────────────────────────────────────────────────────
#  Pretty-print helpers
# ─────────────────────────────────────────────────────────────────

def print_all_students() -> None:
    students = get_all_students()
    if not students:
        print("  No students enrolled.")
        return
    print(f"\n  {'ID':<12} {'Name':<25} {'Imgs':>4}  {'Emb':^5}  Enrolled")
    print("  " + "─" * 65)
    for s in students:
        has_emb = "✓" if s.get("embedding") is not None else "✗"
        print(f"  {s['id']:<12} {s['name']:<25} {s['image_count']:>4}  {has_emb:^5}  {s['enrolled_at']}")
    print()


def print_attendance(date: Optional[str] = None) -> None:
    date    = date or utils.current_date()
    records = get_attendance_by_date(date)
    print(f"\n  Attendance — {date}")
    print("  " + "═" * 55)
    if not records:
        print("  No records found.")
    else:
        print(f"  {'#':<4} {'ID':<12} {'Name':<25} {'Time':<10} Status")
        print("  " + "─" * 55)
        for i, r in enumerate(records, 1):
            print(f"  {i:<4} {r['student_id']:<12} {r['student_name']:<25}"
                  f" {r['time']:<10} {r['status']}")
        print(f"\n  Total present: {len(records)}")
    print()