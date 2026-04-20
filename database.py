"""
database.py — Student record persistence using SQLite.

Schema
------
students(id TEXT PK, name TEXT, enrolled_at TEXT, image_count INT)
"""

import os
import sqlite3
import logging
from typing import Optional, List, Tuple

import config
import utils

logger = logging.getLogger("attendance_system.database")


# ──────────────────────────────────────────────────────────────────
#  Connection helper
# ──────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Return a new SQLite connection with row-factory set."""
    conn = sqlite3.connect(config.STUDENTS_DB)
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────────────────────────────────
#  Initialisation
# ──────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't already exist."""
    utils.ensure_dir(os.path.dirname(config.STUDENTS_DB) or ".")
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                enrolled_at  TEXT NOT NULL,
                image_count  INTEGER DEFAULT 0
            )
        """)
        conn.commit()
    logger.info("Database initialised at %s", config.STUDENTS_DB)


# ──────────────────────────────────────────────────────────────────
#  CRUD
# ──────────────────────────────────────────────────────────────────

def add_student(student_id: str, name: str, image_count: int = 0) -> bool:
    """
    Insert a new student record.
    Returns True on success, False if ID already exists.
    """
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO students (id, name, enrolled_at, image_count) VALUES (?, ?, ?, ?)",
                (student_id, name, utils.current_datetime(), image_count),
            )
            conn.commit()
        logger.info("Student added: %s — %s", student_id, name)
        return True
    except sqlite3.IntegrityError:
        logger.warning("Student ID '%s' already exists.", student_id)
        return False


def update_image_count(student_id: str, count: int) -> None:
    """Update the stored image count for a student."""
    with _connect() as conn:
        conn.execute(
            "UPDATE students SET image_count = ? WHERE id = ?",
            (count, student_id),
        )
        conn.commit()


def get_student(student_id: str) -> Optional[dict]:
    """Return student dict or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE id = ?", (student_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_students() -> List[dict]:
    """Return list of all student records as dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM students ORDER BY enrolled_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def student_exists(student_id: str) -> bool:
    """Quick existence check."""
    return get_student(student_id) is not None


def delete_student(student_id: str) -> bool:
    """Delete a student record. Returns True if a row was deleted."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
    deleted = cur.rowcount > 0
    if deleted:
        logger.info("Student %s deleted.", student_id)
    return deleted


def get_student_count() -> int:
    """Return total number of enrolled students."""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM students").fetchone()
    return row[0] if row else 0


# ──────────────────────────────────────────────────────────────────
#  Display helpers
# ──────────────────────────────────────────────────────────────────

def print_all_students() -> None:
    """Pretty-print the student table to stdout."""
    students = get_all_students()
    if not students:
        print("  No students enrolled yet.")
        return
    print(f"\n  {'ID':<12} {'Name':<25} {'Images':>6}  {'Enrolled At'}")
    print("  " + "-" * 65)
    for s in students:
        print(f"  {s['id']:<12} {s['name']:<25} {s['image_count']:>6}  {s['enrolled_at']}")
    print()
