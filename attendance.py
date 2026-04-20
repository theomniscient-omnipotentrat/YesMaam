"""
attendance.py — Mark attendance, prevent duplicates, save to CSV,
                and generate daily text reports.
"""

import csv
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

import config
import utils

logger = logging.getLogger("attendance_system.attendance")

# CSV column headers
HEADERS = ["student_id", "student_name", "date", "time", "status"]

# In-memory cache: {(student_id, date): True} — reset each process run
_marked_cache: Dict[str, bool] = {}


# ──────────────────────────────────────────────────────────────────
#  Initialisation
# ──────────────────────────────────────────────────────────────────

def init_attendance_file() -> None:
    """Create attendance CSV with header if it doesn't exist."""
    if not os.path.isfile(config.ATTENDANCE_FILE):
        with open(config.ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
        logger.info("Attendance file created at %s", config.ATTENDANCE_FILE)


# ──────────────────────────────────────────────────────────────────
#  Core attendance logic
# ──────────────────────────────────────────────────────────────────

def is_already_marked(student_id: str, date: Optional[str] = None) -> bool:
    """
    Check whether attendance for student_id has been recorded today
    (or on `date` if supplied).  Uses both in-memory cache and CSV.
    """
    date = date or utils.current_date()
    cache_key = f"{student_id}:{date}"

    # Fast path — in-memory cache
    if _marked_cache.get(cache_key):
        return True

    # Slower path — scan CSV (needed after restart)
    if not os.path.isfile(config.ATTENDANCE_FILE):
        return False

    with open(config.ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("student_id") == student_id and row.get("date") == date:
                _marked_cache[cache_key] = True   # warm cache
                return True
    return False


def mark_attendance(student_id: str, student_name: str) -> bool:
    """
    Append an attendance record for student_id if not already marked today.
    Returns True when a new record was written, False on duplicate.
    """
    date = utils.current_date()
    time = utils.current_time()

    if is_already_marked(student_id, date):
        logger.info("Duplicate skip — %s already marked on %s", student_id, date)
        return False

    row = {
        "student_id":   student_id,
        "student_name": student_name,
        "date":         date,
        "time":         time,
        "status":       "Present",
    }

    init_attendance_file()  # idempotent — creates file+header if missing
    with open(config.ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)

    # Update in-memory cache
    _marked_cache[f"{student_id}:{date}"] = True

    logger.info("Attendance marked — %s (%s) at %s %s", student_id, student_name, date, time)
    return True


# ──────────────────────────────────────────────────────────────────
#  Query helpers
# ──────────────────────────────────────────────────────────────────

def get_attendance_by_date(date: Optional[str] = None) -> List[dict]:
    """Return all attendance records for a given date (default: today)."""
    date = date or utils.current_date()
    records = []
    if not os.path.isfile(config.ATTENDANCE_FILE):
        return records

    with open(config.ATTENDANCE_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("date") == date:
                records.append(dict(row))
    return records


def get_all_attendance() -> List[dict]:
    """Return all attendance records."""
    if not os.path.isfile(config.ATTENDANCE_FILE):
        return []
    with open(config.ATTENDANCE_FILE, "r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def get_attendance_by_student(student_id: str) -> List[dict]:
    """Return attendance history for a specific student."""
    return [r for r in get_all_attendance() if r.get("student_id") == student_id]


# ──────────────────────────────────────────────────────────────────
#  Display & reporting
# ──────────────────────────────────────────────────────────────────

def print_attendance(date: Optional[str] = None) -> None:
    """Pretty-print attendance for a date."""
    date = date or utils.current_date()
    records = get_attendance_by_date(date)

    print(f"\n  Attendance Report — {date}")
    print("  " + "=" * 55)

    if not records:
        print("  No records found for this date.")
    else:
        print(f"  {'#':<4} {'ID':<12} {'Name':<25} {'Time':<10} Status")
        print("  " + "-" * 55)
        for i, r in enumerate(records, 1):
            print(
                f"  {i:<4} {r['student_id']:<12} {r['student_name']:<25}"
                f" {r['time']:<10} {r['status']}"
            )
        print(f"\n  Total present: {len(records)}")
    print()


def generate_daily_report(date: Optional[str] = None) -> str:
    """
    Write a plain-text daily report to reports/ directory.
    Returns the path of the report file.
    """
    date = date or utils.current_date()
    records = get_attendance_by_date(date)
    utils.ensure_dir(config.REPORT_DIR)

    report_path = os.path.join(config.REPORT_DIR, f"attendance_{date}.txt")

    lines = [
        "=" * 60,
        f"  ATTENDANCE REPORT  —  {date}",
        "=" * 60,
        f"  Generated: {utils.current_datetime()}",
        f"  Total Present: {len(records)}",
        "",
        f"  {'#':<4} {'ID':<12} {'Name':<25} {'Time':<10} Status",
        "  " + "-" * 55,
    ]
    for i, r in enumerate(records, 1):
        lines.append(
            f"  {i:<4} {r['student_id']:<12} {r['student_name']:<25}"
            f" {r['time']:<10} {r['status']}"
        )
    lines += ["", "=" * 60]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Daily report saved: %s", report_path)
    return report_path
