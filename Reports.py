"""
Reports.py — CSV report generation + optional email dispatch.

Public API
----------
generate_attendance_report(start_date, end_date) → path
generate_absent_report(target_date, send_email)  → path
generate_daily_summary(target_date)              → path

All functions return the path of the saved CSV file.
Dates are ISO strings (YYYY-MM-DD); None defaults to today / 30-days-ago.
"""

import csv
import logging
import os
import smtplib
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import config
import database
import utils

logger = logging.getLogger("attendance_system.reports")


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _ensure_reports_dir() -> None:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_csv(path: str, fieldnames: list, rows: list) -> str:
    """Write rows to a CSV at path. Returns path."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Report written: %s (%d rows)", path, len(rows))
    return path


# ─────────────────────────────────────────────────────────────────
#  1. Full Attendance Report
# ─────────────────────────────────────────────────────────────────

def generate_attendance_report(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> str:
    """
    Export every attendance record between start_date and end_date
    (inclusive) to a timestamped CSV in REPORTS_DIR.

    Parameters
    ----------
    start_date : YYYY-MM-DD (default: 30 days ago)
    end_date   : YYYY-MM-DD (default: today)

    Returns the path of the saved file.
    """
    _ensure_reports_dir()
    start = start_date or _days_ago(30)
    end   = end_date   or _today()

    records = database.get_attendance_between(start, end)

    # Annotate with day-of-week for readability
    for r in records:
        try:
            dow = datetime.strptime(r["date"], "%Y-%m-%d").strftime("%A")
        except ValueError:
            dow = ""
        r["day_of_week"] = dow

    path = os.path.join(
        config.REPORTS_DIR,
        f"attendance_{start}_to_{end}_{_stamp()}.csv",
    )
    fields = ["date", "day_of_week", "time", "student_id", "student_name", "status"]
    _write_csv(path, fields, records)

    print(f"  Full attendance report: {path}  ({len(records)} records)")
    return path


# ─────────────────────────────────────────────────────────────────
#  2. Absent Students Report
# ─────────────────────────────────────────────────────────────────

def generate_absent_report(
    target_date: Optional[str] = None,
    send_email:  bool          = False,
) -> str:
    """
    List every enrolled student who has NO attendance record on target_date.

    Parameters
    ----------
    target_date : YYYY-MM-DD (default: today)
    send_email  : if True and EMAIL_ENABLED, email the CSV to EMAIL_RECIPIENT

    Returns the path of the saved CSV.
    """
    _ensure_reports_dir()
    date     = target_date or _today()
    present  = {r["student_id"] for r in database.get_attendance_by_date(date)}
    students = database.get_all_students()

    absent_rows = [
        {
            "date":         date,
            "student_id":   s["id"],
            "student_name": s["name"],
            "status":       "Absent",
        }
        for s in students
        if s["id"] not in present
    ]

    path = os.path.join(
        config.REPORTS_DIR,
        f"absent_{date}_{_stamp()}.csv",
    )
    fields = ["date", "student_id", "student_name", "status"]
    _write_csv(path, fields, absent_rows)

    total   = len(students)
    n_abs   = len(absent_rows)
    n_pres  = total - n_abs
    print(
        f"  Absent report ({date}): {n_abs} absent / {n_pres} present "
        f"/ {total} total  →  {path}"
    )

    if send_email and config.EMAIL_ENABLED:
        _send_email_report(path, date, n_abs, total)

    return path


# ─────────────────────────────────────────────────────────────────
#  3. Daily Summary
# ─────────────────────────────────────────────────────────────────

def generate_daily_summary(target_date: Optional[str] = None) -> str:
    """
    One row per enrolled student: name, present/absent, time-in if present.

    Parameters
    ----------
    target_date : YYYY-MM-DD (default: today)

    Returns the path of the saved CSV.
    """
    _ensure_reports_dir()
    date      = target_date or _today()
    att_map   = {r["student_id"]: r for r in database.get_attendance_by_date(date)}
    students  = database.get_all_students()

    rows = []
    for s in students:
        rec = att_map.get(s["id"])
        rows.append({
            "date":         date,
            "student_id":   s["id"],
            "student_name": s["name"],
            "status":       "Present" if rec else "Absent",
            "time_in":      rec["time"] if rec else "",
        })

    # Sort: Present first (alphabetical), then Absent
    rows.sort(key=lambda r: (0 if r["status"] == "Present" else 1, r["student_name"]))

    path = os.path.join(
        config.REPORTS_DIR,
        f"daily_summary_{date}_{_stamp()}.csv",
    )
    fields = ["date", "student_id", "student_name", "status", "time_in"]
    _write_csv(path, fields, rows)

    present = sum(1 for r in rows if r["status"] == "Present")
    print(
        f"  Daily summary ({date}): {present}/{len(rows)} present  →  {path}"
    )
    return path


# ─────────────────────────────────────────────────────────────────
#  Email helper
# ─────────────────────────────────────────────────────────────────

def _send_email_report(
    csv_path:  str,
    date:      str,
    n_absent:  int,
    n_total:   int,
) -> None:
    """Send the absent report CSV as an email attachment via SMTP."""
    try:
        msg = MIMEMultipart()
        msg["From"]    = config.SMTP_USER
        msg["To"]      = config.EMAIL_RECIPIENT
        msg["Subject"] = f"{config.EMAIL_SUBJECT} — {date}"

        body = (
            f"Absent Students Report\n"
            f"Date    : {date}\n"
            f"Absent  : {n_absent}\n"
            f"Total   : {n_total}\n"
            f"Present : {n_total - n_absent}\n\n"
            f"See attached CSV for details."
        )
        msg.attach(MIMEText(body, "plain"))

        with open(csv_path, "rb") as fh:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(csv_path)}"',
        )
        msg.attach(part)

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_USER, config.EMAIL_RECIPIENT, msg.as_string())

        logger.info("Absent report emailed to %s", config.EMAIL_RECIPIENT)
        print(f"  Email sent to {config.EMAIL_RECIPIENT}")

    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        print(f"  [WARNING] Email failed: {exc}")
