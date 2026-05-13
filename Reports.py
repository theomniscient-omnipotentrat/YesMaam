"""
Reports.py — Attendance report generation for Face Attendance System v4.

Generates three report types, each saved as CSV to config.REPORTS_DIR:

1. generate_attendance_report(start_date, end_date)
2. generate_absent_report(target_date, send_email)
3. generate_daily_summary(target_date)
"""

import csv
import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional

import config
import database
import utils

logger = logging.getLogger("attendance_system.reports")


def _ensure_reports_dir() -> None:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def generate_attendance_report(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> str:
    _ensure_reports_dir()
    if not end_date:
        end_date   = utils.current_date()
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    records = database.get_attendance_between(start_date, end_date)
    path = config.REPORT_CSV_FILE
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "student_id", "student_name", "date", "time", "status"],
        )
        writer.writeheader()
        writer.writerows(records)

    logger.info("Full report: %d records (%s to %s) saved to %s",
                len(records), start_date, end_date, path)
    print(f"  Full report: {len(records)} records  ({start_date} to {end_date})")
    return path


def generate_absent_report(
    target_date: Optional[str] = None,
    send_email:  bool = False,
) -> str:
    _ensure_reports_dir()
    date    = target_date or utils.current_date()
    present = {r["student_id"] for r in database.get_attendance_by_date(date)}
    absent  = [s for s in database.get_all_students() if s["id"] not in present]

    path = config.ABSENT_REPORT_CSV_FILE
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Student ID", "Name", "Enrolled At", "Date", "Status"])
        for s in absent:
            writer.writerow([s["id"], s["name"], s["enrolled_at"], date, "Absent"])

    logger.info("Absent report: %d absent on %s -> %s", len(absent), date, path)
    print(f"  Absent report: {len(absent)} absent on {date}")
    if send_email:
        _send_email_report(path, date, len(absent))
    return path


def generate_daily_summary(target_date: Optional[str] = None) -> str:
    _ensure_reports_dir()
    date    = target_date or utils.current_date()
    present = {r["student_id"] for r in database.get_attendance_by_date(date)}
    all_stus = database.get_all_students()

    path = os.path.join(config.REPORTS_DIR, f"daily_summary_{date}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Student ID", "Name", "Date", "Status"])
        present_count = 0
        for s in all_stus:
            status = "Present" if s["id"] in present else "Absent"
            if status == "Present":
                present_count += 1
            writer.writerow([s["id"], s["name"], date, status])
        writer.writerow([])
        writer.writerow(["Total Enrolled", len(all_stus), "", ""])
        writer.writerow(["Total Present",  present_count, "", ""])
        writer.writerow(["Total Absent",   len(all_stus) - present_count, "", ""])

    logger.info("Daily summary: %d/%d present on %s -> %s",
                present_count, len(all_stus), date, path)
    return path


def _send_email_report(csv_path: str, date: str, absent_count: int) -> None:
    if not config.EMAIL_ENABLED:
        print("  [INFO] Set EMAIL_ENABLED=True in config.py to enable email reports.")
        return
    msg = MIMEMultipart()
    msg["From"]    = config.SMTP_USER
    msg["To"]      = config.EMAIL_RECIPIENT
    msg["Subject"] = f"{config.EMAIL_SUBJECT} — {date} ({absent_count} absent)"
    msg.attach(MIMEText(
        f"Absent Students Report\nDate: {date}\nAbsent: {absent_count}\n"
        f"\nSee attached CSV.\n\n— Face Attendance System v4", "plain"
    ))
    with open(csv_path, "rb") as fh:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(fh.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition",
                    f'attachment; filename="{os.path.basename(csv_path)}"')
    msg.attach(part)
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_USER, config.EMAIL_RECIPIENT, msg.as_string())
        print(f"  Email sent to {config.EMAIL_RECIPIENT}")
    except Exception as exc:
        print(f"  [ERROR] Email failed: {exc}")
