"""main.py — Face Attendance System v3 entry point.

Interactive menu mode
---------------------
    python main.py

CLI / automation mode
---------------------
    python main.py --generate-report
    python main.py --generate-report --start 2024-01-01 --end 2024-06-30
    python main.py --report-absent
    python main.py --report-absent --date 2024-06-15
    python main.py --report-absent --email          # also sends email
    python main.py --list-students
    python main.py --view-attendance --date 2024-06-15
"""

import argparse
import logging
import os
import sys
import shutil

# ── local modules ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import database
import enroll
import Recognition as recognition
import Reports as reports
import embeddings as emb_module

# ── logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("attendance_system.main")


# ─────────────────────────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────────────────────────

def _bootstrap() -> None:
    """Initialise directories and database schema."""
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    database.init_db()
    logger.info("System bootstrap complete (v3).")


# ─────────────────────────────────────────────────────────────────
#  Interactive menu
# ─────────────────────────────────────────────────────────────────

BANNER = """
╔════════════════════════════════════════════════════════╗
║   FACE ATTENDANCE SYSTEM  v3                           ║
║   YuNet Detection  ·  ArcFace Embeddings  ·  SQLite    ║
╚════════════════════════════════════════════════════════╝"""

MENU = """
  ┌──────────────────────────────────────────┐
  │  1.  Enroll Student                      │
  │  2.  Take Attendance  (threaded camera)  │
  │  3.  View Today's Attendance             │
  │  4.  View Attendance by Date             │
  │  5.  View All Students                   │
  │  6.  Remove Student                      │
  │  7.  Generate Full Attendance Report     │
  │  8.  Generate Absent Report              │
  │  9.  Generate Daily Summary              │
  │  10. Exit                                │
  └──────────────────────────────────────────┘"""


def _prompt(text: str) -> str:
    return input(f"  {text}").strip()


# ── handlers ──────────────────────────────────────────────────────

def _handle_enroll() -> None:
    print("\n  ── Enroll Student ─────────────────────────────────\n")
    enroll.enroll_student()


def _handle_take_attendance() -> None:
    print("\n  ── Take Attendance ────────────────────────────────\n")
    n = database.get_student_count()
    if n == 0:
        print("  No students enrolled.  Add students first.\n")
        return
    recognition.run_recognition()


def _handle_view_today() -> None:
    database.print_attendance()


def _handle_view_by_date() -> None:
    d = _prompt("Date (YYYY-MM-DD) [Enter = today]: ")
    database.print_attendance(d or None)


def _handle_view_students() -> None:
    print()
    database.print_all_students()


def _handle_remove() -> None:
    print("\n  ── Remove Student ─────────────────────────────────\n")
    database.print_all_students()
    if database.get_student_count() == 0:
        return

    sid = _prompt("Student ID to remove (Enter to cancel): ")
    if not sid:
        print("  Cancelled.\n")
        return

    student = database.get_student(sid)
    if not student:
        print(f"  [ERROR] Student '{sid}' not found.\n")
        return

    print(f"\n  Student : {student['name']}  (ID: {sid})")
    if _prompt("  Type YES to confirm deletion: ") != "YES":
        print("  Cancelled.\n")
        return

    # Delete dataset images
    folder = f"{sid}_{student['name'].replace(' ', '_')}"
    img_dir = os.path.join(config.DATASET_DIR, folder)
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
        print(f"  ✓ Images deleted: {img_dir}")

    database.delete_student(sid)
    print(f"  ✓ '{student['name']}' removed from database.")

    # Reload embedding db
    emb_module.get_embedding_db().reload()
    print("  ✓ Embedding database refreshed.\n")


def _handle_full_report() -> None:
    start = _prompt("Start date YYYY-MM-DD (Enter = 30 days ago): ")
    end   = _prompt("End date   YYYY-MM-DD (Enter = today):       ")
    path  = reports.generate_attendance_report(
        start_date = start or None,
        end_date   = end   or None,
    )
    print(f"  Saved → {path}\n")


def _handle_absent_report() -> None:
    d    = _prompt("Date (YYYY-MM-DD) [Enter = today]: ")
    mail = _prompt("Send email? [y/N]: ").lower() == "y"
    path = reports.generate_absent_report(
        target_date = d    or None,
        send_email  = mail,
    )
    print(f"  Saved → {path}\n")


def _handle_daily_summary() -> None:
    d    = _prompt("Date (YYYY-MM-DD) [Enter = today]: ")
    path = reports.generate_daily_summary(d or None)
    print(f"  Saved → {path}\n")


HANDLERS = {
    "1":  _handle_enroll,
    "2":  _handle_take_attendance,
    "3":  _handle_view_today,
    "4":  _handle_view_by_date,
    "5":  _handle_view_students,
    "6":  _handle_remove,
    "7":  _handle_full_report,
    "8":  _handle_absent_report,
    "9":  _handle_daily_summary,
}


def _run_menu() -> None:
    _bootstrap()
    print(BANNER)
    print(f"\n  Students enrolled : {database.get_student_count()}")

    while True:
        print(MENU)
        choice = _prompt("Select [1-10]: ")

        if choice == "10":
            print("\n  Goodbye!\n")
            sys.exit(0)

        handler = HANDLERS.get(choice)
        if handler:
            try:
                handler()
            except KeyboardInterrupt:
                print("\n  Interrupted — returning to menu.\n")
            except Exception as exc:
                logger.exception("Error in handler %s", choice)
                print(f"\n  [ERROR] {exc}\n")
        else:
            print("  Invalid option. Choose 1–10.\n")


# ─────────────────────────────────────────────────────────────────
#  CLI mode (argparse)
# ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Face Attendance System v3 — CLI interface",
    )
    p.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate full attendance CSV report and exit.",
    )
    p.add_argument(
        "--report-absent",
        action="store_true",
        help="Generate absent students report and exit.",
    )
    p.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        default=None,
        help="Report start date (for --generate-report).",
    )
    p.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        default=None,
        help="Report end date (for --generate-report).",
    )
    p.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Target date (for --report-absent or --view-attendance).",
    )
    p.add_argument(
        "--email",
        action="store_true",
        help="Send absent report via email (requires EMAIL_ENABLED=True in config.py).",
    )
    p.add_argument(
        "--list-students",
        action="store_true",
        help="Print all enrolled students and exit.",
    )
    p.add_argument(
        "--view-attendance",
        action="store_true",
        help="Print attendance for --date (default: today) and exit.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _bootstrap()

    # ── CLI flags ─────────────────────────────────────────────────
    if args.generate_report:
        path = reports.generate_attendance_report(
            start_date = args.start,
            end_date   = args.end,
        )
        print(f"\n  Report saved → {path}\n")
        sys.exit(0)

    if args.report_absent:
        path = reports.generate_absent_report(
            target_date = args.date,
            send_email  = args.email,
        )
        print(f"\n  Absent report → {path}\n")
        sys.exit(0)

    if args.list_students:
        database.print_all_students()
        sys.exit(0)

    if args.view_attendance:
        database.print_attendance(args.date or None)
        sys.exit(0)

    # ── Interactive menu ──────────────────────────────────────────
    _run_menu()


if __name__ == "__main__":
    main()
