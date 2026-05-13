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
import campus_monitoring

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
    campus_monitoring.init_campus_tables()
    campus_monitoring.seed_default_zones()
    logger.info("System bootstrap complete with campus simulation.")


# ─────────────────────────────────────────────────────────────────
#  Interactive menu
# ─────────────────────────────────────────────────────────────────

BANNER = """
╔════════════════════════════════════════════════════════╗
║   FACE ATTENDANCE SYSTEM  v4                           ║
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
  │  10. Import Campus Timetable CSV         │
  │  11. Process Campus Zone Video           │
  │  12. View Campus Monitoring Summary      │
  │  13. Export Campus Monitoring Report     │
  │  14. Clear Campus Simulation Data        │
  │  15. Exit                                │
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


def _handle_import_timetable() -> None:
    print("\n  ── Import Campus Timetable ─────────────────────────\n")
    path = _prompt("CSV path: ")
    if not path:
        print("  Cancelled.\n")
        return
    try:
        campus_monitoring.import_timetable_csv(path)
    except Exception as exc:
        print(f"  [ERROR] {exc}")
    print()


def _handle_process_campus_video() -> None:
    print("\n  ── Process Campus Zone Video ───────────────────────\n")
    video_path = _prompt("Video path: ")
    zone = _prompt("Zone name e.g. Library/Classroom A: ")
    date = _prompt("Simulation date YYYY-MM-DD: ")
    start_time = _prompt("Video simulated start time HH:MM [09:00]: ") or "09:00"
    sample_raw = _prompt("Sample every N seconds [2.0]: ") or "2.0"
    cooldown_raw = _prompt("Event cooldown seconds [60]: ") or "60"
    save_video_raw = _prompt("Save annotated video? YES/no: ") or "YES"

    try:
        result = campus_monitoring.process_zone_video(
            video_path=video_path,
            zone_name=zone,
            event_date=date,
            simulated_start_time=start_time,
            sample_every_seconds=float(sample_raw),
            event_cooldown_seconds=int(cooldown_raw),
            save_annotated_video=(save_video_raw.upper() == "YES"),
        )
        print("\n  Processing complete:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        campus_monitoring.print_campus_summary(date)
    except Exception as exc:
        print(f"  [ERROR] {exc}")
    print()


def _handle_view_campus_summary() -> None:
    print("\n  ── Campus Monitoring Summary ───────────────────────\n")
    date = _prompt("Date YYYY-MM-DD [Enter = all]: ")
    campus_monitoring.print_campus_summary(date or None)


def _handle_export_campus_report() -> None:
    print("\n  ── Export Campus Monitoring Report ─────────────────\n")
    date = _prompt("Date YYYY-MM-DD [Enter = all]: ")
    try:
        path = campus_monitoring.export_campus_report(date or None)
        print(f"  ✓ Campus report exported: {path}\n")
    except Exception as exc:
        print(f"  [ERROR] {exc}\n")


def _handle_clear_campus_data() -> None:
    print("\n  ── Clear Campus Simulation Data ────────────────────\n")
    confirm = _prompt("Type YES to delete zone events and alerts: ")
    if confirm == "YES":
        campus_monitoring.clear_campus_data()
    else:
        print("  Cancelled.")
    print()


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
    "10": _handle_import_timetable,
    "11": _handle_process_campus_video,
    "12": _handle_view_campus_summary,
    "13": _handle_export_campus_report,
    "14": _handle_clear_campus_data,
}


def _run_menu() -> None:
    _bootstrap()
    print(BANNER)
    print(f"\n  Students enrolled : {database.get_student_count()}")

    while True:
        print(MENU)
        choice = _prompt("Select [1-15]: ")

        if choice == "15":
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
            print("  Invalid option. Choose 1–15.\n")


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

    # Campus monitoring simulation
    p.add_argument("--import-timetable", help="Import campus timetable from CSV file.")
    p.add_argument("--video-campus", help="Process saved video as simulated campus zone footage.")
    p.add_argument("--zone", help="Zone name for the saved video, e.g. Library or Classroom A.")
    p.add_argument("--start-time", default="09:00", help="Simulated video start time in HH:MM format.")
    p.add_argument("--sample-every-seconds", type=float, default=2.0, help="Sample one video frame every N seconds.")
    p.add_argument("--event-cooldown-seconds", type=int, default=60, help="Minimum seconds before logging the same student again in the same zone.")
    p.add_argument("--campus-report", action="store_true", help="Export campus monitoring report.")
    p.add_argument("--clear-campus-data", action="store_true", help="Clear simulated campus zone events and alerts.")
    p.add_argument("--save-annotated-video", action="store_true", help="Save an annotated copy of the processed campus video.")

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

    # ── Campus monitoring CLI commands ────────────────────────────
    if args.import_timetable:
        campus_monitoring.import_timetable_csv(args.import_timetable)
        sys.exit(0)

    if args.clear_campus_data:
        campus_monitoring.clear_campus_data()
        sys.exit(0)

    if args.video_campus:
        if not args.zone:
            print("[ERROR] --zone is required when using --video-campus")
            sys.exit(1)
        if not args.date:
            print("[ERROR] --date YYYY-MM-DD is required when using --video-campus")
            sys.exit(1)

        result = campus_monitoring.process_zone_video(
            video_path=args.video_campus,
            zone_name=args.zone,
            event_date=args.date,
            simulated_start_time=args.start_time,
            sample_every_seconds=args.sample_every_seconds,
            event_cooldown_seconds=args.event_cooldown_seconds,
            save_annotated_video=args.save_annotated_video,
        )

        print("\nCampus video processing complete")
        print("═" * 50)
        for key, value in result.items():
            print(f"{key}: {value}")
        campus_monitoring.print_campus_summary(args.date)
        sys.exit(0)

    if args.campus_report:
        path = campus_monitoring.export_campus_report(args.date)
        print(f"Campus report exported: {path}")
        sys.exit(0)

    # ── Interactive menu ──────────────────────────────────────────
    _run_menu()


if __name__ == "__main__":
    main()
