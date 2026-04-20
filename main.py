"""
main.py — Entry point for the Face Recognition Attendance System.

Menu
────
1. Enroll Student
2. Take Attendance
3. View Attendance
4. View All Students
5. Generate Daily Report
6. Retrain Model
7. Exit
"""

import os
import sys
import logging

# ── make sure all local modules are importable ────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils
import database
import attendance
import enroll
import recognize

logger = logging.getLogger("attendance_system.main")


# ──────────────────────────────────────────────────────────────────
#  Startup
# ──────────────────────────────────────────────────────────────────

def _bootstrap() -> None:
    """Initialise directories, DB, and attendance file."""
    utils.ensure_dir(config.DATASET_DIR)
    utils.ensure_dir(config.REPORT_DIR)
    database.init_db()
    attendance.init_attendance_file()
    logger.info("System bootstrap complete.")


# ──────────────────────────────────────────────────────────────────
#  Menu helpers
# ──────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════╗
║    FACE RECOGNITION ATTENDANCE SYSTEM  v1.0      ║
║    Raspberry Pi  |  MediaPipe + OpenCV LBPH       ║
╚══════════════════════════════════════════════════╝
"""

MENU = """
  ┌─────────────────────────────────────┐
  │  1.  Enroll Student                 │
  │  2.  Take Attendance (camera)       │
  │  3.  View Today's Attendance        │
  │  4.  View Attendance by Date        │
  │  5.  View All Students              │
  │  6.  Generate Daily Report          │
  │  7.  Retrain Recognition Model      │
  │  8.  Exit                           │
  └─────────────────────────────────────┘
"""


def _prompt(prompt_text: str) -> str:
    return input(f"  {prompt_text}").strip()


def _clear():
    os.system("clear" if os.name != "nt" else "cls")


# ──────────────────────────────────────────────────────────────────
#  Menu handlers
# ──────────────────────────────────────────────────────────────────

def handle_enroll() -> None:
    print("\n  ── Enroll New Student ──────────────────────────────\n")
    enroll.enroll_student()
    # Invalidate model so next recognition session retrains
    if os.path.isfile(config.MODEL_FILE):
        os.remove(config.MODEL_FILE)
        logger.info("Stale model removed — will retrain on next recognition.")


def handle_take_attendance() -> None:
    print("\n  ── Take Attendance ─────────────────────────────────\n")
    n = database.get_student_count()
    if n == 0:
        print("  No students enrolled.  Enroll at least one student first.\n")
        return
    print(f"  {n} student(s) enrolled.")
    recognize.run_recognition()


def handle_view_today() -> None:
    attendance.print_attendance()


def handle_view_by_date() -> None:
    date_str = _prompt("Enter date (YYYY-MM-DD) [Enter = today]: ")
    if not date_str:
        date_str = utils.current_date()
    attendance.print_attendance(date_str)


def handle_view_students() -> None:
    print()
    database.print_all_students()


def handle_generate_report() -> None:
    date_str = _prompt("Enter date (YYYY-MM-DD) [Enter = today]: ")
    if not date_str:
        date_str = utils.current_date()
    path = attendance.generate_daily_report(date_str)
    print(f"\n  Report saved to: {path}\n")


def handle_retrain() -> None:
    print("\n  ── Retrain Recognition Model ───────────────────────\n")
    if os.path.isfile(config.MODEL_FILE):
        os.remove(config.MODEL_FILE)
    model = recognize.train_model(force=True)
    if model:
        print("  Model training successful.\n")
    else:
        print("  Training failed — ensure students are enrolled.\n")


# ──────────────────────────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────────────────────────

HANDLERS = {
    "1": handle_enroll,
    "2": handle_take_attendance,
    "3": handle_view_today,
    "4": handle_view_by_date,
    "5": handle_view_students,
    "6": handle_generate_report,
    "7": handle_retrain,
}


def main() -> None:
    _bootstrap()
    print(BANNER)
    print(f"  Today: {utils.current_date()}    Students enrolled: {database.get_student_count()}")

    while True:
        print(MENU)
        choice = _prompt("Select option [1-8]: ")

        if choice == "8":
            print("\n  Goodbye!\n")
            utils.gpio_cleanup()
            sys.exit(0)

        handler = HANDLERS.get(choice)
        if handler:
            try:
                handler()
            except KeyboardInterrupt:
                print("\n  Interrupted — returning to menu.\n")
            except Exception as exc:
                logger.exception("Unhandled error in menu handler %s", choice)
                print(f"\n  [ERROR] {exc}\n")
        else:
            print("  Invalid option. Please choose 1–8.\n")


if __name__ == "__main__":
    main()
