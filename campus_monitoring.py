"""
campus_monitoring.py — Video-based simulated campus zone monitoring.

This module extends the Face Attendance System with a controlled simulation
layer for the final-year project. A saved video is treated as footage from one
campus zone camera, for example Library, Classroom A, Computer Lab, Corridor,
or Cafeteria.

Workflow
--------
1. Import a timetable CSV for enrolled students.
2. Process a saved video and assign it a zone name.
3. Recognise enrolled faces using the existing YuNet + ArcFace pipeline.
4. Log recognised students into campus_zone_events.
5. Compare detected zone with expected timetable zone.
6. Create campus_alerts when the detected zone is different from the expected zone.

Ethical boundary
----------------
This is a simulation for academic evaluation. It should not be presented as
real continuous campus surveillance. Unknown faces are not stored as named
students.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import cv2

import config
import database
import embeddings as emb_module
from detector import FaceDetector

logger = logging.getLogger("attendance_system.campus_monitoring")

DEFAULT_ZONES = (
    ("Classroom A", "Main scheduled teaching room."),
    ("Computer Lab", "Practical computing laboratory."),
    ("Library", "Independent study zone."),
    ("Corridor", "Movement area between teaching spaces."),
    ("Cafeteria", "Break and social area."),
)


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_campus_tables() -> None:
    """Create campus monitoring tables if they do not already exist."""
    with database._conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS campus_zones (
                zone_id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name TEXT NOT NULL UNIQUE,
                description TEXT
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS campus_timetable (
                timetable_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                day_of_week TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                expected_zone TEXT NOT NULL,
                activity TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS campus_zone_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                detected_zone TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT NOT NULL,
                source_video TEXT,
                confidence REAL,
                distance REAL
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS campus_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT NOT NULL,
                expected_zone TEXT NOT NULL,
                detected_zone TEXT NOT NULL,
                reason TEXT NOT NULL,
                source_video TEXT
            )
        """)

        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_campus_timetable_student_day
            ON campus_timetable(student_id, day_of_week, start_time, end_time)
        """)

        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_campus_events_date
            ON campus_zone_events(event_date, event_time)
        """)

        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_campus_alerts_date
            ON campus_alerts(event_date, event_time)
        """)


def add_zone(zone_name: str, description: str = "") -> None:
    """Add a campus zone such as Library, Classroom A, or Computer Lab."""
    init_campus_tables()
    zone_name = zone_name.strip()

    if not zone_name:
        raise ValueError("Zone name cannot be empty.")

    with database._conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO campus_zones (zone_name, description) VALUES (?, ?)",
            (zone_name, description),
        )


def seed_default_zones() -> None:
    """Insert a small set of demonstration zones."""
    for zone_name, description in DEFAULT_ZONES:
        add_zone(zone_name, description)


# ---------------------------------------------------------------------------
# Timetable management
# ---------------------------------------------------------------------------

def add_timetable_entry(
    student_id: str,
    day_of_week: str,
    start_time: str,
    end_time: str,
    expected_zone: str,
    activity: str = "Scheduled class",
) -> bool:
    """Add one timetable entry for an enrolled student."""
    init_campus_tables()

    student_id = student_id.strip()
    day_of_week = day_of_week.strip()
    start_time = start_time.strip()
    end_time = end_time.strip()
    expected_zone = expected_zone.strip()
    activity = (activity or "Scheduled class").strip()

    student = database.get_student(student_id)
    if not student:
        print(f"[ERROR] Student ID '{student_id}' does not exist. Enroll the student first.")
        return False

    _validate_day(day_of_week)
    _validate_time(start_time)
    _validate_time(end_time)

    if start_time >= end_time:
        raise ValueError("start_time must be earlier than end_time.")

    add_zone(expected_zone)

    with database._conn() as con:
        con.execute(
            """
            INSERT INTO campus_timetable
            (student_id, student_name, day_of_week, start_time, end_time, expected_zone, activity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, student["name"], day_of_week, start_time, end_time, expected_zone, activity),
        )

    print(f"Added timetable: {student['name']} | {day_of_week} {start_time}-{end_time} | {expected_zone}")
    return True


def import_timetable_csv(csv_path: str) -> int:
    """
    Import timetable entries from CSV.

    Required columns:
        student_id, day_of_week, start_time, end_time, expected_zone

    Optional column:
        activity
    """
    init_campus_tables()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    inserted = 0

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"student_id", "day_of_week", "start_time", "end_time", "expected_zone"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            ok = add_timetable_entry(
                student_id=row["student_id"],
                day_of_week=row["day_of_week"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                expected_zone=row["expected_zone"],
                activity=row.get("activity", "Scheduled class"),
            )
            if ok:
                inserted += 1

    print(f"Imported {inserted} timetable entries.")
    return inserted


def get_timetable() -> List[dict]:
    init_campus_tables()
    with database._conn() as con:
        rows = con.execute(
            """
            SELECT * FROM campus_timetable
            ORDER BY day_of_week, start_time, student_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_expected_timetable_entry(student_id: str, event_date: str, event_time: str) -> Optional[dict]:
    """Return the expected timetable entry for a student at a given date/time."""
    day_name = datetime.strptime(event_date, "%Y-%m-%d").strftime("%A")
    short_time = event_time[:5]

    with database._conn() as con:
        row = con.execute(
            """
            SELECT * FROM campus_timetable
            WHERE student_id = ?
              AND day_of_week = ?
              AND start_time <= ?
              AND end_time >= ?
            ORDER BY start_time
            LIMIT 1
            """,
            (student_id, day_name, short_time, short_time),
        ).fetchone()

    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Event and alert logging
# ---------------------------------------------------------------------------

def log_zone_event(
    student_id: str,
    student_name: str,
    detected_zone: str,
    event_date: str,
    event_time: str,
    source_video: str = "",
    confidence: Optional[float] = None,
    distance: Optional[float] = None,
) -> None:
    """Save one recognised student detection in a simulated zone."""
    init_campus_tables()
    with database._conn() as con:
        con.execute(
            """
            INSERT INTO campus_zone_events
            (student_id, student_name, detected_zone, event_date, event_time,
             source_video, confidence, distance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, student_name, detected_zone, event_date, event_time, source_video, confidence, distance),
        )


def evaluate_zone_event(
    student_id: str,
    student_name: str,
    detected_zone: str,
    event_date: str,
    event_time: str,
    source_video: str = "",
) -> bool:
    """Compare detected zone with expected zone and create an alert if needed."""
    expected = get_expected_timetable_entry(student_id, event_date, event_time)

    # If there is no timetable entry, this is not treated as a violation.
    if expected is None:
        return False

    expected_zone = expected["expected_zone"]

    if expected_zone.strip().lower() == detected_zone.strip().lower():
        return False

    reason = f"Student detected in {detected_zone}, but timetable expected {expected_zone}."

    with database._conn() as con:
        con.execute(
            """
            INSERT INTO campus_alerts
            (student_id, student_name, event_date, event_time,
             expected_zone, detected_zone, reason, source_video)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, student_name, event_date, event_time, expected_zone, detected_zone, reason, source_video),
        )

    return True


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_zone_video(
    video_path: str,
    zone_name: str,
    event_date: str,
    simulated_start_time: str = "09:00",
    sample_every_seconds: float = 2.0,
    event_cooldown_seconds: int = 60,
    save_annotated_video: bool = False,
) -> dict:
    """
    Process a saved video as footage from one simulated campus zone.

    Example:
        process_zone_video(
            video_path="sample_videos/library_sample.mp4",
            zone_name="Library",
            event_date="2026-05-07",
            simulated_start_time="09:00",
        )
    """
    init_campus_tables()
    seed_default_zones()
    add_zone(zone_name)

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    _validate_date(event_date)
    _validate_time(simulated_start_time)

    emb_db = emb_module.get_embedding_db()
    if emb_db.count == 0:
        raise RuntimeError("No enrolled embeddings found. Enroll students first.")

    detector = FaceDetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_interval_frames = max(1, int(fps * sample_every_seconds))
    start_dt = datetime.strptime(f"{event_date} {simulated_start_time}", "%Y-%m-%d %H:%M")

    annotated_writer = None
    annotated_path = ""

    if save_annotated_video:
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        safe_zone = zone_name.replace(" ", "_")
        annotated_path = os.path.join(config.REPORTS_DIR, f"annotated_{base}_{safe_zone}.mp4")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or config.FRAME_WIDTH)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or config.FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    frame_index = 0
    sampled_frames = 0
    recognised_faces = 0
    unknown_faces = 0
    zone_events = 0
    alerts = 0
    last_logged: Dict[tuple, datetime] = {}
    start_processing = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_this_frame = frame_index % sample_interval_frames == 0

        if process_this_frame:
            sampled_frames += 1
            simulated_seconds = frame_index / fps
            event_dt = start_dt + timedelta(seconds=simulated_seconds)
            event_time = event_dt.strftime("%H:%M:%S")

            faces = detector.detect(frame)

            for face in faces:
                x, y, w, h = face["bbox"]
                embedding = emb_module.extract_embedding_from_frame(frame, face["bbox"])

                if embedding is None:
                    unknown_faces += 1
                    _draw_label(frame, x, y, w, h, "No embedding", (0, 0, 255))
                    continue

                student_id, student_name, distance = emb_db.recognize(embedding)

                if student_id == "Unknown":
                    unknown_faces += 1
                    _draw_label(frame, x, y, w, h, "Unknown", (0, 0, 255))
                    continue

                recognised_faces += 1
                cooldown_key = (student_id, zone_name)
                previous_time = last_logged.get(cooldown_key)

                if previous_time:
                    seconds_since_last = (event_dt - previous_time).total_seconds()
                    if seconds_since_last < event_cooldown_seconds:
                        _draw_label(frame, x, y, w, h, f"{student_id} cached", (0, 255, 255))
                        continue

                last_logged[cooldown_key] = event_dt

                log_zone_event(
                    student_id=student_id,
                    student_name=student_name,
                    detected_zone=zone_name,
                    event_date=event_date,
                    event_time=event_time,
                    source_video=os.path.basename(video_path),
                    confidence=face.get("confidence"),
                    distance=distance,
                )
                zone_events += 1

                alert_created = evaluate_zone_event(
                    student_id=student_id,
                    student_name=student_name,
                    detected_zone=zone_name,
                    event_date=event_date,
                    event_time=event_time,
                    source_video=os.path.basename(video_path),
                )

                if alert_created:
                    alerts += 1
                    label = f"ALERT: {student_id} {student_name}"
                    colour = (0, 0, 255)
                else:
                    label = f"OK: {student_id} {student_name}"
                    colour = (0, 255, 0)

                _draw_label(frame, x, y, w, h, label, colour)

            if save_annotated_video:
                _draw_header(frame, f"Zone: {zone_name} | Date: {event_date} | Sim time: {event_time}")

        if annotated_writer is not None:
            annotated_writer.write(frame)

        frame_index += 1

    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()

    processing_time = time.perf_counter() - start_processing

    return {
        "video": video_path,
        "zone": zone_name,
        "event_date": event_date,
        "simulated_start_time": simulated_start_time,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "recognised_faces": recognised_faces,
        "unknown_faces": unknown_faces,
        "zone_events_logged": zone_events,
        "alerts_created": alerts,
        "processing_time_seconds": round(processing_time, 2),
        "annotated_video": annotated_path,
    }


# ---------------------------------------------------------------------------
# Reports and viewing
# ---------------------------------------------------------------------------

def get_zone_events(event_date: Optional[str] = None) -> List[dict]:
    init_campus_tables()
    with database._conn() as con:
        if event_date:
            rows = con.execute(
                """
                SELECT * FROM campus_zone_events
                WHERE event_date = ?
                ORDER BY event_time
                """,
                (event_date,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT * FROM campus_zone_events
                ORDER BY event_date DESC, event_time DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


def get_alerts(event_date: Optional[str] = None) -> List[dict]:
    init_campus_tables()
    with database._conn() as con:
        if event_date:
            rows = con.execute(
                """
                SELECT * FROM campus_alerts
                WHERE event_date = ?
                ORDER BY event_time
                """,
                (event_date,),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT * FROM campus_alerts
                ORDER BY event_date DESC, event_time DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


def export_campus_report(event_date: Optional[str] = None, output_path: Optional[str] = None) -> str:
    """Export campus events and alerts to CSV."""
    init_campus_tables()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    if output_path is None:
        suffix = event_date or "all"
        output_path = os.path.join(config.REPORTS_DIR, f"campus_monitoring_report_{suffix}.csv")

    events = get_zone_events(event_date)
    alerts = get_alerts(event_date)
    alert_lookup = {
        (a["student_id"], a["event_date"], a["event_time"], a["detected_zone"]): a
        for a in alerts
    }

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Student ID", "Student Name", "Date", "Time", "Detected Zone",
            "Expected Zone", "Status", "Reason", "Source Video", "Recognition Distance",
        ])

        for event in events:
            key = (event["student_id"], event["event_date"], event["event_time"], event["detected_zone"])
            alert = alert_lookup.get(key)
            expected = get_expected_timetable_entry(event["student_id"], event["event_date"], event["event_time"])

            if alert:
                expected_zone = alert["expected_zone"]
                status = "ALERT"
                reason = alert["reason"]
            else:
                expected_zone = expected["expected_zone"] if expected else "No timetable"
                status = "OK" if expected else "NO TIMETABLE"
                reason = ""

            writer.writerow([
                event["student_id"], event["student_name"], event["event_date"], event["event_time"],
                event["detected_zone"], expected_zone, status, reason, event["source_video"], event["distance"],
            ])

    print(f"Campus monitoring report saved to: {output_path}")
    return output_path


def print_campus_summary(event_date: Optional[str] = None) -> None:
    """Print campus events and alerts in the terminal."""
    events = get_zone_events(event_date)
    alerts = get_alerts(event_date)

    print("\nCampus Zone Events")
    print("=" * 80)
    if not events:
        print("No zone events found.")
    else:
        print(f"{'Time':<10} {'ID':<12} {'Name':<24} {'Zone':<18} {'Video'}")
        print("-" * 80)
        for event in events:
            print(
                f"{event['event_time']:<10} {event['student_id']:<12} "
                f"{event['student_name']:<24} {event['detected_zone']:<18} {event['source_video']}"
            )

    print("\nCampus Alerts")
    print("=" * 80)
    if not alerts:
        print("No alerts found.")
    else:
        print(f"{'Time':<10} {'ID':<12} {'Name':<24} {'Expected':<18} {'Detected'}")
        print("-" * 80)
        for alert in alerts:
            print(
                f"{alert['event_time']:<10} {alert['student_id']:<12} "
                f"{alert['student_name']:<24} {alert['expected_zone']:<18} {alert['detected_zone']}"
            )
    print()


def clear_campus_data() -> None:
    """Delete simulated campus events and alerts, but keep students and timetable."""
    init_campus_tables()
    with database._conn() as con:
        con.execute("DELETE FROM campus_zone_events")
        con.execute("DELETE FROM campus_alerts")
    print("Campus zone events and alerts cleared.")


def clear_timetable() -> None:
    """Delete campus timetable entries only."""
    init_campus_tables()
    with database._conn() as con:
        con.execute("DELETE FROM campus_timetable")
    print("Campus timetable cleared.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_date(value: str) -> None:
    datetime.strptime(value, "%Y-%m-%d")


def _validate_time(value: str) -> None:
    datetime.strptime(value, "%H:%M")


def _validate_day(value: str) -> None:
    allowed = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    if value not in allowed:
        raise ValueError(f"Invalid day_of_week '{value}'. Use Monday, Tuesday, etc.")


def _draw_label(frame, x: int, y: int, w: int, h: int, label: str, colour) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(
        frame,
        label,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        2,
        cv2.LINE_AA,
    )


def _draw_header(frame, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
