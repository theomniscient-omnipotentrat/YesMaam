# Face Attendance System v4 — Pi 5 Optimised

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# For tkinter on Pi OS Lite:
sudo apt install python3-tk
```

### 2. Launch GUI
```bash
python gui.py
```

### 3. Launch CLI (original menu interface)
```bash
python main.py
```

---

## GUI Overview

| Section | Description |
|---|---|
| **Dashboard** | Live stats (enrolled, present, absent, rate) + today's table |
| **Enroll Student** | Enter ID + name, launch camera, SPACE to capture |
| **Take Attendance** | Launch recognition camera; marks rows in real-time |
| **View by Date** | Query attendance for any date |
| **Students** | Full enrolled list; remove students |
| **Reports** | Export CSV reports (full, absent, daily summary) |
| **Campus Sim** | Simulate zone-based campus monitoring and timetable mismatch alerts |

---

## Simulated Campus Monitoring

The project now includes a small simulation layer that matches the proposal's campus-monitoring objective without performing continuous real-world surveillance.

```bash
# Generate simulated zone events and timetable mismatch alerts
python main.py --simulate-campus --date 2026-05-07 --anomaly-rate 0.30

# Export the simulated campus monitoring CSV report
python main.py --campus-report --date 2026-05-07
```

Simulation data is stored in SQLite using these tables:

| Table | Purpose |
|---|---|
| `campus_zones` | Known zones such as Classroom A, Computer Lab, Library, Cafeteria |
| `campus_timetable` | Expected student location by weekday and time range |
| `campus_zone_events` | Simulated detections in campus zones |
| `campus_alerts` | Mismatches between expected zone and detected zone |

The feature is intentionally labelled as a simulation for ethical and academic reasons. It demonstrates smart-campus logic while keeping real biometric use limited to attendance.

## Pi 5 Optimisations Applied

| File | Change | Benefit |
|---|---|---|
| `detector.py` | `DNN_TARGET_CPU_FP16` + 320 px internal downscale | ~3× faster YuNet (~40 ms → ~12 ms) |
| `embeddings.py` | `allowed_modules=['recognition']` | −400 MB RAM, −70% load time |
| `camera.py` | `FrameConsumer` → `multiprocessing.Process` | Real multi-core, bypasses GIL |
| `camera.py` | 300 ms ArcFace throttle per face | −90% embedding CPU load |
| `Recognition.py` | `waitKey(20)` | Wayland/X11 compositor breathing room |
| `enroll.py` | Pipeline-style threaded UI | Non-blocking enrollment window |
| `config.py` | `ENROLL_IMAGES_COUNT=3`, `CAMERA_WARMUP_SECS=0.5` | Faster enrollment |

---

## File Structure

```
attendance_system/
├── gui.py           ← GUI entry point  (NEW)
├── main.py          ← CLI entry point
├── config.py        ← All constants
├── camera.py        ← Producer/Consumer pipeline (multiprocessing)
├── detector.py      ← YuNet ONNX face detection (FP16)
├── embeddings.py    ← ArcFace embeddings (recognition-only)
├── enroll.py        ← Threaded enrollment
├── Recognition.py   ← Real-time attendance session
├── Reports.py       ← CSV report generation
├── campus_monitoring.py ← Simulated timetable + zone monitoring
├── database.py      ← SQLite CRUD layer
├── utils.py         ← Shared helpers
└── requirements.txt
```

## Saved Video → Campus Simulation Sample Data

You can use a saved video as repeatable sample input for the campus monitoring simulation. The video represents one simulated camera zone, such as `Library` or `Computer Lab`. The system samples frames, detects faces, recognises enrolled students using the existing ArcFace embedding database, and logs zone events into the campus simulation tables.

Example:

```bash
python main.py --video-campus sample_videos/library_demo.mp4 \
  --zone "Library" \
  --date 2026-05-07 \
  --start-time 09:00 \
  --sample-every-seconds 2 \
  --event-cooldown-seconds 60
```

Then export the report:

```bash
python main.py --campus-report --date 2026-05-07
```

Important: the video-based mode still requires students to be enrolled first. Unknown faces are counted but not logged as identified students. This keeps the feature suitable for a consent-based academic simulation rather than real continuous campus surveillance.
