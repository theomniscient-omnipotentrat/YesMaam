# Face Recognition Attendance System
### Raspberry Pi · MediaPipe · OpenCV LBPH · Python 3

---

## Project Structure

```
face_attendance/
├── main.py           # CLI menu entry point
├── gui.py            # Optional Tkinter GUI
├── config.py         # All tuneable constants
├── utils.py          # Logging, camera, drawing, GPIO helpers
├── database.py       # SQLite student records
├── enroll.py         # Student enrollment (camera capture)
├── recognize.py      # Real-time face recognition + attendance marking
├── attendance.py     # CSV attendance, duplicate prevention, reports
├── requirements.txt  # Python dependencies
│
├── dataset/          # Auto-created on first enroll
│   └── S001_Jane_Doe/
│       ├── 20240101_120000_000001.jpg
│       └── ...
├── students.db       # SQLite database (auto-created)
├── attendance.csv    # Attendance records (auto-created)
├── face_model.yml    # Trained LBPH model (auto-created)
├── reports/          # Daily text reports (auto-created)
└── system.log        # Application log
```

---

## How It Works

```
Camera frame
    │
    ▼
MediaPipe Face Detection ──► bounding boxes (fast, CPU-only)
    │
    ▼
Crop & grayscale face ROI
    │
    ▼
OpenCV LBPH Recognizer ──► label + confidence score
    │
    ▼
Consecutive-frame vote buffer (5 frames) ──► stable identity
    │
    ▼
Confidence < threshold? ──► Mark attendance in CSV
                             Log to students.db
                             GPIO LED + buzzer pulse
```

---

## Raspberry Pi Setup

### 1 — Enable the Camera

```bash
sudo raspi-config
# Interface Options ► Camera ► Enable
sudo reboot
```

### 2 — Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

### 3 — Install system-level dependencies

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    python3-tk \
    v4l-utils
```

### 4 — Verify camera is visible

```bash
v4l2-ctl --list-devices
# Expected: /dev/video0 (or /dev/video1)
```

### 5 — Create Python virtual environment

```bash
cd ~
python3 -m venv venv_attendance
source venv_attendance/bin/activate
```

### 6 — Install Python packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **On Raspberry Pi 4 (ARMv7/ARMv8)** use the pre-built wheels:
>
> ```bash
> pip install opencv-contrib-python==4.8.0.76
> pip install mediapipe-rpi4          # community ARM wheel
> # or build from source if the above is unavailable
> ```
>
> MediaPipe's official ARM wheels:
> https://github.com/nicholasgasior/mediapipe-rpi

### 7 — (Optional) Enable GPIO feedback

Edit `config.py`:
```python
USE_GPIO     = True
GPIO_LED_PIN = 17    # BCM numbering
GPIO_BUZZER_PIN = 27
```

Wire up:
```
Pi Pin 11 (GPIO17) ── 220Ω ── LED (+) ── GND
Pi Pin 13 (GPIO27) ── NPN transistor base ── Buzzer
```

---

## Running the System

### CLI (recommended for Pi)

```bash
source ~/venv_attendance/bin/activate
cd ~/face_attendance
python main.py
```

```
╔══════════════════════════════════════════════════╗
║    FACE RECOGNITION ATTENDANCE SYSTEM  v1.0      ║
╚══════════════════════════════════════════════════╝

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
```

### GUI (requires display / VNC)

```bash
python gui.py
```

---

## Step-by-Step Usage

### Enroll a Student

1. Select **1. Enroll Student**
2. Enter Student ID (e.g. `S001`) and full name (e.g. `Jane Doe`)
3. A camera window opens — press **SPACE** to start capturing
4. Look directly at the camera; slowly turn head slightly left/right
5. 30 images are captured automatically
6. The student record is saved to SQLite; model will retrain on next run

Dataset folder created:
```
dataset/S001_Jane_Doe/
    20240101_120000_001.jpg  ...  (30 images)
```

### Take Attendance

1. Select **2. Take Attendance**
2. Camera window opens showing real-time detection
3. Student stands in front of camera
4. When recognised with confidence ≥ threshold for 5 consecutive frames,
   attendance is automatically marked in `attendance.csv`
5. A green overlay and optional LED/beep confirms the mark
6. Press **Q** to stop

### Keyboard Controls (in camera window)

| Key | Action |
|-----|--------|
| `q` | Quit recognition / enrollment |
| `s` | Save current frame as snapshot |
| `r` | Restart camera |
| `t` | Retrain model on-the-fly |
| `Space` | Start capture (enrollment only) |

---

## attendance.csv Format

```
student_id,student_name,date,time,status
S001,Jane Doe,2024-01-15,09:03:22,Present
S002,John Smith,2024-01-15,09:05:11,Present
```

---

## Tuning for Your Environment

Edit `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RECOGNITION_THRESHOLD` | 70 | LBPH distance; **lower = stricter**. Try 60–80. |
| `CONSECUTIVE_FRAMES` | 5 | Frames face must be stable before marking. |
| `ENROLL_IMAGES_COUNT` | 30 | More images → better accuracy, longer enrolment. |
| `MIN_DETECTION_CONFIDENCE` | 0.6 | MediaPipe sensitivity. |
| `FRAME_SCALE` | 0.5 | Downscale for faster detection. Lower = faster. |
| `ATTENDANCE_COOLDOWN_SECS` | 60 | Prevents same student re-marking within N seconds. |

---

## Performance on Raspberry Pi 4

| Setting | FPS (typical) |
|---------|--------------|
| 640×480, scale=0.5 | 12–18 FPS |
| 640×480, scale=0.25 | 18–25 FPS |
| 320×240, scale=0.5 | 22–30 FPS |

Tips:
- Close unused processes
- Use `FRAME_SCALE = 0.25` for maximum speed
- Run headless (no GUI) saves ~3 FPS

---

## Troubleshooting

**Camera not found**
```bash
ls /dev/video*          # check device exists
sudo usermod -aG video $USER   # add user to video group
```

**Low accuracy**
- Re-enroll in consistent lighting
- Increase `ENROLL_IMAGES_COUNT` to 50
- Lower `RECOGNITION_THRESHOLD` to 60

**MediaPipe import error on Pi**
```bash
pip install mediapipe-rpi4
# or
pip install mediapipe==0.9.3.0   # older version with ARM support
```

**cv2.face not found**
```bash
pip install opencv-contrib-python==4.8.0.76
```

---

## Auto-start on Boot (systemd)

```bash
sudo nano /etc/systemd/system/attendance.service
```

```ini
[Unit]
Description=Face Attendance System
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/face_attendance
ExecStart=/home/pi/venv_attendance/bin/python main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable attendance
sudo systemctl start attendance
```

---

## License
MIT — free to use in educational and non-commercial projects.
