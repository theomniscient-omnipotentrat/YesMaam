"""
config.py — Central configuration for the Face Recognition Attendance System.
All tuneable constants live here; no magic numbers elsewhere.
"""

import os
import logging

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
STUDENTS_DB     = os.path.join(BASE_DIR, "students.db")      # SQLite
MODEL_FILE      = os.path.join(BASE_DIR, "face_model.yml")   # LBPH model
LOG_FILE        = os.path.join(BASE_DIR, "system.log")
REPORT_DIR      = os.path.join(BASE_DIR, "reports")

# ─────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────
CAMERA_INDEX        = 0          # 0 = first camera (Pi Camera via v4l2)
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
FRAME_SCALE         = 0.5        # downscale for faster MediaPipe inference
TARGET_FPS          = 20
CAMERA_WARMUP_SECS  = 2          # seconds to let sensor settle

# ─────────────────────────────────────────────
#  Enrollment
# ─────────────────────────────────────────────
ENROLL_IMAGES_COUNT = 30         # images captured per student
ENROLL_DELAY_MS     = 200        # ms between captures
MIN_FACE_SIZE       = 80         # px — reject tiny detections during enroll

# ─────────────────────────────────────────────
#  Recognition
# ─────────────────────────────────────────────
RECOGNITION_THRESHOLD   = 70     # LBPH confidence: lower = stricter
                                 # (distance; 0 = perfect match)
MIN_DETECTION_CONFIDENCE = 0.6   # MediaPipe face-detection confidence
CONSECUTIVE_FRAMES      = 5      # frames face must be seen before marking

# ─────────────────────────────────────────────
#  Attendance
# ─────────────────────────────────────────────
ATTENDANCE_COOLDOWN_SECS = 60    # minimum gap before re-marking same student

# ─────────────────────────────────────────────
#  Display / UI
# ─────────────────────────────────────────────
FONT                = 0          # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE    = 0.8
FONT_SCALE_SMALL    = 0.55
THICKNESS           = 2
COLOR_GREEN         = (0, 255, 0)
COLOR_RED           = (0, 0, 255)
COLOR_YELLOW        = (0, 220, 255)
COLOR_WHITE         = (255, 255, 255)
COLOR_BLACK         = (0, 0, 0)
COLOR_CYAN          = (255, 220, 0)

# ─────────────────────────────────────────────
#  GPIO (Raspberry Pi) — set USE_GPIO = False on non-Pi hardware
# ─────────────────────────────────────────────
USE_GPIO        = False   # flip to True when running on real Pi
GPIO_LED_PIN    = 17      # BCM pin — green LED (recognised)
GPIO_BUZZER_PIN = 27      # BCM pin — short beep on success

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
LOG_LEVEL   = logging.DEBUG
LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
