"""
config.py — Central configuration for Face Attendance System v4.

Rules
-----
• NO cv2 import here — config must be importable before OpenCV loads.
• NO side-effects — only constant definitions.
• All other modules import this first; keep it lightweight.
"""

import os
import logging

# ─────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOG_FILE    = os.path.join(BASE_DIR, "system.log")

DATABASE_FILE = os.path.join(BASE_DIR, "attendance_system.db")

# ─────────────────────────────────────────────────────────────────
#  YuNet face detector
# ─────────────────────────────────────────────────────────────────
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_FILE   = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THRESH = 0.70
YUNET_NMS_THRESH   = 0.30
YUNET_TOP_K        = 5000

# ─────────────────────────────────────────────────────────────────
#  InsightFace / ArcFace embeddings
# ─────────────────────────────────────────────────────────────────
INSIGHTFACE_MODEL  = "buffalo_sc"   # buffalo_sc = fast CPU; buffalo_l = accurate
INSIGHTFACE_CTX_ID = -1             # -1 = CPU (ARM NEON optimised); 0 = CUDA GPU

# L2 distance threshold for identity matching
EMBEDDING_THRESHOLD = 0.65

# ─────────────────────────────────────────────────────────────────
#  Camera
#  OPTIMIZED: standard 640×480 capture; fast 0.5 s warmup
# ─────────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0
FRAME_WIDTH        = 640   # standard capture resolution
FRAME_HEIGHT       = 480   # standard capture resolution
TARGET_FPS         = 30
CAMERA_WARMUP_SECS = 0.5   # reduced from 2 s — Pi 5 cam initialises faster

# Queue sizes for the Producer–Consumer pipeline
FRAME_QUEUE_SIZE   = 2
DISPLAY_QUEUE_SIZE = 3
ATTEND_QUEUE_SIZE  = 8

# ─────────────────────────────────────────────────────────────────
#  Enrollment
#  OPTIMIZED: 3 shots are sufficient for ArcFace mean embedding
# ─────────────────────────────────────────────────────────────────
ENROLL_IMAGES_COUNT  = 3     # reduced from 5 — ArcFace is robust enough
MIN_FACE_SIZE_PX     = 80
ENROLL_CAPTURE_DELAY = 0.4   # seconds between auto-captures

# ─────────────────────────────────────────────────────────────────
#  Attendance
# ─────────────────────────────────────────────────────────────────
ATTENDANCE_COOLDOWN_SECS = 30
CONSECUTIVE_FRAMES       = 4

# ─────────────────────────────────────────────────────────────────
#  Recognition throttle
#  Run ArcFace only once every N ms to cut CPU load ~90 %
# ─────────────────────────────────────────────────────────────────
RECOGNITION_THROTTLE_MS = 300   # ms between full embedding extractions per face

# ─────────────────────────────────────────────────────────────────
#  Reports
# ─────────────────────────────────────────────────────────────────
REPORT_CSV_FILE        = os.path.join(REPORTS_DIR, "attendance_report.csv")
ABSENT_REPORT_CSV_FILE = os.path.join(REPORTS_DIR, "absent_report.csv")
CAMPUS_MONITORING_REPORT_CSV_FILE = os.path.join(REPORTS_DIR, "campus_monitoring_report.csv")

# ─────────────────────────────────────────────────────────────────
#  Email
# ─────────────────────────────────────────────────────────────────
EMAIL_ENABLED   = False
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
SMTP_USER       = "your@email.com"
SMTP_PASSWORD   = "your_app_password"
EMAIL_RECIPIENT = "teacher@school.com"
EMAIL_SUBJECT   = "Absent Students Report"

# ─────────────────────────────────────────────────────────────────
#  Drawing / overlay
# ─────────────────────────────────────────────────────────────────
FONT             = 0       # cv2.FONT_HERSHEY_SIMPLEX = 0
FONT_SCALE_LARGE = 0.75
FONT_SCALE_SMALL = 0.50
THICKNESS        = 2

COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 220, 0)
COLOR_ORANGE = (0, 165, 255)

# ─────────────────────────────────────────────────────────────────
#  GPIO  (Raspberry Pi only)
# ─────────────────────────────────────────────────────────────────
USE_GPIO        = False
GPIO_LED_PIN    = 17
GPIO_BUZZER_PIN = 27

# ─────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────
LOG_LEVEL   = logging.INFO
LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
