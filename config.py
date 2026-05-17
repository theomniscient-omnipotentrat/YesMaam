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
# Direct ONNX file download (GitHub raw CDN — stable link)
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_FILE   = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THRESH = 0.70   # lower slightly for Pi camera quality
YUNET_NMS_THRESH   = 0.30
YUNET_TOP_K        = 5000

# ─────────────────────────────────────────────────────────────────
#  InsightFace / ArcFace embeddings
# ─────────────────────────────────────────────────────────────────
INSIGHTFACE_MODEL  = "buffalo_sc"   # buffalo_sc = fast CPU; buffalo_l = accurate
INSIGHTFACE_CTX_ID = -1             # -1 = CPU;  0 = first CUDA GPU

# L2 distance threshold for identity matching
# < 0.40 = very strict   |  0.50 = balanced (default)  |  > 0.60 = loose
EMBEDDING_THRESHOLD = 0.50

# ─────────────────────────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0
FRAME_WIDTH        = 640   # standard capture resolution
FRAME_HEIGHT       = 480   # standard capture resolution
TARGET_FPS         = 30
CAMERA_WARMUP_SECS = 0.5   # Pi 5 is fast; 0.5 s is enough

# Queue sizes for the Producer–Consumer pipeline
FRAME_QUEUE_SIZE   = 2   # raw frames: keep small so consumer gets fresh frames
DISPLAY_QUEUE_SIZE = 3   # annotated frames for the display thread
ATTEND_QUEUE_SIZE  = 8   # recognition data for the attendance worker

# ─────────────────────────────────────────────────────────────────
#  Enrollment
# ─────────────────────────────────────────────────────────────────
ENROLL_IMAGES_COUNT  = 3    # 3 high-quality ArcFace shots are sufficient
MIN_FACE_SIZE_PX     = 80   # reject faces narrower than this (px)
ENROLL_CAPTURE_DELAY = 0.4  # seconds between auto-captures

# ─────────────────────────────────────────────────────────────────
#  Attendance
# ─────────────────────────────────────────────────────────────────
ATTENDANCE_COOLDOWN_SECS = 30  # min gap before same student can be re-marked
CONSECUTIVE_FRAMES       = 4   # stable frames required before marking

# ─────────────────────────────────────────────────────────────────
#  Reports
# ─────────────────────────────────────────────────────────────────
REPORT_CSV_FILE        = os.path.join(REPORTS_DIR, "attendance_report.csv")
ABSENT_REPORT_CSV_FILE = os.path.join(REPORTS_DIR, "absent_report.csv")

# ─────────────────────────────────────────────────────────────────
#  Email  (absent report notifications)
# ─────────────────────────────────────────────────────────────────
EMAIL_ENABLED   = False
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 587
SMTP_USER       = "your@email.com"
SMTP_PASSWORD   = "your_app_password"   # Gmail App Password
EMAIL_RECIPIENT = "teacher@school.com"
EMAIL_SUBJECT   = "Absent Students Report"

# ─────────────────────────────────────────────────────────────────
#  Drawing / overlay  (cv2 integer constants — no cv2 import needed)
# ─────────────────────────────────────────────────────────────────
FONT             = 0       # cv2.FONT_HERSHEY_SIMPLEX = 0
FONT_SCALE_LARGE = 0.75
FONT_SCALE_SMALL = 0.50
THICKNESS        = 2

# BGR colour tuples
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 220, 0)
COLOR_ORANGE = (0, 165, 255)

# ─────────────────────────────────────────────────────────────────
#  GPIO  (Raspberry Pi only — set USE_GPIO = True on real hardware)
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
