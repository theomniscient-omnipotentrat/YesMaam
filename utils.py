"""
utils.py — Shared helper functions: logging setup, date/time, file I/O,
           GPIO feedback, and frame annotation utilities.
"""

import os
import cv2
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

import config


# ──────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure root logger to write to both console and log file."""
    os.makedirs(os.path.dirname(config.LOG_FILE) or ".", exist_ok=True)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFMT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("attendance_system")


logger = setup_logging()


# ──────────────────────────────────────────────────────────────────
#  Date / Time helpers
# ──────────────────────────────────────────────────────────────────

def current_date() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def current_time() -> str:
    """Return current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def current_datetime() -> str:
    """Return full timestamp as YYYY-MM-DD HH:MM:SS."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_for_filename() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ──────────────────────────────────────────────────────────────────
#  File / directory helpers
# ──────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def student_image_dir(student_id: str, student_name: str) -> str:
    """Return and create the dataset folder for a student."""
    folder_name = f"{student_id}_{student_name.replace(' ', '_')}"
    path = os.path.join(config.DATASET_DIR, folder_name)
    ensure_dir(path)
    return path


def list_student_dirs() -> list:
    """Return list of (student_id, student_name, dir_path) tuples."""
    if not os.path.isdir(config.DATASET_DIR):
        return []
    entries = []
    for entry in sorted(os.listdir(config.DATASET_DIR)):
        full = os.path.join(config.DATASET_DIR, entry)
        if os.path.isdir(full) and "_" in entry:
            sid, *name_parts = entry.split("_")
            sname = " ".join(name_parts)
            entries.append((sid, sname, full))
    return entries


def count_images_in_dir(directory: str) -> int:
    """Count .jpg / .png images in a directory."""
    exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )


# ──────────────────────────────────────────────────────────────────
#  Camera helpers
# ──────────────────────────────────────────────────────────────────

def open_camera(index: int = config.CAMERA_INDEX) -> Optional[cv2.VideoCapture]:
    """Open camera, set resolution, return cap or None on failure."""
    import time
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", index)
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
    time.sleep(config.CAMERA_WARMUP_SECS)
    logger.info("Camera %d opened (%dx%d)", index, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    return cap


def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
    """Safely release camera and destroy windows."""
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera released.")


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Read one frame; return (success, frame)."""
    ret, frame = cap.read()
    if not ret:
        logger.warning("Failed to read frame from camera.")
    return ret, frame


# ──────────────────────────────────────────────────────────────────
#  Frame processing
# ──────────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, scale: float = config.FRAME_SCALE) -> np.ndarray:
    """Downscale frame for faster processing."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS counter (top-left)."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 25),
        config.FONT, config.FONT_SCALE_SMALL,
        config.COLOR_CYAN, 1, cv2.LINE_AA,
    )


def draw_label(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = config.COLOR_GREEN,
    scale: float = config.FONT_SCALE_SMALL,
) -> None:
    """Draw text with a dark background for readability."""
    font, thickness = config.FONT, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw, y + baseline), config.COLOR_BLACK, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_bounding_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int] = config.COLOR_GREEN,
) -> None:
    """Draw a bounding box + label on frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.THICKNESS)
    draw_label(frame, label, (x1, y1 - 5), color)


# ──────────────────────────────────────────────────────────────────
#  GPIO feedback (optional — Raspberry Pi only)
# ──────────────────────────────────────────────────────────────────

_gpio_initialised = False


def gpio_setup() -> bool:
    """Initialise GPIO pins. Returns True on success."""
    global _gpio_initialised
    if not config.USE_GPIO or _gpio_initialised:
        return _gpio_initialised
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(config.GPIO_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(config.GPIO_BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
        _gpio_initialised = True
        logger.info("GPIO initialised (LED=%d, BUZZER=%d)", config.GPIO_LED_PIN, config.GPIO_BUZZER_PIN)
    except ImportError:
        logger.warning("RPi.GPIO not available — GPIO feedback disabled.")
    return _gpio_initialised


def gpio_feedback_success() -> None:
    """Blink LED + short beep to signal successful recognition."""
    if not (config.USE_GPIO and _gpio_initialised):
        return
    import threading, time
    import RPi.GPIO as GPIO  # type: ignore

    def _pulse():
        GPIO.output(config.GPIO_LED_PIN, GPIO.HIGH)
        GPIO.output(config.GPIO_BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.15)
        GPIO.output(config.GPIO_LED_PIN, GPIO.LOW)
        GPIO.output(config.GPIO_BUZZER_PIN, GPIO.LOW)

    threading.Thread(target=_pulse, daemon=True).start()


def gpio_cleanup() -> None:
    """Clean up GPIO on exit."""
    if not (config.USE_GPIO and _gpio_initialised):
        return
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.cleanup()
        logger.info("GPIO cleaned up.")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────
#  FPS tracker
# ──────────────────────────────────────────────────────────────────

class FPSCounter:
    """Lightweight rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self._times: list = []
        self._window = window
        self._prev = None

    def tick(self) -> float:
        import time
        now = time.perf_counter()
        if self._prev is not None:
            self._times.append(now - self._prev)
            if len(self._times) > self._window:
                self._times.pop(0)
        self._prev = now
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))
