"""
utils.py — Shared utility functions for Face Attendance System v4.

Provides
--------
• Logging setup (call setup_logging() once at startup)
• Date / time helpers
• Directory helpers
• OpenCV drawing helpers  ← moved here from detector.py so all modules share them
• GPIO feedback (Raspberry Pi)
• Student dataset directory helpers
"""

import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

import config

# ─────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure root logger to write to file and console. Call once at startup."""
    os.makedirs(os.path.dirname(config.LOG_FILE) or ".", exist_ok=True)
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFMT,
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ─────────────────────────────────────────────────────────────────
#  Date / time
# ─────────────────────────────────────────────────────────────────

def current_date() -> str:
    """Return today as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def current_time() -> str:
    """Return current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def current_datetime() -> str:
    """Return full timestamp as YYYY-MM-DD HH:MM:SS."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_filename() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ─────────────────────────────────────────────────────────────────
#  Directory helpers
# ─────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory and all parents if they don't exist."""
    os.makedirs(path, exist_ok=True)


def student_image_dir(student_id: str, student_name: str) -> str:
    """Return (and create) the dataset sub-folder for one student."""
    folder = f"{student_id}_{student_name.replace(' ', '_')}"
    path   = os.path.join(config.DATASET_DIR, folder)
    ensure_dir(path)
    return path


def list_student_dirs() -> list:
    """
    Scan DATASET_DIR and return [(student_id, student_name, dir_path), …].
    Folder name format: <id>_<name_with_underscores>
    """
    if not os.path.isdir(config.DATASET_DIR):
        return []
    results = []
    for entry in sorted(os.listdir(config.DATASET_DIR)):
        full = os.path.join(config.DATASET_DIR, entry)
        if os.path.isdir(full) and "_" in entry:
            sid, *name_parts = entry.split("_")
            sname = " ".join(name_parts)
            results.append((sid, sname, full))
    return results


# ─────────────────────────────────────────────────────────────────
#  OpenCV drawing helpers  (shared by detector.py, camera.py, enroll.py)
# ─────────────────────────────────────────────────────────────────

def put_text_bg(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = config.COLOR_WHITE,
    scale: float = config.FONT_SCALE_SMALL,
    thickness: int = 1,
) -> None:
    """
    Draw text with a solid black background rectangle for readability.

    Parameters
    ----------
    frame     : BGR image to draw on (in-place)
    text      : string to render
    pos       : (x, y) bottom-left corner of the text
    color     : BGR foreground colour
    scale     : font scale factor
    thickness : text stroke thickness
    """
    font = config.FONT
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    # Background rectangle
    cv2.rectangle(
        frame,
        (x, y - th - baseline - 2),
        (x + tw + 2, y + baseline + 2),
        config.COLOR_BLACK,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_bbox(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str,
    color: Tuple[int, int, int] = config.COLOR_GREEN,
) -> None:
    """Draw a labelled bounding box on `frame` in-place."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.THICKNESS)
    put_text_bg(frame, label, (x, y - 6), color)


def draw_hud(frame: np.ndarray, fps: float, n_faces: int, extra: str = "") -> None:
    """Draw the standard FPS / face-count HUD on a frame."""
    h = frame.shape[0]
    put_text_bg(frame, f"FPS: {fps:.1f}", (10, 25), config.COLOR_CYAN, 0.5)
    put_text_bg(frame, f"Faces: {n_faces}", (10, 48), config.COLOR_WHITE, 0.5)
    hint = extra or "[Q] quit  [S] snapshot  |  v4"
    put_text_bg(frame, hint, (10, h - 12), config.COLOR_WHITE, 0.40)


# ─────────────────────────────────────────────────────────────────
#  GPIO feedback  (Raspberry Pi only)
# ─────────────────────────────────────────────────────────────────

_gpio_ready = False
_gpio_logger = logging.getLogger("attendance_system.gpio")


def gpio_setup() -> bool:
    """Initialise GPIO pins. Returns True on success, False if unavailable."""
    global _gpio_ready
    if not config.USE_GPIO or _gpio_ready:
        return _gpio_ready
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(config.GPIO_LED_PIN,    GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(config.GPIO_BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
        _gpio_ready = True
        _gpio_logger.info("GPIO ready (LED=%d, BUZZER=%d)",
                          config.GPIO_LED_PIN, config.GPIO_BUZZER_PIN)
    except ImportError:
        _gpio_logger.warning("RPi.GPIO not available — GPIO disabled.")
    return _gpio_ready


def gpio_pulse() -> None:
    """Short LED + buzzer pulse to signal a successful attendance mark."""
    if not (config.USE_GPIO and _gpio_ready):
        return
    import threading, time
    try:
        import RPi.GPIO as GPIO  # type: ignore
        def _pulse():
            GPIO.output(config.GPIO_LED_PIN,    GPIO.HIGH)
            GPIO.output(config.GPIO_BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.15)
            GPIO.output(config.GPIO_LED_PIN,    GPIO.LOW)
            GPIO.output(config.GPIO_BUZZER_PIN, GPIO.LOW)
        threading.Thread(target=_pulse, daemon=True).start()
    except ImportError:
        pass


def gpio_cleanup() -> None:
    """Release GPIO resources on exit."""
    if not (config.USE_GPIO and _gpio_ready):
        return
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.cleanup()
        _gpio_logger.info("GPIO cleaned up.")
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────
#  FPS counter
# ─────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 20) -> None:
        self._times: list = []
        self._window = window
        self._prev: Optional[float] = None

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