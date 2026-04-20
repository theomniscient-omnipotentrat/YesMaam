"""
recognize.py — Real-time face recognition + attendance marking.

Detection  : OpenCV Haar Cascade (built into every OpenCV install)
Recognition: OpenCV LBPH Face Recognizer (from opencv-contrib-python)
Attendance : attendance.mark_attendance() → CSV + duplicate guard

Pipeline
--------
Frame → grayscale → Haar cascade → face ROI(s)
      → LBPH predict → confidence filter
      → consecutive-frame vote buffer → stable identity
      → attendance.mark_attendance()

Keyboard controls (camera window)
----------------------------------
Q / ESC  quit
S        save snapshot
R        restart camera
T        retrain model on-the-fly
"""

import os
import cv2
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

import config
import utils
import database
import attendance

logger = logging.getLogger("attendance_system.recognize")

# ──────────────────────────────────────────────────────────────────
#  Haar Cascade (ships with every OpenCV build — no download needed)
# ──────────────────────────────────────────────────────────────────

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _load_cascade() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(f"Cannot load Haar cascade: {_CASCADE_PATH}")
    return cascade


def _detect_faces(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> list:
    """Return list of (x, y, w, h); empty list when nothing found."""
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(config.MIN_FACE_SIZE, config.MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return list(faces) if len(faces) > 0 else []


# ──────────────────────────────────────────────────────────────────
#  Label ↔ integer mapping  (LBPH needs integer labels)
# ──────────────────────────────────────────────────────────────────

def _build_label_map() -> Tuple[Dict[int, Tuple[str, str]], Dict[str, int]]:
    """
    Scan dataset directory and return:
      int_to_student : {label_int: (student_id, student_name)}
      student_to_int : {student_id: label_int}
    """
    int_to_student: Dict[int, Tuple[str, str]] = {}
    student_to_int: Dict[str, int] = {}
    for idx, (sid, sname, _) in enumerate(utils.list_student_dirs()):
        int_to_student[idx] = (sid, sname)
        student_to_int[sid] = idx
    return int_to_student, student_to_int


# ──────────────────────────────────────────────────────────────────
#  LBPH model: train / load
# ──────────────────────────────────────────────────────────────────

def train_model(force: bool = False) -> Optional[cv2.face.LBPHFaceRecognizer]:
    """
    Train an LBPH recognizer from dataset images and save to disk.
    If MODEL_FILE already exists and force=False, load it directly.

    Returns the trained recognizer, or None if no data is available.
    """
    student_dirs = utils.list_student_dirs()
    if not student_dirs:
        logger.warning("No student data found — cannot train.")
        return None

    # ── load cached model ─────────────────────────────────────────
    if not force and os.path.isfile(config.MODEL_FILE):
        logger.info("Loading cached model from %s", config.MODEL_FILE)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(config.MODEL_FILE)
        return recognizer

    # ── fresh training ────────────────────────────────────────────
    print("  Training recognition model …", end="", flush=True)
    _, student_to_int = _build_label_map()
    faces:  List[np.ndarray] = []
    labels: List[int]        = []

    for sid, sname, dir_path in student_dirs:
        label = student_to_int.get(sid)
        if label is None:
            continue
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.equalizeHist(img)
            img = cv2.resize(img, (100, 100))
            faces.append(img)
            labels.append(label)

    if not faces:
        print(" FAILED (no readable images in dataset)")
        logger.error("Training failed — no valid images.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))
    recognizer.save(config.MODEL_FILE)

    print(f" done  ({len(faces)} images, {len(student_dirs)} student(s))")
    logger.info("Model saved → %s  [%d images, %d students]",
                config.MODEL_FILE, len(faces), len(student_dirs))
    return recognizer


# ──────────────────────────────────────────────────────────────────
#  Per-ROI recognition
# ──────────────────────────────────────────────────────────────────

def _recognise_roi(
    gray_roi: np.ndarray,
    recognizer: cv2.face.LBPHFaceRecognizer,
    int_to_student: Dict[int, Tuple[str, str]],
) -> Tuple[str, str, float]:
    """
    Predict identity from a single grayscale face ROI.

    Returns
    -------
    (student_id, student_name, confidence)
    confidence = LBPH distance; lower is better (0 = perfect).
    Returns ("Unknown", "Unknown", 9999.0) when above threshold.
    """
    roi_eq     = cv2.equalizeHist(cv2.resize(gray_roi, (100, 100)))
    label, conf = recognizer.predict(roi_eq)

    if conf > config.RECOGNITION_THRESHOLD:
        return "Unknown", "Unknown", conf

    student = int_to_student.get(label, ("Unknown", "Unknown"))
    return student[0], student[1], conf


# ──────────────────────────────────────────────────────────────────
#  Main recognition session
# ──────────────────────────────────────────────────────────────────

def run_recognition() -> None:
    """
    Open the camera, detect and recognise faces in real time,
    and mark attendance automatically.
    """
    # ── load / train ──────────────────────────────────────────────
    recognizer = train_model()
    if recognizer is None:
        print("\n  [ERROR] No trained model — enroll at least one student first.\n")
        return

    int_to_student, _ = _build_label_map()
    cascade            = _load_cascade()
    utils.gpio_setup()

    cap = utils.open_camera()
    if cap is None:
        print("  [ERROR] Cannot access camera.")
        return

    fps_counter = utils.FPSCounter()

    # Consecutive-frame vote buffer  { grid_cell_key : [student_id, …] }
    vote_buffer: Dict[str, List[str]] = {}

    # Per-student cooldown           { student_id : last_marked_unix_ts }
    last_marked: Dict[str, float] = {}

    print("\n  Recognition running …")
    print("  Controls: [Q] quit   [S] snapshot   [R] restart camera   [T] retrain\n")

    while True:
        ret, frame = utils.read_frame(cap)
        if not ret:
            logger.warning("Frame read failed — restarting camera …")
            utils.release_camera(cap)
            time.sleep(1)
            cap = utils.open_camera()
            if cap is None:
                break
            continue

        fps     = fps_counter.tick()
        h, w    = frame.shape[:2]
        display = frame.copy()

        # ── detect on downscaled greyscale frame ──────────────────
        small    = utils.resize_frame(frame, config.FRAME_SCALE)
        gray_s   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_s   = cv2.equalizeHist(gray_s)
        faces_s  = _detect_faces(gray_s, cascade)

        inv      = 1.0 / config.FRAME_SCALE   # coord scale-back factor
        curr_keys = set()

        for (fx, fy, fw, fh) in faces_s:
            # ── map back to original frame ─────────────────────────
            x1 = max(0, int(fx * inv))
            y1 = max(0, int(fy * inv))
            x2 = min(w, int((fx + fw) * inv))
            y2 = min(h, int((fy + fh) * inv))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # ── vote-buffer key (coarse grid cell) ────────────────
            vkey = f"{x1 // 80}:{y1 // 80}"
            curr_keys.add(vkey)

            # ── recognise ─────────────────────────────────────────
            gray_roi              = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sid, sname, conf      = _recognise_roi(gray_roi, recognizer, int_to_student)

            # ── update vote buffer ────────────────────────────────
            buf = vote_buffer.setdefault(vkey, [])
            buf.append(sid)
            if len(buf) > config.CONSECUTIVE_FRAMES:
                buf.pop(0)

            # majority vote
            voted_id       = max(set(buf), key=buf.count) if buf else "Unknown"
            vote_conf_pct  = buf.count(voted_id) / max(len(buf), 1)

            # ── choose display colour ─────────────────────────────
            if voted_id == "Unknown" or vote_conf_pct < 0.6:
                box_color  = config.COLOR_RED
                label_text = f"Unknown  [{conf:.1f}]"
            else:
                box_color  = config.COLOR_GREEN
                label_text = f"{voted_id}: {sname}  [{conf:.1f}]"

            utils.draw_bounding_box(display, x1, y1, x2, y2, label_text, box_color)

            # ── mark attendance ───────────────────────────────────
            now = time.time()
            cooldown_ok = (now - last_marked.get(voted_id, 0)) > config.ATTENDANCE_COOLDOWN_SECS

            if voted_id != "Unknown" and vote_conf_pct >= 0.8 and cooldown_ok:
                marked = attendance.mark_attendance(voted_id, sname)
                if marked:
                    last_marked[voted_id] = now
                    utils.gpio_feedback_success()
                    utils.draw_label(
                        display,
                        f"✓  Attendance marked: {sname}",
                        (10, h - 50),
                        config.COLOR_GREEN,
                        config.FONT_SCALE_LARGE,
                    )
                    logger.info("Marked attendance — %s (%s)  conf=%.1f", voted_id, sname, conf)

        # ── remove stale vote cells ───────────────────────────────
        for k in list(vote_buffer.keys()):
            if k not in curr_keys:
                del vote_buffer[k]

        # ── HUD ───────────────────────────────────────────────────
        utils.draw_fps(display, fps)
        utils.draw_label(
            display,
            "Face Recognition  [Q]quit [S]snap [R]restart [T]retrain",
            (10, h - 15), config.COLOR_WHITE, 0.42,
        )
        utils.draw_label(
            display,
            f"Students: {database.get_student_count()}",
            (w - 140, 25), config.COLOR_CYAN,
        )

        cv2.imshow("Attendance System — Recognition", display)

        # ── keyboard ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):           # Q / ESC → quit
            break

        elif key == ord("s"):               # S → save snapshot
            snap = os.path.join(
                config.BASE_DIR,
                f"snapshot_{utils.timestamp_for_filename()}.jpg"
            )
            cv2.imwrite(snap, display)
            print(f"  Snapshot saved → {snap}")

        elif key == ord("r"):               # R → restart camera
            print("  Restarting camera …")
            utils.release_camera(cap)
            cap = utils.open_camera()
            if cap is None:
                break

        elif key == ord("t"):               # T → retrain model
            print("  Retraining model …")
            utils.release_camera(cap)
            if os.path.isfile(config.MODEL_FILE):
                os.remove(config.MODEL_FILE)
            recognizer     = train_model(force=True)
            int_to_student, _ = _build_label_map()
            cap = utils.open_camera()
            if cap is None or recognizer is None:
                break

    utils.release_camera(cap)
    utils.gpio_cleanup()
    print("\n  Recognition session ended.\n")
