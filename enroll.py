"""
enroll.py — Capture student face images using the Pi/webcam camera
            and OpenCV Haar Cascade face detection.

Works with any OpenCV version (4.x) — no MediaPipe dependency.
Saves grayscale, histogram-equalised face ROIs to:
    dataset/<student_id>_<student_name>/
"""

import os
import cv2
import time
import logging
import numpy as np

import config
import utils
import database

logger = logging.getLogger("attendance_system.enroll")

# ──────────────────────────────────────────────────────────────────
#  Load Haar Cascade (ships with every OpenCV installation)
# ──────────────────────────────────────────────────────────────────

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def _load_cascade() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            f"Could not load Haar cascade from:\n  {_CASCADE_PATH}\n"
            "Ensure opencv-contrib-python is installed correctly."
        )
    return cascade


# ──────────────────────────────────────────────────────────────────
#  Detect faces in a grayscale frame
# ──────────────────────────────────────────────────────────────────

def _detect_faces(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> list:
    """
    Return list of (x, y, w, h) bounding boxes.
    Tuned for Pi performance — scaleFactor and minNeighbors balance
    speed vs. false-positive rate.
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(config.MIN_FACE_SIZE, config.MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    # detectMultiScale returns () when nothing found — normalise to list
    return list(faces) if len(faces) > 0 else []


# ──────────────────────────────────────────────────────────────────
#  Enrollment session
# ──────────────────────────────────────────────────────────────────

def enroll_student(student_id: str = "", student_name: str = "") -> bool:
    """
    Interactive enrollment:
      1. Prompt for student ID + name (or accept as parameters).
      2. Open camera.
      3. Capture ENROLL_IMAGES_COUNT face images.
      4. Save grayscale ROIs to dataset/<id>_<name>/.
      5. Persist student record in SQLite.

    Keyboard controls
    -----------------
    SPACE  — start auto-capture
    Q/ESC  — abort
    """
    # ── collect student info ──────────────────────────────────────
    if not student_id:
        student_id = input("  Enter Student ID   : ").strip()
    if not student_id:
        print("  [ERROR] Student ID cannot be empty.")
        return False

    if database.student_exists(student_id):
        print(f"  [WARNING] Student ID '{student_id}' already enrolled.")
        overwrite = input("  Re-enroll and overwrite images? [y/N]: ").strip().lower()
        if overwrite != "y":
            return False

    if not student_name:
        student_name = input("  Enter Student Name : ").strip()
    if not student_name:
        print("  [ERROR] Student name cannot be empty.")
        return False

    print(f"\n  Enrolling : {student_name}  (ID: {student_id})")
    print(f"  Target    : {config.ENROLL_IMAGES_COUNT} face images")
    print("  Press [SPACE] to start  |  [Q] to abort\n")

    # ── setup ─────────────────────────────────────────────────────
    cascade  = _load_cascade()
    cap      = utils.open_camera()
    if cap is None:
        print("  [ERROR] Cannot access camera.")
        return False

    save_dir        = utils.student_image_dir(student_id, student_name)
    fps_counter     = utils.FPSCounter()
    captured        = 0
    last_capture_ts = 0.0
    started         = False

    while True:
        ret, frame = utils.read_frame(cap)
        if not ret:
            break

        fps     = fps_counter.tick()
        display = frame.copy()
        h, w    = frame.shape[:2]

        # ── detect faces (operate on small grey frame) ────────────
        small = utils.resize_frame(frame, config.FRAME_SCALE)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = _detect_faces(gray, cascade)

        face_found = False
        inv_scale  = 1.0 / config.FRAME_SCALE   # map coords back to original

        for (fx, fy, fw, fh) in faces:
            # Scale back to original frame coordinates
            x1 = int(fx * inv_scale)
            y1 = int(fy * inv_scale)
            x2 = int((fx + fw) * inv_scale)
            y2 = int((fy + fh) * inv_scale)

            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            face_found = True
            color = config.COLOR_GREEN if started else config.COLOR_YELLOW
            utils.draw_bounding_box(display, x1, y1, x2, y2, "Face detected", color)

            # ── auto-capture when running ─────────────────────────
            now = time.time()
            delay_secs = config.ENROLL_DELAY_MS / 1000.0
            if started and (now - last_capture_ts) >= delay_secs:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.equalizeHist(gray_roi)
                img_name  = f"{utils.timestamp_for_filename()}.jpg"
                save_path = os.path.join(save_dir, img_name)
                cv2.imwrite(save_path, gray_roi)
                captured        += 1
                last_capture_ts  = now
                logger.debug("Saved image %d → %s", captured, img_name)

            break   # use only the first (largest) detected face

        # ── HUD overlay ───────────────────────────────────────────
        utils.draw_fps(display, fps)

        if started:
            status = f"Captured: {captured} / {config.ENROLL_IMAGES_COUNT}"
            utils.draw_label(display, status, (10, 50), config.COLOR_WHITE)
        else:
            utils.draw_label(display, "Press SPACE to start capturing", (10, 50), config.COLOR_YELLOW)

        if started and not face_found:
            utils.draw_label(display, "No face detected — move closer", (10, 78), config.COLOR_RED)

        # Progress bar along bottom edge
        if config.ENROLL_IMAGES_COUNT > 0:
            bar_w = int((captured / config.ENROLL_IMAGES_COUNT) * (w - 20))
            cv2.rectangle(display, (10, h - 18), (10 + bar_w, h - 6),
                          config.COLOR_GREEN, cv2.FILLED)
            cv2.rectangle(display, (10, h - 18), (w - 10, h - 6),
                          config.COLOR_WHITE, 1)

        cv2.imshow(f"Enroll — {student_name}", display)

        # ── keyboard ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):           # Q or ESC → abort
            print("\n  Enrollment aborted.")
            utils.release_camera(cap)
            return False
        elif key == ord(" ") and not started:
            started = True
            print("  Capturing …")

        if captured >= config.ENROLL_IMAGES_COUNT:
            break

    utils.release_camera(cap)

    # ── minimum viable capture check ─────────────────────────────
    min_required = config.ENROLL_IMAGES_COUNT // 2
    if captured < min_required:
        print(f"\n  [ERROR] Only {captured} images captured (minimum: {min_required}).")
        print("  Enrollment failed — please try again in better lighting.")
        return False

    # ── persist to SQLite ─────────────────────────────────────────
    if database.student_exists(student_id):
        database.update_image_count(student_id, captured)
    else:
        database.add_student(student_id, student_name, captured)

    print(f"\n  ✓ Enrollment complete — {captured} images saved to:")
    print(f"    {save_dir}\n")
    logger.info("Enrolled %s (%s): %d images → %s", student_id, student_name, captured, save_dir)
    return True
