"""
enroll.py — Student enrollment using YuNet detection + ArcFace embeddings.

Enrollment flow
---------------
1. Prompt for student ID + name (or accept as arguments)
2. Warmup InsightFace model (shows progress, avoids first-frame lag)
3. Open camera, show live YuNet preview
4. SPACE to start capture → captures ENROLL_IMAGES_COUNT face shots
5. For each: extract ArcFace embedding (padded crop)
6. Average all embeddings → unit-normalise → store in SQLite
7. Reload EmbeddingDatabase so recognition is immediately available

Why only 5 images?
-  ArcFace embeddings are robust; the mean of 3–10 shots is sufficient.
-  More images = more robustness to lighting, angle, glasses.
"""

import logging
import os
import time
from typing import List, Optional

import cv2
import numpy as np

import config
import database
import utils
import embeddings as emb_module
from detector import FaceDetector

logger = logging.getLogger("attendance_system.enroll")


def enroll_student(student_id: str = "", student_name: str = "") -> bool:
    """
    Interactive camera-based enrollment.

    Parameters
    ----------
    student_id   : pre-filled (will prompt if empty)
    student_name : pre-filled (will prompt if empty)

    Returns True on success, False on abort or failure.
    """
    # ── collect info ──────────────────────────────────────────────
    if not student_id:
        student_id = input("  Enter Student ID   : ").strip()
    if not student_id:
        print("  [ERROR] Student ID cannot be empty.")
        return False

    if database.student_exists(student_id):
        print(f"  [WARNING] Student '{student_id}' is already enrolled.")
        ans = input("  Re-enroll and overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            return False

    if not student_name:
        student_name = input("  Enter Student Name : ").strip()
    if not student_name:
        print("  [ERROR] Student name cannot be empty.")
        return False

    print(f"\n  Enrolling : {student_name}  (ID: {student_id})")
    print(f"  Captures  : {config.ENROLL_IMAGES_COUNT} face shots")
    print("  Press [SPACE] to start  |  [Q] to abort\n")

    # ── warmup InsightFace (download + load model if needed) ──────
    # Call warmup() explicitly — this loads the model before the camera
    # opens, giving the user clear progress feedback.
    print("  Warming up face embedding model …", end="", flush=True)
    try:
        emb_module.warmup()
        print(" done.")
    except ImportError as exc:
        print(f"\n  [ERROR] {exc}")
        return False

    # ── open camera ───────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera.")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    time.sleep(config.CAMERA_WARMUP_SECS)

    # ── prepare dataset folder ────────────────────────────────────
    save_dir = utils.student_image_dir(student_id, student_name)
    detector = FaceDetector()
    fps_ctr  = utils.FPSCounter()

    embeddings_list: List[np.ndarray] = []
    captured   = 0
    started    = False
    last_cap_t = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("enroll: frame read failed")
            time.sleep(0.05)
            continue

        fps     = fps_ctr.tick()
        display = frame.copy()
        h, w    = frame.shape[:2]

        # ── detect ───────────────────────────────────────────────
        faces = detector.detect(frame)

        face_found = False
        for det in faces[:1]:   # only use the largest face
            x, y, bw, bh = det["bbox"]

            if bw < config.MIN_FACE_SIZE_PX:
                utils.put_text_bg(display, "Move closer", (x, y - 6), config.COLOR_RED)
                continue

            face_found = True
            color = config.COLOR_GREEN if started else config.COLOR_YELLOW
            utils.draw_bbox(display, x, y, bw, bh,
                            f"conf={det['confidence']:.2f}", color)
            for lx, ly in det["landmarks"]:
                cv2.circle(display, (lx, ly), 2, config.COLOR_YELLOW, -1)

            # ── auto-capture ──────────────────────────────────────
            if started and (time.time() - last_cap_t) >= config.ENROLL_CAPTURE_DELAY:
                emb = emb_module.extract_embedding_from_frame(frame, det["bbox"])
                if emb is not None:
                    embeddings_list.append(emb)
                    captured  += 1
                    last_cap_t = time.time()
                    # Save raw image for audit trail
                    img_path = os.path.join(save_dir, f"{utils.timestamp_filename()}.jpg")
                    cv2.imwrite(img_path, frame[y: y + bh, x: x + bw])
                    logger.debug("Captured %d/%d", captured, config.ENROLL_IMAGES_COUNT)
                else:
                    utils.put_text_bg(display, "Embedding failed — hold still",
                                      (x, y + bh + 20), config.COLOR_ORANGE)

        # ── HUD ───────────────────────────────────────────────────
        utils.put_text_bg(display, f"FPS: {fps:.1f}", (10, 25), config.COLOR_CYAN, 0.5)

        if started:
            utils.put_text_bg(
                display,
                f"Captured: {captured} / {config.ENROLL_IMAGES_COUNT}",
                (10, 50), config.COLOR_WHITE,
            )
        else:
            utils.put_text_bg(display, "SPACE to start  |  Q to abort",
                              (10, 50), config.COLOR_YELLOW)

        if started and not face_found:
            utils.put_text_bg(display, "No face — look at camera",
                              (10, 78), config.COLOR_RED)

        # Progress bar
        if config.ENROLL_IMAGES_COUNT > 0:
            bar = int((captured / config.ENROLL_IMAGES_COUNT) * (w - 20))
            cv2.rectangle(display, (10, h - 18), (w - 10, h - 6), config.COLOR_WHITE, 1)
            if bar > 0:
                cv2.rectangle(display, (10, h - 18), (10 + bar, h - 6),
                              config.COLOR_GREEN, cv2.FILLED)

        cv2.imshow(f"Enrolling — {student_name}", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("\n  Enrollment aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord(" ") and not started:
            started    = True
            last_cap_t = 0.0
            print("  Capturing …")

        if captured >= config.ENROLL_IMAGES_COUNT:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── minimum viable check ─────────────────────────────────────
    min_ok = max(1, config.ENROLL_IMAGES_COUNT // 2)
    if len(embeddings_list) < min_ok:
        print(f"\n  [ERROR] Only {len(embeddings_list)} embeddings captured (need ≥ {min_ok}).")
        print("  Try again in better lighting.\n")
        return False

    # ── mean embedding ────────────────────────────────────────────
    stacked  = np.stack(embeddings_list, axis=0)          # (N, 512)
    mean_emb = np.mean(stacked, axis=0).astype(np.float32)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm

    # ── persist ───────────────────────────────────────────────────
    if database.student_exists(student_id):
        database.update_student_embedding(student_id, mean_emb, len(embeddings_list))
    else:
        database.add_student(student_id, student_name, len(embeddings_list), mean_emb)

    # Reload the shared embedding DB immediately
    emb_module.get_embedding_db().reload()

    print(f"\n  ✓ Enrolled {student_name} ({student_id})")
    print(f"    Shots averaged : {len(embeddings_list)}")
    print(f"    Images saved   : {save_dir}\n")
    logger.info("Enrolled %s (%s) — %d shots", student_id, student_name, len(embeddings_list))
    return True