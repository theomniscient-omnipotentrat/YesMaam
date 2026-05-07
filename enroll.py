"""
enroll.py — Student enrollment using YuNet detection + ArcFace embeddings.

Pi 5 optimisations over v4
--------------------------
THREADED DISPLAY PIPELINE
    The capture loop now mirrors Recognition.py exactly: a lightweight
    AttendancePipeline-style structure separates camera I/O from the GUI.
    The main thread only pulls processed frames from a display_queue and
    calls cv2.imshow — it never blocks on inference.

cv2.startWindowThread()
    Called once at the top of enroll_student() to register an OpenCV
    window-management thread with the Wayland/X11 compositor.  Without it
    the imshow window hangs or never appears on Pi OS Bookworm (Wayland).

ENROLL_IMAGES_COUNT = 3
    ArcFace embeddings are 512-dimensional and highly discriminative.
    Three well-lit, slightly varied shots produce a mean embedding that is
    just as robust as ten shots while taking 70 % less time to capture.

Enrollment flow
---------------
1. Prompt for student ID + name (or accept as arguments)
2. Warmup InsightFace recognition model (clear progress feedback)
3. cv2.startWindowThread() — register with display server
4. Producer thread reads camera → display_queue (annotated) + capture_queue
5. Main thread: imshow from display_queue, SPACE to start capturing
6. Auto-capture ENROLL_IMAGES_COUNT embeddings via capture_queue
7. Average embeddings → unit-normalise → store in SQLite
8. Reload EmbeddingDatabase so recognition is immediately available
"""

import logging
import queue
import threading
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


# ─────────────────────────────────────────────────────────────────
#  Internal producer thread  (camera → annotated display + capture)
# ─────────────────────────────────────────────────────────────────

class _EnrollProducer(threading.Thread):
    """
    Reads raw frames, runs YuNet, annotates them, and pushes to two queues:

    display_queue  — annotated BGR frame for cv2.imshow (main thread)
    capture_queue  — (raw_frame, bbox) tuple for embedding extraction

    Keeps queues fresh by dropping the oldest item when full.
    """

    def __init__(
        self,
        display_queue: queue.Queue,
        capture_queue: queue.Queue,
        student_name:  str,
        camera_index:  int = config.CAMERA_INDEX,
    ) -> None:
        super().__init__(name="EnrollProducer", daemon=True)
        self.display_queue = display_queue
        self.capture_queue = capture_queue
        self.student_name  = student_name
        self.camera_index  = camera_index

        self.started   = False       # set True when user presses SPACE
        self.captured  = 0           # incremented by main thread
        self.stop_event     = threading.Event()
        self._fps      = utils.FPSCounter()

    def stop(self) -> None:
        self.stop_event.set()

    @staticmethod
    def _push(q: queue.Queue, item) -> None:
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("EnrollProducer: cannot open camera %d", self.camera_index)
            self._push(self.display_queue, None)   # sentinel
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        time.sleep(config.CAMERA_WARMUP_SECS)

        detector  = FaceDetector()
        win_title = f"Enrolling — {self.student_name}"

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            fps     = self._fps.tick()
            display = frame.copy()
            h, w    = frame.shape[:2]
            faces   = detector.detect(frame)

            face_found = False
            for det in faces[:1]:
                x, y, bw, bh = det["bbox"]
                if bw < config.MIN_FACE_SIZE_PX:
                    utils.put_text_bg(display, "Move closer",
                                      (x, y - 6), config.COLOR_RED)
                    continue

                face_found = True
                color = config.COLOR_GREEN if self.started else config.COLOR_YELLOW
                utils.draw_bbox(display, x, y, bw, bh,
                                f"conf={det['confidence']:.2f}", color)
                for lx, ly in det["landmarks"]:
                    cv2.circle(display, (lx, ly), 2, config.COLOR_YELLOW, -1)

                # Signal main thread to extract an embedding
                if self.started:
                    self._push(self.capture_queue, (frame.copy(), det["bbox"]))

            # ── HUD ──────────────────────────────────────────────
            utils.put_text_bg(display, f"FPS: {fps:.1f}",
                              (10, 25), config.COLOR_CYAN, 0.5)
            if self.started:
                utils.put_text_bg(
                    display,
                    f"Captured: {self.captured} / {config.ENROLL_IMAGES_COUNT}",
                    (10, 50), config.COLOR_WHITE,
                )
            else:
                utils.put_text_bg(display, "SPACE to start  |  Q to abort",
                                  (10, 50), config.COLOR_YELLOW)

            if self.started and not face_found:
                utils.put_text_bg(display, "No face — look at camera",
                                  (10, 78), config.COLOR_RED)

            # Progress bar
            if config.ENROLL_IMAGES_COUNT > 0:
                bar = int((self.captured / config.ENROLL_IMAGES_COUNT) * (w - 20))
                cv2.rectangle(display, (10, h - 18), (w - 10, h - 6),
                              config.COLOR_WHITE, 1)
                if bar > 0:
                    cv2.rectangle(display, (10, h - 18), (10 + bar, h - 6),
                                  config.COLOR_GREEN, cv2.FILLED)

            self._push(self.display_queue, display)

        cap.release()
        # Sentinel so main thread knows to exit
        try:
            self.display_queue.put(None, timeout=1)
        except queue.Full:
            pass
        logger.info("EnrollProducer stopped.")


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────

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

    # ── warmup InsightFace ────────────────────────────────────────
    print("  Warming up face embedding model …", end="", flush=True)
    try:
        emb_module.warmup()
        print(" done.")
    except ImportError as exc:
        print(f"\n  [ERROR] {exc}")
        return False

    # ── register display thread with Wayland/X11 compositor ──────
    # Without this call the imshow window hangs on Pi OS Bookworm (Wayland).
    cv2.startWindowThread()

    # ── set up queues + producer thread ──────────────────────────
    display_queue: queue.Queue = queue.Queue(maxsize=config.DISPLAY_QUEUE_SIZE)
    capture_queue: queue.Queue = queue.Queue(maxsize=8)

    save_dir = utils.student_image_dir(student_id, student_name)

    producer = _EnrollProducer(
        display_queue, capture_queue, student_name, config.CAMERA_INDEX
    )
    producer.start()

    embeddings_list: List[np.ndarray] = []
    last_cap_t = 0.0
    aborted    = False
    win_title  = f"Enrolling — {student_name}"

    # ── main display loop (GUI thread only) ──────────────────────
    while True:
        try:
            frame = display_queue.get(timeout=0.1)
        except queue.Empty:
            if not producer.is_alive():
                break
            frame = None

        if frame is None:
            # Sentinel or producer died
            break

        cv2.imshow(win_title, frame)

        # ── drain capture_queue and extract embeddings ────────────
        while True:
            try:
                raw_frame, bbox = capture_queue.get_nowait()
            except queue.Empty:
                break

            now = time.time()
            if (now - last_cap_t) < config.ENROLL_CAPTURE_DELAY:
                continue
            if len(embeddings_list) >= config.ENROLL_IMAGES_COUNT:
                break

            emb = emb_module.extract_embedding_from_frame(raw_frame, bbox)
            if emb is not None:
                embeddings_list.append(emb)
                producer.captured = len(embeddings_list)
                last_cap_t = now

                # Save raw face crop for audit trail
                x, y, bw, bh = bbox
                img_path = f"{save_dir}/{utils.timestamp_filename()}.jpg"
                cv2.imwrite(img_path, raw_frame[y: y + bh, x: x + bw])
                logger.debug("Captured %d/%d", len(embeddings_list),
                             config.ENROLL_IMAGES_COUNT)

        # Done capturing?
        if len(embeddings_list) >= config.ENROLL_IMAGES_COUNT:
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("\n  Enrollment aborted.")
            aborted = True
            break
        elif key == ord(" ") and not producer.started:
            producer.started = True
            last_cap_t = 0.0
            print("  Capturing …")

    # ── cleanup ───────────────────────────────────────────────────
    producer.stop()
    producer.join(timeout=5)
    cv2.destroyAllWindows()

    if aborted:
        return False

    # ── minimum viable check ─────────────────────────────────────
    min_ok = max(1, config.ENROLL_IMAGES_COUNT // 2)
    if len(embeddings_list) < min_ok:
        print(f"\n  [ERROR] Only {len(embeddings_list)} embeddings captured "
              f"(need ≥ {min_ok}).")
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

    emb_module.get_embedding_db().reload()

    print(f"\n  ✓ Enrolled {student_name} ({student_id})")
    print(f"    Shots averaged : {len(embeddings_list)}")
    print(f"    Images saved   : {save_dir}\n")
    logger.info("Enrolled %s (%s) — %d shots",
                student_id, student_name, len(embeddings_list))
    return True
