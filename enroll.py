"""
enroll.py — Student enrollment using YuNet detection + ArcFace embeddings.

Pi 5 Optimization Applied
--------------------------
The original enroll_student() was a single-threaded blocking loop that ran
detection + embedding extraction in the same thread as cv2.imshow().  On a
loaded Pi 5 this caused the window to hang for hundreds of milliseconds
between frames.

Refactored architecture (mirrors AttendancePipeline from camera.py):
  • Camera reading runs in a background thread (EnrollProducer).
  • Detection + embedding extraction runs in a separate process (EnrollConsumer)
    — real parallelism, same GIL-bypass benefit as the recognition pipeline.
  • The main thread ONLY calls cv2.imshow() and cv2.waitKey(), never blocks.
  • cv2.startWindowThread() is called once at the start to register the window
    with the Wayland/X11 display server before any heavy work begins.

Enrollment flow
---------------
1. Prompt for student ID + name (or accept as arguments).
2. Warmup InsightFace model (show progress, avoid first-frame lag).
3. Start EnrollProducer + EnrollConsumer.
4. Main loop: pull annotated frames from display_queue → cv2.imshow.
5. SPACE to start capture → EnrollConsumer collects ENROLL_IMAGES_COUNT shots.
6. Consumer averages embeddings → puts result dict on result_queue.
7. Main thread reads result, writes to SQLite, reloads EmbeddingDatabase.
"""

import logging
import multiprocessing
import os
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

logger = logging.getLogger("attendance_system.enroll")


# ─────────────────────────────────────────────────────────────────
#  EnrollProducer — camera thread (main process)
# ─────────────────────────────────────────────────────────────────

class _EnrollProducer(threading.Thread):
    """Reads raw frames from camera → frame_queue."""

    def __init__(self, frame_queue: multiprocessing.Queue) -> None:
        super().__init__(name="EnrollProducer", daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("EnrollProducer: cannot open camera.")
            try:
                self.frame_queue.put(None, timeout=1)
            except Exception:
                pass
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        time.sleep(config.CAMERA_WARMUP_SECS)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            # Keep queue fresh — drop oldest
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

        cap.release()
        try:
            self.frame_queue.put(None, timeout=1)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────
#  EnrollConsumer — detection + embedding process
# ─────────────────────────────────────────────────────────────────

class _EnrollConsumer(multiprocessing.Process):
    """
    Runs in a child process (real core, no GIL).
    Detects faces, extracts ArcFace embeddings, annotates frames.

    Queues
    ------
    frame_queue   ← raw frames from EnrollProducer
    display_queue → annotated frames for main thread (cv2.imshow)
    cmd_queue     ← commands from main thread ("START" / "STOP")
    result_queue  → final averaged embedding dict for main thread
    """

    def __init__(
        self,
        frame_queue:   multiprocessing.Queue,
        display_queue: multiprocessing.Queue,
        cmd_queue:     multiprocessing.Queue,
        result_queue:  multiprocessing.Queue,
        student_name:  str,
        student_id:    str,
        save_dir:      str,
    ) -> None:
        super().__init__(name="EnrollConsumer", daemon=True)
        self.frame_queue   = frame_queue
        self.display_queue = display_queue
        self.cmd_queue     = cmd_queue
        self.result_queue  = result_queue
        self.student_name  = student_name
        self.student_id    = student_id
        self.save_dir      = save_dir

    @staticmethod
    def _push_display(q: multiprocessing.Queue, item) -> None:
        try:
            if q.full():
                q.get_nowait()
            q.put_nowait(item)
        except Exception:
            pass

    def run(self) -> None:
        # Import + instantiate models inside process (not picklable)
        import embeddings as emb_module
        from detector import FaceDetector

        detector   = FaceDetector()
        fps_ctr    = utils.FPSCounter()

        embeddings_list: List[np.ndarray] = []
        captured   = 0
        started    = False
        last_cap_t = 0.0
        running    = True

        while running:
            # ── check for commands ────────────────────────────────
            try:
                cmd = self.cmd_queue.get_nowait()
                if cmd == "START":
                    started    = True
                    last_cap_t = 0.0
                elif cmd == "STOP":
                    running = False
                    break
            except Exception:
                pass

            # ── get frame ─────────────────────────────────────────
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Exception:
                continue

            if frame is None:
                break

            fps     = fps_ctr.tick()
            display = frame.copy()
            h, w    = frame.shape[:2]

            # ── detect ────────────────────────────────────────────
            faces      = detector.detect(frame)
            face_found = False

            for det in faces[:1]:
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

                # ── auto-capture ──────────────────────────────────
                if started and (time.monotonic() - last_cap_t) >= config.ENROLL_CAPTURE_DELAY:
                    emb = emb_module.extract_embedding_from_frame(frame, det["bbox"])
                    if emb is not None:
                        embeddings_list.append(emb)
                        captured   += 1
                        last_cap_t  = time.monotonic()
                        img_path    = os.path.join(
                            self.save_dir,
                            f"{utils.timestamp_filename()}.jpg",
                        )
                        cv2.imwrite(img_path, frame[y: y + bh, x: x + bw])
                        logger.debug("Captured %d/%d", captured, config.ENROLL_IMAGES_COUNT)
                    else:
                        utils.put_text_bg(display, "Embedding failed — hold still",
                                          (x, y + bh + 20), config.COLOR_ORANGE)

            # ── HUD ───────────────────────────────────────────────
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
                cv2.rectangle(display, (10, h - 18), (w - 10, h - 6),
                              config.COLOR_WHITE, 1)
                if bar > 0:
                    cv2.rectangle(display, (10, h - 18), (10 + bar, h - 6),
                                  config.COLOR_GREEN, cv2.FILLED)

            self._push_display(self.display_queue, display)

            # ── done? ─────────────────────────────────────────────
            if captured >= config.ENROLL_IMAGES_COUNT:
                break

        # Compute and return result
        if embeddings_list:
            stacked  = np.stack(embeddings_list, axis=0)
            mean_emb = np.mean(stacked, axis=0).astype(np.float32)
            norm     = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb /= norm
            try:
                self.result_queue.put({
                    "ok":        True,
                    "embedding": mean_emb,
                    "count":     len(embeddings_list),
                }, timeout=2)
            except Exception:
                pass
        else:
            try:
                self.result_queue.put({"ok": False, "count": 0}, timeout=2)
            except Exception:
                pass

        # Drain display_queue sentinel
        self._push_display(self.display_queue, None)
        logger.info("EnrollConsumer process stopped.")


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────

def enroll_student(student_id: str = "", student_name: str = "") -> bool:
    """
    Interactive camera-based enrollment (Pi 5 optimised).

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

    # OPTIMIZATION: register window with display server BEFORE heavy work
    cv2.startWindowThread()

    # ── prepare dataset folder ────────────────────────────────────
    save_dir = utils.student_image_dir(student_id, student_name)

    # ── build queues ──────────────────────────────────────────────
    frame_q   = multiprocessing.Queue(maxsize=2)
    display_q = multiprocessing.Queue(maxsize=3)
    cmd_q     = multiprocessing.Queue(maxsize=4)
    result_q  = multiprocessing.Queue(maxsize=1)

    producer = _EnrollProducer(frame_q)
    consumer = _EnrollConsumer(
        frame_queue   = frame_q,
        display_queue = display_q,
        cmd_queue     = cmd_q,
        result_queue  = result_q,
        student_name  = student_name,
        student_id    = student_id,
        save_dir      = save_dir,
    )

    producer.start()
    consumer.start()

    win_title = f"Enrolling — {student_name}"
    result    = None
    aborted   = False

    # ── main display loop (this thread ONLY does imshow) ─────────
    while True:
        try:
            frame = display_q.get(timeout=0.1)
        except Exception:
            # Timeout — consumer may have finished
            if not consumer.is_alive():
                break
            # Check for a result even without a new frame
            try:
                result = result_q.get_nowait()
                break
            except Exception:
                pass
            continue

        if frame is None:
            # Sentinel — consumer is done
            break

        cv2.imshow(win_title, frame)

        # OPTIMIZATION: waitKey(20) — same as Recognition.py
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            aborted = True
            try:
                cmd_q.put_nowait("STOP")
            except Exception:
                pass
            break
        elif key == ord(" "):
            try:
                cmd_q.put_nowait("START")
                print("  Capturing …")
            except Exception:
                pass

    # Collect result if not already grabbed
    if result is None and not aborted:
        try:
            result = result_q.get(timeout=3)
        except Exception:
            pass

    # ── cleanup ───────────────────────────────────────────────────
    producer.stop()
    producer.join(timeout=3)
    consumer.join(timeout=5)
    cv2.destroyAllWindows()

    if aborted:
        print("\n  Enrollment aborted.")
        return False

    if result is None or not result.get("ok"):
        count = result.get("count", 0) if result else 0
        min_ok = max(1, config.ENROLL_IMAGES_COUNT // 2)
        print(f"\n  [ERROR] Only {count} embeddings captured (need ≥ {min_ok}).")
        print("  Try again in better lighting.\n")
        return False

    mean_emb = result["embedding"]
    n_shots  = result["count"]

    # ── persist ───────────────────────────────────────────────────
    if database.student_exists(student_id):
        database.update_student_embedding(student_id, mean_emb, n_shots)
    else:
        database.add_student(student_id, student_name, n_shots, mean_emb)

    emb_module.get_embedding_db().reload()

    print(f"\n  ✓ Enrolled {student_name} ({student_id})")
    print(f"    Shots averaged : {n_shots}")
    print(f"    Images saved   : {save_dir}\n")
    logger.info("Enrolled %s (%s) — %d shots", student_id, student_name, n_shots)
    return True
