"""
camera.py — Producer–Consumer camera pipeline (v4, Pi 5 optimised).

Pi 5 Optimizations Applied
---------------------------
1. FrameConsumer → multiprocessing.Process  (bypasses the GIL)
   Python's GIL means threads share one core for CPU-bound work.
   On a Pi 5 (4× Cortex-A76 @ 2.4 GHz) converting FrameConsumer to a
   real OS process lets detection + embedding run on a dedicated core
   while the main thread renders the display on another — genuinely
   parallel, not time-sliced.

   Consequence: objects passed between processes must be picklable.
   FaceDetector and the embedding models are NOT picklable, so they are
   instantiated inside run() rather than __init__().

2. Recognition throttle — ArcFace runs at most once per 300 ms per face
   Humans don't change identity 30 times a second.  Running ArcFace on
   every frame at 30 fps wastes ~90 % of inference cycles.  The throttle:
   • detects every frame with YuNet (fast, ~12 ms FP16)
   • runs ArcFace only when the last result for that face position is
     older than RECOGNITION_THROTTLE_MS
   • reuses the cached (student_id, name, distance) for all other frames
   This keeps the UI fluid at ~30 fps while cutting CPU load dramatically.

IPC between processes
---------------------
multiprocessing.Queue is used instead of queue.Queue for all queues that
cross the process boundary (frame_queue, display_queue, attend_queue).
marked_queue stays as a threading.Queue because AttendanceWorker is still
a thread in the main process.

Stop signalling
---------------
A multiprocessing.Event (_mp_stop_event) is shared with the child process.
The threading.Event (_stop_event) is kept for AttendanceWorker / main thread.

Bug fixes from v3 are all preserved (see original camera.py docstring).
"""

import logging
import multiprocessing
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config
import database
import utils

logger = logging.getLogger("attendance_system.camera")

# ─────────────────────────────────────────────────────────────────
#  Stop signals — one per layer
# ─────────────────────────────────────────────────────────────────

# Used by FrameProducer (thread in main process) and AttendanceWorker
_stop_event = threading.Event()

# Shared with FrameConsumer (child process)
_mp_stop_event: Optional[multiprocessing.Event] = None   # set in AttendancePipeline.__init__


def request_stop() -> None:
    _stop_event.set()
    if _mp_stop_event is not None:
        _mp_stop_event.set()


def reset_stop() -> None:
    _stop_event.clear()
    # _mp_stop_event is re-created fresh in AttendancePipeline.__init__


# ─────────────────────────────────────────────────────────────────
#  Thread 1 — Frame Producer  (unchanged — runs in main process)
# ─────────────────────────────────────────────────────────────────

class FrameProducer(threading.Thread):
    """
    Reads frames from the camera and pushes (monotonic_ts, frame) into
    frame_queue.  Drops the oldest queued frame when full so the consumer
    always gets the most recent capture.
    """

    def __init__(
        self,
        frame_queue:  multiprocessing.Queue,
        camera_index: int = config.CAMERA_INDEX,
    ) -> None:
        super().__init__(name="FrameProducer", daemon=True)
        self.frame_queue  = frame_queue
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps  = 0.0
        self._fps = utils.FPSCounter()

    def _open(self) -> bool:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d", self.camera_index)
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
        time.sleep(config.CAMERA_WARMUP_SECS)
        self._cap = cap
        logger.info("Camera %d opened.", self.camera_index)
        return True

    def run(self) -> None:
        if not self._open():
            request_stop()
            return

        while not _stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("FrameProducer: read failed — retrying …")
                time.sleep(0.05)
                continue

            self.fps = self._fps.tick()

            # Keep queue fresh — drop oldest if full
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                self.frame_queue.put_nowait((time.monotonic(), frame))
            except Exception:
                pass

        self._cap.release()
        logger.info("FrameProducer: camera released.")
        try:
            self.frame_queue.put(None, timeout=2)
        except Exception:
            request_stop()


# ─────────────────────────────────────────────────────────────────
#  Process 2 — Frame Consumer  (OPTIMIZED: real OS process)
# ─────────────────────────────────────────────────────────────────

class FrameConsumer(multiprocessing.Process):
    """
    OPTIMIZATION 1: Inherits from multiprocessing.Process.

    Runs on a dedicated CPU core, bypassing the GIL.  The main process
    can render the UI at full speed while this process runs YuNet + ArcFace.

    IMPORTANT: FaceDetector and embedding models are created inside run()
    (not __init__) because ONNX Runtime handles / file descriptors are NOT
    picklable and would crash during process fork/spawn.

    OPTIMIZATION 2: Recognition throttle
    -------------------------------------
    _last_recog tracks the last time ArcFace was run for each face grid cell.
    If the result is fresher than RECOGNITION_THROTTLE_MS, the cached
    (student_id, name, distance) is reused and the expensive ArcFace call
    is skipped.  YuNet still runs every frame (~12 ms) so bbox tracking
    stays real-time.
    """

    def __init__(
        self,
        frame_queue:   multiprocessing.Queue,
        display_queue: multiprocessing.Queue,
        attend_queue:  multiprocessing.Queue,
        stop_event:    multiprocessing.Event,
        producer_fps_arr,  # multiprocessing.Array('d', [0.0])
    ) -> None:
        super().__init__(name="FrameConsumer", daemon=True)
        self.frame_queue      = frame_queue
        self.display_queue    = display_queue
        self.attend_queue     = attend_queue
        self.stop_event       = stop_event
        self.producer_fps_arr = producer_fps_arr

        # Models are NOT created here — see run()
        self._detector = None
        self._emb_db   = None

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _push(q: multiprocessing.Queue, item: Any, maxsize: int) -> None:
        """Non-blocking push; silently drop if full."""
        try:
            if q.qsize() >= maxsize:
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put_nowait(item)
        except Exception:
            pass

    # ── entry point ──────────────────────────────────────────────

    def run(self) -> None:
        """
        Entry point for the child process.

        Models are instantiated HERE so they are created inside the new
        process (after fork/spawn) and never need to be pickled.
        """
        # Import inside process to avoid any cross-process state issues
        import embeddings as emb_module
        from detector import FaceDetector

        self._detector = FaceDetector()
        self._emb_db   = emb_module.get_embedding_db()

        # OPTIMIZATION 2: throttle state
        # Maps grid_key → {"ts": float, "sid": str, "name": str, "dist": float}
        _last_recog: Dict[str, Dict] = {}
        # Consecutive-frame vote buffer: grid_key → [student_id, …]
        _votes: Dict[str, List[str]] = {}

        throttle_ms = config.RECOGNITION_THROTTLE_MS / 1000.0   # convert to seconds

        logger.info("FrameConsumer process started (PID=%d).", self.pid or -1)

        while not self.stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=0.2)
            except Exception:
                continue

            if item is None:   # sentinel from producer
                break

            _ts, frame = item
            display    = frame.copy()
            h, w       = frame.shape[:2]
            now        = time.monotonic()

            # ── detect ───────────────────────────────────────────
            raw_faces   = self._detector.detect(frame)
            attend_data: List[Dict] = []
            curr_keys   = set()

            for det in raw_faces:
                x, y, bw, bh = det["bbox"]

                # Grid key for throttle + vote buffer
                vkey = f"{x // 80}:{y // 80}"
                curr_keys.add(vkey)

                # ── OPTIMIZATION 2: throttled ArcFace ────────────
                cached = _last_recog.get(vkey)
                if cached is None or (now - cached["ts"]) >= throttle_ms:
                    # Time to run a fresh ArcFace extraction
                    import embeddings as emb_module
                    embedding = emb_module.extract_embedding_from_frame(frame, det["bbox"])
                    if embedding is None:
                        sid, sname, dist = "Unknown", "Unknown", 9999.0
                    else:
                        sid, sname, dist = self._emb_db.recognize(embedding)
                    _last_recog[vkey] = {
                        "ts": now, "sid": sid, "name": sname, "dist": dist,
                    }
                else:
                    # Reuse cached result — ArcFace skipped this frame
                    sid   = cached["sid"]
                    sname = cached["name"]
                    dist  = cached["dist"]

                # ── consecutive-frame vote ────────────────────────
                buf = _votes.setdefault(vkey, [])
                buf.append(sid)
                if len(buf) > config.CONSECUTIVE_FRAMES:
                    buf.pop(0)

                voted_id  = max(set(buf), key=buf.count) if buf else "Unknown"
                vote_conf = buf.count(voted_id) / max(len(buf), 1)
                is_known  = voted_id != "Unknown" and vote_conf >= 0.8

                # ── annotate ─────────────────────────────────────
                color = config.COLOR_GREEN if is_known else config.COLOR_RED
                label = (
                    f"{voted_id}: {sname}  d={dist:.2f}"
                    if is_known else f"Unknown  d={dist:.2f}"
                )
                utils.draw_bbox(display, x, y, bw, bh, label, color)
                for lx, ly in det["landmarks"]:
                    cv2.circle(display, (lx, ly), 2, config.COLOR_YELLOW, -1)

                attend_data.append({
                    "student_id":   voted_id if is_known else "Unknown",
                    "student_name": sname    if is_known else "Unknown",
                    "distance":     dist,
                    "is_known":     is_known,
                })

            # Purge stale vote / throttle entries
            for k in list(_votes.keys()):
                if k not in curr_keys:
                    del _votes[k]
            for k in list(_last_recog.keys()):
                if k not in curr_keys:
                    del _last_recog[k]

            # ── HUD ──────────────────────────────────────────────
            producer_fps = self.producer_fps_arr[0]
            utils.draw_hud(display, producer_fps, len(raw_faces))

            # Push to downstream queues
            self._push(self.display_queue, display, config.DISPLAY_QUEUE_SIZE)
            if attend_data:
                self._push(self.attend_queue, attend_data, config.ATTEND_QUEUE_SIZE)

        # Sentinels to unblock downstream consumers
        for q in (self.display_queue, self.attend_queue):
            try:
                q.put(None, timeout=1)
            except Exception:
                pass

        logger.info("FrameConsumer process stopped.")


# ─────────────────────────────────────────────────────────────────
#  Thread 3 — Attendance Worker  (unchanged — thread in main process)
# ─────────────────────────────────────────────────────────────────

class AttendanceWorker(threading.Thread):
    """
    Reads recognition results from attend_queue.
    Marks attendance with per-student cooldown.
    Notifies main thread via marked_queue.
    """

    def __init__(
        self,
        attend_queue: multiprocessing.Queue,
        marked_queue: queue.Queue,
    ) -> None:
        super().__init__(name="AttendanceWorker", daemon=True)
        self.attend_queue = attend_queue
        self.marked_queue = marked_queue
        self._lock        = threading.Lock()
        self._last: Dict[str, float] = {}

    def run(self) -> None:
        logger.info("AttendanceWorker started.")

        while not _stop_event.is_set():
            try:
                item = self.attend_queue.get(timeout=0.2)
            except Exception:
                continue

            if item is None:   # sentinel from consumer process
                break

            for face in item:
                if not face.get("is_known"):
                    continue
                sid, sname = face["student_id"], face["student_name"]
                now = time.time()

                with self._lock:
                    if now - self._last.get(sid, 0.0) < config.ATTENDANCE_COOLDOWN_SECS:
                        continue
                    self._last[sid] = now

                if database.mark_attendance(sid, sname):
                    logger.info("Marked: %s (%s)", sid, sname)
                    utils.gpio_pulse()
                    try:
                        self.marked_queue.put_nowait({"student_id": sid, "student_name": sname})
                    except queue.Full:
                        pass

        logger.info("AttendanceWorker stopped.")


# ─────────────────────────────────────────────────────────────────
#  Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────

class AttendancePipeline:
    """
    Wires FrameProducer (thread) → FrameConsumer (process)
         → {display_queue, attend_queue}
         → AttendanceWorker (thread)

    Queue layout
    ────────────────────────────────────────────────────────────────
    frame_queue   Producer → Consumer          mp.Queue (cross-process)
    display_queue Consumer → main thread       mp.Queue (cross-process)
    attend_queue  Consumer → AttendanceWorker  mp.Queue (cross-process)
    marked_queue  Worker   → main thread       queue.Queue (in-process)
    producer_fps  Producer → Consumer HUD      mp.Array (shared memory)
    ────────────────────────────────────────────────────────────────
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX) -> None:
        global _mp_stop_event
        reset_stop()

        # Shared stop event for the child process
        _mp_stop_event = multiprocessing.Event()

        # Cross-process queues
        self.frame_queue   = multiprocessing.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.display_queue = multiprocessing.Queue(maxsize=config.DISPLAY_QUEUE_SIZE)
        self.attend_queue  = multiprocessing.Queue(maxsize=config.ATTEND_QUEUE_SIZE)
        # In-process queue
        self.marked_queue  = queue.Queue(maxsize=20)

        # Shared memory for producer FPS → consumer HUD (avoids a separate queue)
        self._producer_fps = multiprocessing.Array("d", [0.0])

        self.producer = FrameProducer(self.frame_queue, camera_index)
        self.consumer = FrameConsumer(
            frame_queue      = self.frame_queue,
            display_queue    = self.display_queue,
            attend_queue     = self.attend_queue,
            stop_event       = _mp_stop_event,
            producer_fps_arr = self._producer_fps,
        )
        self.att_worker = AttendanceWorker(self.attend_queue, self.marked_queue)

        # Patch producer so it updates shared FPS for consumer HUD
        _orig_run = self.producer.run
        _arr      = self._producer_fps

        def _patched_run():
            _orig_run()

        # Override fps property setter to also write to shared array
        orig_tick = self.producer._fps.tick

        def _patched_tick():
            fps = orig_tick()
            _arr[0] = fps
            return fps

        self.producer._fps.tick = _patched_tick  # type: ignore

    def start(self) -> None:
        self.producer.start()
        self.consumer.start()
        self.att_worker.start()
        logger.info("AttendancePipeline started (consumer PID=%d).", self.consumer.pid or -1)

    def stop(self) -> None:
        request_stop()
        self.producer.join(timeout=4)
        self.consumer.join(timeout=10)
        self.att_worker.join(timeout=3)
        cv2.destroyAllWindows()
        logger.info("AttendancePipeline stopped.")

    def is_alive(self) -> bool:
        return self.consumer.is_alive() and not _stop_event.is_set()

    def get_display_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Pop one annotated frame for display.
        Returns None on timeout or when consumer sends sentinel.
        """
        try:
            return self.display_queue.get(timeout=timeout)
        except Exception:
            return None

    def drain_marked(self) -> List[Dict[str, str]]:
        """Return and clear all recently-marked student records."""
        items = []
        while True:
            try:
                items.append(self.marked_queue.get_nowait())
            except queue.Empty:
                break
        return items
