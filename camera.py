"""
camera.py — Producer–Consumer threaded camera pipeline (v4, all bugs fixed).

Bugs fixed from v3
------------------
BUG 1 — Frozen display (root cause):
    Single result_queue had two consumers (AttendanceWorker + display loop)
    racing. AttendanceWorker won every frame → display froze on frame 1.
    FIX: Two separate output queues: display_queue (main thread) and
         attend_queue (AttendanceWorker). Consumer pushes to both.

BUG 2 — All faces Unknown:
    get_face_embedding(crop) passed a bare tight crop to InsightFace which
    re-runs its own detector internally — tight crops cause 0 detections.
    FIX: embeddings.extract_embedding_from_frame() adds 30% padding + retry.

BUG 3 — AttributeError on stop:
    recognition.py checked `result == "STOP"` but sentinel is None.
    FIX: Consumer sends None; display loop checks pipeline.is_alive().

BUG 4 — Worker or display loop hangs:
    One None sentinel, two consumers — whoever got it stopped; other hung.
    FIX: Consumer sends None to display_queue AND attend_queue separately.

BUG 5 — Loop exits before all frames shown:
    pipeline.is_alive() checked producer.is_alive(); producer stops before
    consumer processes remaining queued frames.
    FIX: is_alive() checks consumer.is_alive().

Architecture
────────────
  Camera
    │
    ▼  frame_queue (maxsize=FRAME_QUEUE_SIZE)
  FrameProducer → drops oldest when full so consumer always gets fresh frame
    │
    ▼
  FrameConsumer
    │   YuNet detect → padded embedding → vote buffer → annotate
    ├── display_queue → main thread (cv2.imshow)
    └── attend_queue  → AttendanceWorker (mark_attendance)
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config
import database
import utils
import embeddings as emb_module
from detector import FaceDetector

logger = logging.getLogger("attendance_system.camera")

# ─────────────────────────────────────────────────────────────────
#  Global stop event — shared across all threads in one session
# ─────────────────────────────────────────────────────────────────

_stop_event = threading.Event()


def request_stop() -> None:
    _stop_event.set()


def reset_stop() -> None:
    """Must be called before starting a new pipeline session."""
    _stop_event.clear()


# ─────────────────────────────────────────────────────────────────
#  Thread 1 — Frame Producer
# ─────────────────────────────────────────────────────────────────

class FrameProducer(threading.Thread):
    """
    Reads frames from the camera and pushes (monotonic_ts, frame) tuples
    into frame_queue.  Drops the oldest queued frame when the queue is full
    so the consumer always receives the most recent capture.
    """

    def __init__(
        self,
        frame_queue:  queue.Queue,
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
            _stop_event.set()   # propagate failure
            return

        while not _stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("FrameProducer: read failed — retrying …")
                time.sleep(0.05)
                continue

            self.fps = self._fps.tick()

            # Drop oldest to make room (keeps queue fresh, not stale)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait((time.monotonic(), frame))
            except queue.Full:
                pass   # rare race — acceptable

        self._cap.release()
        logger.info("FrameProducer: camera released.")

        # BUG 4 FIX: blocking put so sentinel is never silently dropped
        try:
            self.frame_queue.put(None, timeout=2)
        except queue.Full:
            _stop_event.set()


# ─────────────────────────────────────────────────────────────────
#  Thread 2 — Frame Consumer
# ─────────────────────────────────────────────────────────────────

class FrameConsumer(threading.Thread):
    """
    Pulls frames from frame_queue.
    For each frame: YuNet detection → padded ArcFace embedding → vote buffer.
    Pushes annotated frame to display_queue and recognition data to attend_queue.

    BUG 1 FIX: two separate output queues — zero contention between display
    loop and AttendanceWorker.
    """

    def __init__(
        self,
        frame_queue:   queue.Queue,
        display_queue: queue.Queue,
        attend_queue:  queue.Queue,
        producer:      FrameProducer,
    ) -> None:
        super().__init__(name="FrameConsumer", daemon=True)
        self.frame_queue   = frame_queue
        self.display_queue = display_queue
        self.attend_queue  = attend_queue
        self.producer      = producer
        self._detector     = FaceDetector()
        self._emb_db       = emb_module.get_embedding_db()
        # Consecutive-frame vote buffer: grid_key → [student_id, …]
        self._votes: Dict[str, List[str]] = {}

    def _push(self, q: queue.Queue, item: Any) -> None:
        """Non-blocking push; drop oldest if full."""
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
        logger.info("FrameConsumer started.")

        while not _stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:   # BUG 3 FIX: sentinel from producer
                break

            _ts, frame = item
            display    = frame.copy()
            h, w       = frame.shape[:2]

            # ── detect ───────────────────────────────────────────
            raw_faces   = self._detector.detect(frame)
            attend_data: List[Dict] = []
            curr_keys   = set()

            for det in raw_faces:
                x, y, bw, bh = det["bbox"]

                # BUG 2 FIX: padded extraction avoids InsightFace re-detection fail
                embedding = emb_module.extract_embedding_from_frame(frame, det["bbox"])

                if embedding is None:
                    sid, sname, dist = "Unknown", "Unknown", 9999.0
                else:
                    sid, sname, dist = self._emb_db.recognize(embedding)

                # ── consecutive-frame vote ────────────────────────
                vkey = f"{x // 80}:{y // 80}"
                curr_keys.add(vkey)
                buf  = self._votes.setdefault(vkey, [])
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

            # Purge stale vote keys
            for k in list(self._votes.keys()):
                if k not in curr_keys:
                    del self._votes[k]

            # ── HUD ──────────────────────────────────────────────
            utils.draw_hud(display, self.producer.fps, len(raw_faces))

            # BUG 1 FIX: push to SEPARATE queues
            self._push(self.display_queue, display)
            if attend_data:
                self._push(self.attend_queue, attend_data)

        # BUG 4 FIX: sentinel to BOTH downstream queues
        for q in (self.display_queue, self.attend_queue):
            try:
                q.put(None, timeout=1)
            except queue.Full:
                _stop_event.set()

        logger.info("FrameConsumer stopped.")


# ─────────────────────────────────────────────────────────────────
#  Thread 3 — Attendance Worker
# ─────────────────────────────────────────────────────────────────

class AttendanceWorker(threading.Thread):
    """
    Reads from attend_queue only (never display_queue).
    Marks attendance with per-student cooldown.
    Notifies main thread via marked_queue.
    """

    def __init__(
        self,
        attend_queue: queue.Queue,
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
            except queue.Empty:
                continue

            if item is None:   # BUG 4 FIX
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
    Wires FrameProducer → FrameConsumer → {display_queue, attend_queue}.

    Queue layout
    ────────────────────────────────────────────────────────────────
    frame_queue   Producer → Consumer          maxsize=FRAME_QUEUE_SIZE
    display_queue Consumer → main thread       maxsize=DISPLAY_QUEUE_SIZE
    attend_queue  Consumer → AttendanceWorker  maxsize=ATTEND_QUEUE_SIZE
    marked_queue  Worker   → main thread       maxsize=20
    ────────────────────────────────────────────────────────────────
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX) -> None:
        reset_stop()
        self.frame_queue   = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.display_queue = queue.Queue(maxsize=config.DISPLAY_QUEUE_SIZE)
        self.attend_queue  = queue.Queue(maxsize=config.ATTEND_QUEUE_SIZE)
        self.marked_queue  = queue.Queue(maxsize=20)

        self.producer   = FrameProducer(self.frame_queue, camera_index)
        self.consumer   = FrameConsumer(
            self.frame_queue, self.display_queue, self.attend_queue, self.producer
        )
        self.att_worker = AttendanceWorker(self.attend_queue, self.marked_queue)

    def start(self) -> None:
        self.producer.start()
        self.consumer.start()
        self.att_worker.start()
        logger.info("AttendancePipeline started.")

    def stop(self) -> None:
        request_stop()
        self.producer.join(timeout=4)
        self.consumer.join(timeout=10)   # inference can be slow on Pi
        self.att_worker.join(timeout=3)
        cv2.destroyAllWindows()
        logger.info("AttendancePipeline stopped.")

    def is_alive(self) -> bool:
        # BUG 5 FIX: check consumer (feeds display), not producer
        return self.consumer.is_alive() and not _stop_event.is_set()

    def get_display_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Pop one annotated frame for display.
        Returns None on timeout (no new frame) or when consumer sends sentinel.
        Caller should check is_alive() to distinguish the two cases.
        """
        try:
            return self.display_queue.get(timeout=timeout)
        except queue.Empty:
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