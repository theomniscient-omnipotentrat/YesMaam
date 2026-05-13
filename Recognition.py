"""
Recognition.py — Real-time attendance recognition session (v4, Pi 5 optimised).

Uses AttendancePipeline from camera.py (FrameConsumer is now a real process).
The main thread only: displays frames, handles keyboard, shows banners.
All detection / embedding / DB work runs in background process + threads.

Pi 5 Optimization Applied
--------------------------
cv2.waitKey(1)  →  cv2.waitKey(20)

On Wayland/X11 with a heavily loaded CPU, waitKey(1) essentially
busy-polls the event loop, stealing ~5 % of a core for GUI events.
waitKey(20) yields the display server ~20 ms per iteration — enough for
the compositor to commit the new frame without tearing — while capping
the UI event rate at 50 Hz, which is imperceptible to users.

Keyboard
--------
Q / ESC  — quit
S        — save snapshot
"""

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

import config
import database
import utils
import embeddings as emb_module
from camera import AttendancePipeline

logger = logging.getLogger("attendance_system.recognition")

_BANNER_SECS = 2.5   # seconds to show "✓ Marked" overlay


def run_recognition(camera_index: int = config.CAMERA_INDEX) -> None:
    """
    Start the threaded/multiprocessing pipeline and run the display loop
    until the user presses Q/ESC or the pipeline stops.
    """
    if database.get_student_count() == 0:
        print("\n  [ERROR] No students enrolled — use Enroll first.\n")
        return

    emb_db = emb_module.get_embedding_db()
    if emb_db.count == 0:
        print("\n  [ERROR] Embedding database is empty — re-enroll students.\n")
        return

    print(f"\n  Recognition session starting …")
    print(f"  Students: {database.get_student_count()}  |  Embeddings: {emb_db.count}")
    print("  [Q] quit   [S] snapshot\n")

    utils.gpio_setup()

    pipeline = AttendancePipeline(camera_index)
    pipeline.start()

    last_frame: Optional[np.ndarray] = None
    banner_text = ""
    banner_ts   = 0.0

    while True:
        frame = pipeline.get_display_frame(timeout=0.05)

        if frame is None:
            if not pipeline.is_alive():
                break
            if last_frame is not None:
                cv2.imshow("Attendance System", last_frame)
        else:
            last_frame = frame

            for rec in pipeline.drain_marked():
                banner_text = f"✓  Marked: {rec['student_name']}"
                banner_ts   = time.time()
                logger.info("Banner: %s", banner_text)

            if banner_text and (time.time() - banner_ts) < _BANNER_SECS:
                h = frame.shape[0]
                utils.put_text_bg(
                    frame, banner_text,
                    (10, h - 40),
                    config.COLOR_GREEN,
                    config.FONT_SCALE_LARGE,
                )

            cv2.imshow("Attendance System", frame)

        # OPTIMIZATION: waitKey(20) instead of waitKey(1)
        # Gives Wayland/X11 compositor 20 ms to render each frame,
        # preventing GUI freezes when all 4 cores are under load.
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s") and last_frame is not None:
            snap = os.path.join(
                config.BASE_DIR,
                f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
            )
            cv2.imwrite(snap, last_frame)
            print(f"  Snapshot → {snap}")

    pipeline.stop()
    utils.gpio_cleanup()
    print("\n  Recognition session ended.\n")
