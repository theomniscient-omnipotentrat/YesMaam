"""
detector.py — YuNet ONNX face detector (OpenCV FaceDetectorYN).

Replaces Haar cascade with a modern CNN detector that:
  • Handles face tilt ±30°
  • Returns per-detection confidence
  • Returns 5 facial landmark points
  • Runs real-time on CPU

The ONNX model is downloaded automatically on first use (~300 KB).

Pi 5 optimisations
------------------
• target_id = DNN_TARGET_CPU_FP16  — ARM NEON processes 16-bit floats at
  roughly 2× the throughput of FP32, cutting YuNet latency significantly.
• Internal 320 px downscale — YuNet runs on a fixed 320-wide thumbnail;
  bboxes and landmarks are scaled back up to the original resolution before
  returning, so callers see full-resolution coordinates.

Drawing helpers are in utils.py — this module only handles detection.
"""

import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("attendance_system.detector")

_N_LANDMARKS  = 5
_DETECT_WIDTH = 320   # internal detection width — keeps YuNet cheap on Pi 5


# ─────────────────────────────────────────────────────────────────
#  Model download
# ─────────────────────────────────────────────────────────────────

def _ensure_model() -> str:
    """Download YuNet ONNX model if not present. Return local path."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    if os.path.isfile(config.YUNET_MODEL_FILE):
        return config.YUNET_MODEL_FILE

    logger.info("Downloading YuNet model …")
    print("  Downloading YuNet model … ", end="", flush=True)
    try:
        urllib.request.urlretrieve(config.YUNET_MODEL_URL, config.YUNET_MODEL_FILE)
        print("done.")
        logger.info("YuNet model saved: %s", config.YUNET_MODEL_FILE)
    except Exception as exc:
        print(f"FAILED: {exc}")
        # Clean up partial file
        if os.path.isfile(config.YUNET_MODEL_FILE):
            os.remove(config.YUNET_MODEL_FILE)
        raise RuntimeError(
            f"YuNet download failed.\n"
            f"Manual download: {config.YUNET_MODEL_URL}\n"
            f"Save to:         {config.YUNET_MODEL_FILE}"
        ) from exc

    return config.YUNET_MODEL_FILE


# ─────────────────────────────────────────────────────────────────
#  FaceDetector
# ─────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Wraps cv2.FaceDetectorYN (YuNet ONNX) with Pi 5 hardware acceleration.

    Key differences from the base version
    --------------------------------------
    • Uses DNN_TARGET_CPU_FP16 — ARM NEON 16-bit path, ~2× faster than FP32.
    • Internally downscales every frame to _DETECT_WIDTH px before running
      YuNet, then scales bbox/landmarks back to original dimensions.  This
      reduces the pixels the CNN processes by ~4× for a 640×480 input while
      keeping face detection reliable down to MIN_FACE_SIZE_PX.

    Usage
    -----
        detector = FaceDetector()
        faces = detector.detect(frame)
        for f in faces:
            x, y, w, h = f["bbox"]
            conf        = f["confidence"]   # 0.0–1.0
            landmarks   = f["landmarks"]    # 5 × (x, y) tuples
            # All coordinates are in the ORIGINAL frame's coordinate space.
    """

    def __init__(
        self,
        score_threshold: float = config.YUNET_SCORE_THRESH,
        nms_threshold:   float = config.YUNET_NMS_THRESH,
        top_k:           int   = config.YUNET_TOP_K,
    ) -> None:
        model_path = _ensure_model()
        self._score_thresh = score_threshold

        # Detector always runs at _DETECT_WIDTH; height is recalculated
        # per-frame inside detect() when the aspect ratio is known.
        self._det_size: Tuple[int, int] = (_DETECT_WIDTH, _DETECT_WIDTH)

        self._det = cv2.FaceDetectorYN.create(
            model           = model_path,
            config          = "",
            input_size      = self._det_size,
            score_threshold = score_threshold,
            nms_threshold   = nms_threshold,
            top_k           = top_k,
            backend_id      = cv2.dnn.DNN_BACKEND_OPENCV,
            target_id       = cv2.dnn.DNN_TARGET_CPU_FP16,  # ARM NEON FP16 path
        )
        logger.info(
            "YuNet ready (FP16, detect_width=%d, score≥%.2f)",
            _DETECT_WIDTH, score_threshold,
        )

    def _set_det_size(self, det_w: int, det_h: int) -> None:
        """Update YuNet's internal input size only when it changes."""
        size = (det_w, det_h)
        if size != self._det_size:
            self._det_size = size
            self._det.setInputSize(size)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a BGR frame.

        Internally downscales to _DETECT_WIDTH for speed, then maps all
        output coordinates back to the original frame dimensions so callers
        do not need to know about the internal scaling.

        Returns
        -------
        List of dicts sorted by confidence (highest first):
          { "bbox": [x, y, w, h], "confidence": float, "landmarks": [(x,y)×5] }
          All coordinates are in the ORIGINAL frame's pixel space.
        """
        orig_h, orig_w = frame.shape[:2]

        # ── scale down ───────────────────────────────────────────
        scale = _DETECT_WIDTH / orig_w
        det_w = _DETECT_WIDTH
        det_h = max(1, int(orig_h * scale))
        small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)

        self._set_det_size(det_w, det_h)

        _, raw = self._det.detect(small)
        if raw is None or len(raw) == 0:
            return []

        # ── inverse scale factor ─────────────────────────────────
        inv = 1.0 / scale   # multiply small-space coords to get orig-space coords

        results: List[Dict[str, Any]] = []
        for det in raw:
            # YuNet row: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
            #             rm_x, rm_y, lm_x, lm_y, score]
            bx = int(det[0] * inv)
            by = int(det[1] * inv)
            bw = int(det[2] * inv)
            bh = int(det[3] * inv)
            score = float(det[14])

            # Clamp to original frame bounds
            bx = max(0, bx);  by = max(0, by)
            bw = min(bw, orig_w - bx);  bh = min(bh, orig_h - by)

            if bw < config.MIN_FACE_SIZE_PX or bh < config.MIN_FACE_SIZE_PX:
                continue

            landmarks = [
                (int(det[4 + i * 2] * inv), int(det[5 + i * 2] * inv))
                for i in range(_N_LANDMARKS)
            ]
            results.append({
                "bbox":       [bx, by, bw, bh],
                "confidence": round(score, 4),
                "landmarks":  landmarks,
            })

        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results

    def detect_largest(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Return only the largest face by area, or None."""
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
