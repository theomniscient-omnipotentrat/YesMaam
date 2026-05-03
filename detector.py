"""
detector.py — YuNet ONNX face detector (OpenCV FaceDetectorYN).

Replaces Haar cascade with a modern CNN detector that:
  • Handles face tilt ±30°
  • Returns per-detection confidence
  • Returns 5 facial landmark points
  • Runs real-time on CPU

The ONNX model is downloaded automatically on first use (~300 KB).

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

_N_LANDMARKS = 5


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
    Wraps cv2.FaceDetectorYN (YuNet ONNX).

    Usage
    -----
        detector = FaceDetector()
        faces = detector.detect(frame)
        for f in faces:
            x, y, w, h = f["bbox"]
            conf        = f["confidence"]   # 0.0–1.0
            landmarks   = f["landmarks"]    # 5 × (x, y) tuples
    """

    def __init__(
        self,
        input_size: Optional[Tuple[int, int]] = None,
        score_threshold: float = config.YUNET_SCORE_THRESH,
        nms_threshold:   float = config.YUNET_NMS_THRESH,
        top_k:           int   = config.YUNET_TOP_K,
    ) -> None:
        model_path = _ensure_model()
        self._input_size   = input_size or (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        self._score_thresh = score_threshold

        self._det = cv2.FaceDetectorYN.create(
            model          = model_path,
            config         = "",
            input_size     = self._input_size,
            score_threshold= score_threshold,
            nms_threshold  = nms_threshold,
            top_k          = top_k,
            backend_id     = cv2.dnn.DNN_BACKEND_OPENCV,
            target_id      = cv2.dnn.DNN_TARGET_CPU,
        )
        logger.info("YuNet ready (input=%s, score≥%.2f)", self._input_size, score_threshold)

    def _update_size(self, w: int, h: int) -> None:
        if (w, h) != self._input_size:
            self._input_size = (w, h)
            self._det.setInputSize(self._input_size)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a BGR frame.

        Returns
        -------
        List of dicts sorted by confidence (highest first):
          { "bbox": [x, y, w, h], "confidence": float, "landmarks": [(x,y)×5] }
        """
        h, w = frame.shape[:2]
        self._update_size(w, h)

        _, raw = self._det.detect(frame)
        if raw is None or len(raw) == 0:
            return []

        results: List[Dict[str, Any]] = []
        for det in raw:
            # YuNet row: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
            #             rm_x, rm_y, lm_x, lm_y, score]
            bx, by, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            score = float(det[14])

            # Clamp to frame
            bx = max(0, bx);  by = max(0, by)
            bw = min(bw, w - bx);  bh = min(bh, h - by)

            if bw < config.MIN_FACE_SIZE_PX or bh < config.MIN_FACE_SIZE_PX:
                continue

            landmarks = [
                (int(det[4 + i * 2]), int(det[5 + i * 2]))
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