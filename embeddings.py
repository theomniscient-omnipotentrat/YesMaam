"""
embeddings.py — InsightFace ArcFace embedding extraction + recognition.

Key design decisions
--------------------
1.  InsightFace is loaded lazily (first call) and kept as a singleton.
    Download is ~150 MB on first run; subsequent runs are instant.

2.  Embedding extraction always passes a PADDED crop (not raw YuNet bbox).
    InsightFace's internal detector (RetinaFace) needs background margin
    to locate the face inside the crop.  Without padding it returns None
    on almost every frame.

3.  EmbeddingDatabase keeps an in-memory cache of all (id, name, embedding)
    triples and provides fast nearest-neighbour search (O(n) L2 scan).
    n ≤ ~1000 students is fast; for larger deployments consider faiss.

4.  Recognition uses L2 distance on unit-norm vectors
    (equivalent to angular / cosine distance but faster to compute).
"""

import logging
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
import database

logger = logging.getLogger("attendance_system.embeddings")

# ─────────────────────────────────────────────────────────────────
#  InsightFace singleton
# ─────────────────────────────────────────────────────────────────

_app      = None
_app_lock = threading.Lock()


def _load_app():
    """
    Lazy-load InsightFace FaceAnalysis app.
    Downloads model weights on first call.
    Thread-safe via double-checked locking.

    Raises ImportError with an actionable message if insightface is missing.
    """
    global _app
    if _app is not None:
        return _app

    with _app_lock:
        if _app is not None:
            return _app

        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed.\n"
                "Run:  pip install insightface onnxruntime\n"
                "For GPU: pip install insightface onnxruntime-gpu"
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if config.INSIGHTFACE_CTX_ID >= 0
            else ["CPUExecutionProvider"]
        )
        print(f"  Loading InsightFace '{config.INSIGHTFACE_MODEL}' … ", end="", flush=True)
        app = FaceAnalysis(
            name      = config.INSIGHTFACE_MODEL,
            root      = config.MODELS_DIR,
            providers = providers,
        )
        # det_size controls the internal RetinaFace detection resolution
        app.prepare(ctx_id=config.INSIGHTFACE_CTX_ID, det_size=(320, 320))
        print("ready.")
        _app = app
        logger.info("InsightFace loaded (model=%s)", config.INSIGHTFACE_MODEL)

    return _app


def warmup() -> None:
    """Pre-load the InsightFace model (call at startup to avoid first-frame lag)."""
    _load_app()


# ─────────────────────────────────────────────────────────────────
#  Embedding extraction
# ─────────────────────────────────────────────────────────────────

def _normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector to unit length."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def get_face_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract a 512-d ArcFace embedding from the largest face in `image`.

    Parameters
    ----------
    image : BGR numpy array — can be a full frame or a padded face crop.
            Must be at least ~112×112 px.

    Returns
    -------
    Unit-norm float32 ndarray of shape (512,), or None if no face found.
    """
    app = _load_app()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        faces = app.get(rgb)
    except Exception as exc:
        logger.debug("InsightFace app.get() error: %s", exc)
        return None

    if not faces:
        return None

    # Pick the largest bounding box
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return _normalise(np.array(largest.embedding, dtype=np.float32))


def extract_embedding_from_frame(
    frame: np.ndarray,
    bbox: List[int],
) -> Optional[np.ndarray]:
    """
    Extract embedding for a specific face in `frame` identified by `bbox`.

    Why not just crop and call get_face_embedding?
    ----------------------------------------------
    InsightFace runs its own internal detector (RetinaFace) on whatever
    image you pass.  If you pass a tight crop (0% padding), RetinaFace
    sees a face that fills 100% of the image — it expects background
    margin and usually returns zero detections → None embedding.

    This function adds 30% symmetric padding before cropping so
    InsightFace's internal detector reliably finds the face.

    Falls back to a 224×224 rescaled crop on first failure.
    """
    fh, fw = frame.shape[:2]
    x, y, bw, bh = bbox

    pad_x = int(bw * 0.30)
    pad_y = int(bh * 0.30)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + bw + pad_x)
    y2 = min(fh, y + bh + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Ensure minimum spatial size
    ch, cw = crop.shape[:2]
    if cw < 112 or ch < 112:
        scale = max(112 / cw, 112 / ch)
        crop  = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

    # Attempt 1 — padded crop
    emb = get_face_embedding(crop)
    if emb is not None:
        return emb

    # Attempt 2 — larger rescale gives InsightFace more context
    logger.debug("extract_embedding: retry at 224×224")
    return get_face_embedding(cv2.resize(crop, (224, 224)))


def compare_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    """
    L2 distance between two unit-norm embeddings.
    Range: 0.0 (identical) → ~2.0 (completely different).
    """
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────
#  In-memory embedding database
# ─────────────────────────────────────────────────────────────────

class EmbeddingDatabase:
    """
    Thread-safe in-memory cache of all enrolled face embeddings.

    Loaded from SQLite on construction; call reload() after enrollment
    to pick up new students without restarting.
    """

    def __init__(self) -> None:
        self._lock    = threading.RLock()
        self._entries: List[Tuple[str, str, np.ndarray]] = []
        self.reload()

    def reload(self) -> int:
        """Refresh cache from SQLite. Returns number of entries loaded."""
        entries = database.get_all_embeddings()
        with self._lock:
            self._entries = entries
        logger.info("EmbeddingDatabase: %d embedding(s) loaded.", len(entries))
        return len(entries)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def recognize(
        self,
        query: np.ndarray,
        threshold: float = config.EMBEDDING_THRESHOLD,
    ) -> Tuple[str, str, float]:
        """
        Nearest-neighbour search against all stored embeddings.

        Returns
        -------
        (student_id, student_name, distance)
        Returns ("Unknown", "Unknown", 9999.0) when distance > threshold.
        """
        with self._lock:
            if not self._entries:
                return "Unknown", "Unknown", 9999.0

            best_id   = "Unknown"
            best_name = "Unknown"
            best_dist = float("inf")

            for sid, sname, stored in self._entries:
                d = compare_embeddings(query, stored)
                if d < best_dist:
                    best_dist = d
                    best_id   = sid
                    best_name = sname

        if best_dist > threshold:
            return "Unknown", "Unknown", best_dist
        return best_id, best_name, best_dist


# ─────────────────────────────────────────────────────────────────
#  Module-level singleton
# ─────────────────────────────────────────────────────────────────

_emb_db: Optional[EmbeddingDatabase] = None
_emb_db_lock = threading.Lock()


def get_embedding_db() -> EmbeddingDatabase:
    """Return the shared EmbeddingDatabase, creating it if needed."""
    global _emb_db
    if _emb_db is not None:
        return _emb_db
    with _emb_db_lock:
        if _emb_db is None:
            _emb_db = EmbeddingDatabase()
    return _emb_db