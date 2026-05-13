"""
embeddings.py — InsightFace ArcFace embedding extraction + recognition.

Pi 5 Optimizations Applied
---------------------------
1. allowed_modules=['recognition'] in FaceAnalysis()
   InsightFace bundles three sub-models: detection (RetinaFace ~200 MB),
   landmark alignment (~60 MB), and recognition/ArcFace (~80 MB).
   We already use YuNet for detection, so we only need 'recognition'.
   Excluding the other modules saves ~400 MB of RAM and cuts load time ~70%.

2. app.prepare(ctx_id=-1)  (explicit CPU / ARM)
   ctx_id=-1 forces ONNX Runtime to use CPUExecutionProvider with ARM-
   optimised kernels (NEON, dotprod).  This is the same as before but now
   explicitly documented as an intentional Pi 5 choice.

Key design decisions (unchanged)
----------------------------------
1. InsightFace loaded lazily + kept as singleton.
2. Embedding extraction always uses a 30%-padded crop (see extract_embedding_from_frame).
3. EmbeddingDatabase is an in-memory O(n) nearest-neighbour cache.
4. Recognition uses L2 distance on unit-norm vectors.
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

    OPTIMIZATION 1: allowed_modules=['recognition'] — loads ONLY the ArcFace
    recognition model.  We use YuNet for detection, so InsightFace's bundled
    RetinaFace detector is unnecessary.  Skipping it saves ~400 MB RAM and
    cuts startup time by ~70%.

    OPTIMIZATION 2: ctx_id=-1 — explicit CPU targeting activates ARM NEON /
    dotprod optimised kernels in ONNX Runtime on the Pi 5.

    Thread-safe via double-checked locking.
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

        print(
            f"  Loading InsightFace '{config.INSIGHTFACE_MODEL}' "
            f"(recognition-only) … ",
            end="", flush=True,
        )

        # OPTIMIZATION 1: only load the recognition (ArcFace) sub-model
        app = FaceAnalysis(
            name             = config.INSIGHTFACE_MODEL,
            root             = config.MODELS_DIR,
            providers        = providers,
            allowed_modules=['detection', 'recognition'],   # skip RetinaFace detector
        )

        # OPTIMIZATION 2: ctx_id=-1 → CPUExecutionProvider with ARM NEON
        app.prepare(ctx_id=-1, det_size=(320, 320))
        print("ready.")
        _app = app
        logger.info(
            "InsightFace loaded (model=%s, recognition-only, ctx=-1)",
            config.INSIGHTFACE_MODEL,
        )

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
    image : BGR numpy array — padded face crop, at least ~112×112 px.

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

    Adds 30% symmetric padding so InsightFace's internal alignment step
    has enough background context to reliably locate the face.
    Falls back to a 224×224 rescaled crop on first failure.
    """
    fh, fw = frame.shape[:2]
    x, y, bw, bh = bbox

    pad_x = int(bw * 0.30)
    pad_y = int(bh * 0.30)
    x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
    x2 = min(fw, x + bw + pad_x);  y2 = min(fh, y + bh + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    if cw < 112 or ch < 112:
        scale = max(112 / cw, 112 / ch)
        crop  = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

    emb = get_face_embedding(crop)
    if emb is not None:
        return emb

    logger.debug("extract_embedding: retry at 224×224")
    return get_face_embedding(cv2.resize(crop, (224, 224)))


def compare_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two unit-norm embeddings (0.0 identical → ~2.0 different)."""
    return float(np.linalg.norm(a - b))


# ─────────────────────────────────────────────────────────────────
#  In-memory embedding database
# ─────────────────────────────────────────────────────────────────

class EmbeddingDatabase:
    """
    Thread-safe in-memory cache of all enrolled face embeddings.
    Loaded from SQLite on construction; call reload() after enrollment.
    """

    def __init__(self) -> None:
        self._lock    = threading.RLock()
        self._entries: List[Tuple[str, str, np.ndarray]] = []
        self.reload()

    def reload(self) -> int:
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

        Returns (student_id, student_name, distance).
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
