"""
embeddings.py — InsightFace ArcFace embedding extraction + recognition.

Key design decisions
--------------------
1.  InsightFace is loaded lazily (first call) and kept as a singleton.
    Download is ~150 MB on first run; subsequent runs are instant.

2.  allowed_modules=['recognition'] prevents InsightFace from loading its
    own RetinaFace detector.  We already use YuNet, so the extra ~400 MB
    and 70 % of startup time are completely avoided.

3.  Embedding extraction always passes a PADDED crop (not raw YuNet bbox).
    InsightFace's recognition model still needs background margin on the
    crop to align the face correctly.  Without padding quality drops.

4.  EmbeddingDatabase keeps an in-memory cache of all (id, name, embedding)
    triples and provides fast nearest-neighbour search (O(n) L2 scan).
    n ≤ ~1000 students is fast; for larger deployments consider faiss.

5.  Recognition uses L2 distance on unit-norm vectors
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
    Lazy-load InsightFace FaceAnalysis — recognition module only.

    allowed_modules=['recognition']
        Skips loading RetinaFace (InsightFace's built-in detector).
        We supply our own YuNet crops, so the detector is dead weight:
        removing it saves ~400 MB RAM and ~70 % of startup time.

    ctx_id=-1
        Explicitly targets CPU, which on the Pi 5 ARM cores activates
        the optimised NEON code paths inside ONNX Runtime.

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
        print(f"  Loading InsightFace '{config.INSIGHTFACE_MODEL}' … ",
              end="", flush=True)

        # Try recognition-only first (saves ~400 MB download + load time).
        # Some InsightFace versions assert that a detection model is present
        # at __init__ time — if that assert fires we fall back to loading all
        # modules so the app works correctly on every platform.
        app = None
        for modules in (["recognition"], None):
            kwargs = dict(
                name      = config.INSIGHTFACE_MODEL,
                root      = config.MODELS_DIR,
                providers = providers,
            )
            if modules is not None:
                kwargs["allowed_modules"] = modules
            try:
                candidate = FaceAnalysis(**kwargs)
                # ctx_id=-1 → CPU; activates optimised ARM NEON paths on Pi 5
                candidate.prepare(ctx_id=-1, det_size=(320, 320))
                app = candidate
                tag = "recognition-only" if modules else "all-modules"
                break
            except (AssertionError, KeyError, Exception) as exc:
                if modules is None:
                    # Both attempts failed — re-raise so the caller gets a
                    # meaningful error rather than a silent None.
                    raise RuntimeError(
                        f"InsightFace failed to load model '{config.INSIGHTFACE_MODEL}'.\n"
                        f"Check that the model name is correct and the network is available.\n"
                        f"Original error: {exc}"
                    ) from exc
                logger.debug(
                    "recognition-only load failed (%s) — retrying with all modules", exc
                )

        print("ready.")
        _app = app
        logger.info(
            "InsightFace loaded (model=%s, mode=%s, ctx_id=-1)",
            config.INSIGHTFACE_MODEL, tag,
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

    Because we loaded InsightFace with allowed_modules=['recognition'], the
    app no longer runs an internal detector.  Instead it processes the entire
    image as a single aligned face crop.  The caller is responsible for
    supplying a properly padded, roughly face-centred crop.

    Parameters
    ----------
    image : BGR numpy array — a padded face crop from extract_embedding_from_frame.
            Must be at least ~112×112 px.

    Returns
    -------
    Unit-norm float32 ndarray of shape (512,), or None if extraction fails.
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

    Why padded crop?
    ----------------
    Even with allowed_modules=['recognition'], InsightFace still aligns the
    face internally using the bounding box it finds in the crop.  Without
    background margin the alignment step fails and the returned embedding is
    degraded or None.  30 % symmetric padding gives the aligner enough
    context to work reliably.

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

    # Attempt 2 — larger rescale gives the aligner more context
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