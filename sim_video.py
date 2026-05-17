"""
sim_video.py — Synthetic "live hallway feed" frame generator.

No real camera required. Every frame is drawn procedurally in OpenCV:

• A tiled corridor background with perspective lines.
• Walking "face blobs" — coloured ellipses representing detected persons.
  Each blob has a name tag, confidence score, and colour-coded status.
• A simulated YuNet detection box drawn around each visible person.
• HUD overlay: sim clock, zone label, FPS, detection count.

The simulator intentionally adds noise:
• Occasional false positives (unknown faces) pass through.
• Detection confidence jitters ±5 % per frame.
• Faces enter/exit from the sides of the frame.

All coordinates are deterministic per (sim_time, student_id) so the
animation is smooth and reproducible.
"""

import math
import random
import time as _time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sim_data

# ── palette ──────────────────────────────────────────────────────
_BG_DARK   = (20,  20,  30)
_BG_TILE   = (28,  28,  42)
_BUNKER    = (50,  80, 220)   # blue-ish BGR → red on screen (BGR: 0,0,220 = red)
_BUNKER    = (0,   0,  220)   # pure red
_FREE      = (40, 180,  40)   # green (in free period, ok to be here)
_UNKNOWN   = (180, 180, 50)   # cyan-ish for unknown
_TEXT_BG   = (10,  10,  20)
_WHITE     = (240, 240, 240)
_CYAN      = (220, 200,  50)
_YELLOW    = (0,  210, 255)

W, H = 960, 540


# ─────────────────────────────────────────────────────────────────
#  Background — tiled corridor
# ─────────────────────────────────────────────────────────────────

def _draw_background(frame: np.ndarray) -> None:
    frame[:] = _BG_DARK

    # Ceiling / floor gradient bands
    cv2.rectangle(frame, (0, 0),    (W, 60),       (30, 30, 48), -1)
    cv2.rectangle(frame, (0, H-70), (W, H),        (30, 30, 48), -1)

    # Perspective floor tiles
    vp = (W // 2, H // 2 - 20)   # vanishing point
    n_cols = 10
    for i in range(n_cols + 1):
        x_bot = int(i * W / n_cols)
        cv2.line(frame, vp, (x_bot, H), _BG_TILE, 1)

    n_rows = 8
    for j in range(1, n_rows + 1):
        t   = j / n_rows
        y   = int(vp[1] + (H - vp[1]) * (t ** 0.6))
        cv2.line(frame, (0, y), (W, y), _BG_TILE, 1)

    # Left wall
    cv2.rectangle(frame, (0, 0), (60, H), (24, 24, 38), -1)
    cv2.line(frame, (60, 0), (60, H), _BG_TILE, 1)
    # Right wall
    cv2.rectangle(frame, (W-60, 0), (W, H), (24, 24, 38), -1)
    cv2.line(frame, (W-60, 0), (W-60, H), _BG_TILE, 1)

    # Ceiling strip lights
    for lx in range(120, W - 120, 160):
        cv2.ellipse(frame, (lx, 30), (30, 8), 0, 0, 360, (80, 80, 100), -1)
        cv2.ellipse(frame, (lx, 30), (30, 8), 0, 0, 360, (200, 200, 220),  1)


# ─────────────────────────────────────────────────────────────────
#  Per-person position — smooth sinusoidal walk
# ─────────────────────────────────────────────────────────────────

def _person_pos(student_id: str, t: float, idx: int) -> Tuple[int, int, float]:
    """
    Return (cx, cy, scale) for a person at simulation time t.
    Each student walks at a different speed and vertical offset.
    scale ∈ [0.5, 1.1] simulates depth (far → near).
    """
    seed  = int(student_id[-4:]) if student_id[-4:].isdigit() else hash(student_id) & 0xFFFF
    rng   = random.Random(seed)
    speed = rng.uniform(0.08, 0.18)
    phase = rng.uniform(0, 2 * math.pi)
    lane  = rng.uniform(0.35, 0.75)   # vertical lane fraction

    x_frac = ((t * speed + phase / (2 * math.pi)) % 1.4) - 0.2
    cx = int(x_frac * W)

    # Depth: persons near the centre-top are smaller (far), near bottom are larger
    depth = 0.3 + abs(x_frac - 0.5) * 0.1 + lane * 0.55
    depth = min(max(depth, 0.35), 1.05)
    cy    = int(H * (0.38 + lane * 0.28))
    return cx, cy, depth


# ─────────────────────────────────────────────────────────────────
#  Draw one person blob
# ─────────────────────────────────────────────────────────────────

def _draw_person(
    frame:       np.ndarray,
    cx: int, cy: int, scale: float,
    name:        str,
    student_id:  str,
    status:      str,          # "BUNKING" | "FREE" | "UNKNOWN"
    confidence:  float,
    is_bunking:  bool,
) -> Tuple[int, int, int, int]:
    """
    Draw a coloured ellipse (face blob) + detection box + label.
    Returns (bx, by, bw, bh) of the detection box.
    """
    rw = int(38 * scale)   # face ellipse half-width
    rh = int(48 * scale)   # face ellipse half-height

    # Body rectangle (rough silhouette)
    bw_body = int(50 * scale)
    bh_body = int(110 * scale)
    cv2.rectangle(
        frame,
        (cx - bw_body // 2, cy - rh),
        (cx + bw_body // 2, cy - rh + bh_body),
        (35, 35, 55), -1,
    )

    # Face ellipse
    color = _BUNKER if is_bunking else (_UNKNOWN if status == "UNKNOWN" else _FREE)
    cv2.ellipse(frame, (cx, cy - rh + 18), (rw, rh - 10), 0, 0, 360, color, -1)

    # Eye dots
    eye_y = cy - rh + 12
    cv2.circle(frame, (cx - rw // 3, eye_y), max(2, int(3 * scale)), _TEXT_BG, -1)
    cv2.circle(frame, (cx + rw // 3, eye_y), max(2, int(3 * scale)), _TEXT_BG, -1)

    # Detection bounding box
    box_x = cx - bw_body // 2 - 4
    box_y = cy - rh - 4
    box_w = bw_body + 8
    box_h = bh_body + 8
    cv2.rectangle(frame, (box_x, box_y),
                  (box_x + box_w, box_y + box_h), color, 2)

    # Corner decorations
    cl = 10
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        x0 = box_x if dx < 0 else box_x + box_w
        y0 = box_y if dy < 0 else box_y + box_h
        cv2.line(frame, (x0, y0), (x0 + dx * cl, y0),          color, 2)
        cv2.line(frame, (x0, y0), (x0, y0 + dy * cl),          color, 2)

    # Label pill background
    font       = cv2.FONT_HERSHEY_SIMPLEX
    short_name = name.split()[0] if name != "UNKNOWN" else "UNKNOWN"
    label      = f"{short_name}  {confidence:.0%}"
    status_lbl = f"⚠ {status}" if is_bunking else status

    (tw, th), _ = cv2.getTextSize(label, font, 0.42 * scale, 1)
    lx = cx - tw // 2
    ly = box_y - 6
    cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4),
                  color, -1)
    cv2.putText(frame, label, (lx, ly), font, 0.42 * scale, _TEXT_BG, 1, cv2.LINE_AA)

    # Status tag below box
    (sw, sh), _ = cv2.getTextSize(status_lbl, font, 0.38 * scale, 1)
    sx = cx - sw // 2
    sy = box_y + box_h + sh + 6
    cv2.rectangle(frame, (sx - 4, sy - sh - 2), (sx + sw + 4, sy + 4),
                  _TEXT_BG, -1)
    cv2.putText(frame, status_lbl, (sx, sy), font, 0.38 * scale,
                color, 1, cv2.LINE_AA)

    return box_x, box_y, box_w, box_h


# ─────────────────────────────────────────────────────────────────
#  HUD overlay
# ─────────────────────────────────────────────────────────────────

def _draw_hud(
    frame:      np.ndarray,
    sim_dt:     datetime,
    fps:        float,
    n_detected: int,
    n_bunking:  int,
    zone:       str = "MAIN CORRIDOR — BLOCK A",
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top bar
    cv2.rectangle(frame, (0, 0), (W, 42), (10, 10, 22), -1)
    cv2.line(frame, (0, 42), (W, 42), (50, 50, 80), 1)

    # Zone name
    cv2.putText(frame, zone, (12, 28), font, 0.60, _CYAN, 1, cv2.LINE_AA)

    # Simulated clock
    clock = f"SIM  {sim_dt.strftime('%A  %H:%M:%S')}"
    (cw, _), _ = cv2.getTextSize(clock, font, 0.55, 1)
    cv2.putText(frame, clock, (W // 2 - cw // 2, 28), font, 0.55, _WHITE, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS {fps:.1f}", (W - 90, 28), font, 0.50, _CYAN, 1, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, H - 36), (W, H), (10, 10, 22), -1)
    cv2.line(frame, (0, H - 36), (W, H - 36), (50, 50, 80), 1)

    detected_txt = f"DETECTED: {n_detected}"
    cv2.putText(frame, detected_txt, (12, H - 12), font, 0.52, _WHITE, 1, cv2.LINE_AA)

    bunk_color = _BUNKER if n_bunking > 0 else _FREE
    bunk_txt   = f"BUNKING: {n_bunking}"
    cv2.putText(frame, bunk_txt, (180, H - 12), font, 0.52, bunk_color, 1, cv2.LINE_AA)

    rec_txt = "● REC"
    (rw, _), _ = cv2.getTextSize(rec_txt, font, 0.50, 1)
    cv2.putText(frame, rec_txt, (W - rw - 12, H - 12), font, 0.50,
                _BUNKER, 1, cv2.LINE_AA)

    # SIMULATED watermark
    cv2.putText(frame, "[ SIMULATED FEED ]",
                (W // 2 - 82, H - 12), font, 0.45, (60, 60, 90), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  False-positive generator
# ─────────────────────────────────────────────────────────────────

_fp_pool = ["UNK-001", "UNK-002", "UNK-003"]


def _maybe_false_positive(t: float) -> Optional[str]:
    """Return a fake unknown ID occasionally (roughly every 12 s)."""
    slot = int(t / 12)
    rng  = random.Random(slot * 999)
    if rng.random() < 0.25:
        return rng.choice(_fp_pool)
    return None


# ─────────────────────────────────────────────────────────────────
#  Main frame builder
# ─────────────────────────────────────────────────────────────────

class SimVideoFeed:
    """
    Generates synthetic hallway frames for a given simulated datetime.

    Usage
    -----
        feed = SimVideoFeed()
        while True:
            sim_dt = ...   # advance your simulated clock
            frame, detections = feed.next_frame(sim_dt)
            cv2.imshow("Feed", frame)
    """

    def __init__(self, speed: float = 1.0) -> None:
        """
        speed : how many sim-seconds pass per real second.
                e.g. speed=60 → 1 sim-minute per real second.
        """
        self.speed     = speed
        self._t0_real  = _time.monotonic()
        self._fps_buf: List[float] = []
        self._last_t   = _time.monotonic()

    def _fps(self) -> float:
        now = _time.monotonic()
        dt  = now - self._last_t
        self._last_t = now
        self._fps_buf.append(dt)
        if len(self._fps_buf) > 20:
            self._fps_buf.pop(0)
        avg = sum(self._fps_buf) / len(self._fps_buf)
        return 1.0 / avg if avg > 0 else 0.0

    def next_frame(
        self,
        sim_dt:  datetime,
        visible: List[sim_data.SimStudent],
        bunking: List[str],   # set of student IDs currently bunking
    ):
        """
        Build and return one annotated BGR frame plus a list of detection dicts.

        Parameters
        ----------
        sim_dt  : current simulated datetime
        visible : students the hallway camera currently sees
        bunking : student IDs that are classified as bunking right now

        Returns
        -------
        (frame: np.ndarray, detections: list[dict])
        """
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_background(frame)

        t   = _time.monotonic() - self._t0_real
        fps = self._fps()

        detections = []
        n_bunking  = 0

        # Possibly add a false-positive unknown
        fp_id = _maybe_false_positive(t)
        fp_visible = []
        if fp_id:
            fp_visible = [fp_id]

        all_ids = [(s.student_id, s.name, fp_id is None) for s in visible]
        for fid in fp_visible:
            all_ids.append((fid, "UNKNOWN", False))

        for idx, (sid, name, is_real) in enumerate(all_ids):
            cx, cy, scale = _person_pos(sid, t, idx)
            is_bunking    = sid in bunking

            # Jitter confidence ±5 %
            rng_conf = random.Random(int(t * 10) + hash(sid) & 0xFFFF)
            conf     = rng_conf.uniform(0.87, 0.97)

            if not is_real:
                status = "UNKNOWN"
            elif is_bunking:
                status = "BUNKING"
                n_bunking += 1
            else:
                status = "FREE"

            bx, by, bw, bh = _draw_person(
                frame, cx, cy, scale, name, sid,
                status, conf, is_bunking,
            )
            detections.append({
                "student_id":  sid,
                "name":        name,
                "status":      status,
                "confidence":  round(conf, 3),
                "bbox":        [bx, by, bw, bh],
                "is_bunking":  is_bunking,
            })

        _draw_hud(frame, sim_dt, fps, len(detections), n_bunking)
        return frame, detections
