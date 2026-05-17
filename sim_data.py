"""
sim_data.py — Synthetic data layer for the campus bunking simulator.

Everything here is FAKE. No real camera, no real face, no real student.

What it provides
----------------
• A roster of 20 simulated students with realistic names and IDs.
• A weekly timetable: every student has 3–5 classes per weekday.
• Fake ArcFace-style 512-d embeddings (unit-norm random vectors, seeded
  so the same student always gets the same vector).
• A helper that returns which students are "supposed to be in class"
  at any given simulated time.
• A helper that returns which students the simulator "sees" in the
  simulated hallway feed at any given simulated time.

Design
------
Embeddings use a per-student random seed derived from the student ID so
they are stable across restarts. L2 distance between two different
students' embeddings will naturally be ~1.0–1.4 (well above the
EMBEDDING_THRESHOLD of 0.50), mimicking real ArcFace behaviour.
"""

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────
#  Student roster
# ─────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Aisha", "Bilal", "Chloe", "Daniyar", "Elena",
    "Farhan", "Grace", "Hassan", "Ingrid", "Javier",
    "Kira",  "Liam",  "Mia",   "Nadia",  "Omar",
    "Priya", "Quinn", "Rania", "Samuel", "Tariq",
]

LAST_NAMES = [
    "Al-Rashid", "Bennett",  "Chen",    "Demir",   "Evans",
    "Farouk",    "Garcia",   "Hassan",  "Ivanova",  "Jimenez",
    "Khan",      "Laurent",  "Mendez",  "Nakamura", "Osei",
    "Patel",     "Qureshi",  "Romero",  "Santos",   "Thompson",
]

COURSES = [
    "CS101 - Intro to Programming",
    "CS201 - Data Structures",
    "CS301 - Algorithms",
    "CS401 - Operating Systems",
    "MA101 - Calculus I",
    "MA201 - Linear Algebra",
    "PH101 - Physics I",
    "EN101 - Technical Writing",
    "DS301 - Machine Learning",
    "NW201 - Computer Networks",
]

ROOMS = ["Room A1", "Room A2", "Room B1", "Lab 1", "Lab 2", "Lecture Hall"]
DAYS  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class ClassSlot:
    course:     str
    room:       str
    day:        str          # e.g. "Monday"
    start_time: time         # e.g. time(9, 0)
    end_time:   time         # e.g. time(10, 30)

    @property
    def label(self) -> str:
        return (f"{self.day}  {self.start_time.strftime('%H:%M')}–"
                f"{self.end_time.strftime('%H:%M')}  {self.course}  [{self.room}]")


@dataclass
class SimStudent:
    student_id:  str
    name:        str
    schedule:    List[ClassSlot] = field(default_factory=list)
    embedding:   Optional[np.ndarray] = field(default=None, repr=False)

    # Simulation personality
    bunk_prob:   float = 0.0   # probability of bunking any given class

    def classes_on(self, day: str) -> List[ClassSlot]:
        return [c for c in self.schedule if c.day == day]

    def in_class_at(self, day: str, t: time) -> Optional[ClassSlot]:
        """Return the ClassSlot the student should be attending at time t, or None."""
        for slot in self.schedule:
            if slot.day == day and slot.start_time <= t < slot.end_time:
                return slot
        return None


# ─────────────────────────────────────────────────────────────────
#  Embedding factory
# ─────────────────────────────────────────────────────────────────

def _make_embedding(student_id: str) -> np.ndarray:
    """
    Deterministic unit-norm 512-d vector derived from the student ID.
    Same ID → same vector every time (seeded RNG).
    """
    seed = int(hashlib.md5(student_id.encode()).hexdigest(), 16) % (2**32)
    rng  = np.random.default_rng(seed)
    v    = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────────────
#  Schedule generator
# ─────────────────────────────────────────────────────────────────

# Fixed 90-minute slots starting on the hour: 08:00, 09:30, 11:00, 13:00, 14:30
SLOT_TIMES: List[Tuple[time, time]] = [
    (time(8,  0), time(9,  30)),
    (time(9, 30), time(11,  0)),
    (time(11, 0), time(12, 30)),
    (time(13, 0), time(14, 30)),
    (time(14,30), time(16,  0)),
]


def _make_schedule(student_id: str, n_classes_per_day: int = 3) -> List[ClassSlot]:
    seed = int(hashlib.md5((student_id + "sched").encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    slots: List[ClassSlot] = []
    for day in DAYS:
        chosen_slots   = rng.sample(SLOT_TIMES, k=min(n_classes_per_day, len(SLOT_TIMES)))
        chosen_courses = rng.sample(COURSES, k=len(chosen_slots))
        chosen_rooms   = [rng.choice(ROOMS) for _ in chosen_slots]
        for (st, et), course, room in zip(chosen_slots, chosen_courses, chosen_rooms):
            slots.append(ClassSlot(course=course, room=room, day=day,
                                   start_time=st, end_time=et))
    slots.sort(key=lambda s: (DAYS.index(s.day), s.start_time))
    return slots


# ─────────────────────────────────────────────────────────────────
#  Roster factory — called once, cached as module-level singleton
# ─────────────────────────────────────────────────────────────────

_roster: Optional[List[SimStudent]] = None


def get_roster() -> List[SimStudent]:
    """Return the module-level cached roster of 20 simulated students."""
    global _roster
    if _roster is not None:
        return _roster

    rng = random.Random(42)
    students: List[SimStudent] = []

    first = FIRST_NAMES[:]
    last  = LAST_NAMES[:]
    rng.shuffle(first)
    rng.shuffle(last)

    for i in range(20):
        sid  = f"S{2024000 + (i + 1) * 7:07d}"
        name = f"{first[i]} {last[i]}"
        s    = SimStudent(
            student_id = sid,
            name       = name,
            schedule   = _make_schedule(sid, n_classes_per_day=rng.randint(2, 4)),
            embedding  = _make_embedding(sid),
            bunk_prob  = rng.choice([0.0, 0.0, 0.0, 0.15, 0.35, 0.55]),
        )
        students.append(s)

    _roster = students
    return _roster


def get_student_by_id(sid: str) -> Optional[SimStudent]:
    return next((s for s in get_roster() if s.student_id == sid), None)


# ─────────────────────────────────────────────────────────────────
#  Simulated clock helpers
# ─────────────────────────────────────────────────────────────────

def sim_day_and_time(sim_dt: datetime) -> Tuple[str, time]:
    """Return (weekday_name, time_of_day) for a simulated datetime."""
    day = sim_dt.strftime("%A")        # "Monday" … "Sunday"
    tod = sim_dt.time()
    return day, tod


# ─────────────────────────────────────────────────────────────────
#  Who should be in class right now?
# ─────────────────────────────────────────────────────────────────

def students_in_class_at(sim_dt: datetime) -> List[Tuple[SimStudent, ClassSlot]]:
    """Return [(student, slot), …] for all students with a class at sim_dt."""
    day, tod = sim_day_and_time(sim_dt)
    out = []
    for s in get_roster():
        slot = s.in_class_at(day, tod)
        if slot:
            out.append((s, slot))
    return out


# ─────────────────────────────────────────────────────────────────
#  Who does the hallway camera "see" right now?
# ─────────────────────────────────────────────────────────────────

_bunk_state: Dict[str, bool] = {}   # per student, per sim session


def reset_bunk_state() -> None:
    _bunk_state.clear()


def students_visible_in_hallway(sim_dt: datetime) -> List[SimStudent]:
    """
    Simulate which students the hallway camera detects at sim_dt.

    Logic
    -----
    • Students with no class → 30 % chance of wandering past the camera.
    • Students with a class  → they *bunk* with probability bunk_prob.
      Once they decide to bunk a slot they stay visible for that whole slot.
    • A small number of non-bunking students also walk past (between classes).
    """
    day, tod = sim_day_and_time(sim_dt)

    # Seed per-minute so the result is stable within a minute
    minute_seed = int(sim_dt.strftime("%Y%m%d%H%M"))
    rng = random.Random(minute_seed)

    visible: List[SimStudent] = []
    for s in get_roster():
        slot = s.in_class_at(day, tod)

        if slot is None:
            # Free period — 30 % chance they walk past
            if rng.random() < 0.30:
                visible.append(s)
        else:
            # Class time — do they bunk?
            key = f"{s.student_id}_{slot.day}_{slot.start_time}"
            if key not in _bunk_state:
                _bunk_state[key] = rng.random() < s.bunk_prob
            if _bunk_state[key]:
                visible.append(s)

    return visible
