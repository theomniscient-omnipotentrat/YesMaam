"""
gui.py — Graphical User Interface for Face Attendance System v4.

Architecture
------------
• Pure tkinter + ttk — no extra GUI dependencies (works on Pi out of the box).
• All camera/ML work runs in daemon threads so the UI never freezes.
• cv2.imshow() windows appear as separate OS windows alongside the GUI.
• The GUI polls a result_queue at 100 ms intervals to receive async updates.

Design: industrial dark theme — high contrast, amber accents, monospace data.
"""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import sys
import os
import time
from datetime import datetime, timedelta
from typing import Optional

# ── ensure local modules importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import database
import utils

# ─────────────────────────────────────────────────────────────────
#  Design tokens
# ─────────────────────────────────────────────────────────────────

BG_BASE   = "#0f1117"
BG_PANEL  = "#181c27"
BG_CARD   = "#1e2235"
BG_HOVER  = "#252a40"
BG_INPUT  = "#252a40"

ACCENT    = "#f5a623"      # amber
ACCENT2   = "#4fc3f7"      # sky blue
SUCCESS   = "#4caf7d"
DANGER    = "#e05c5c"
WARNING   = "#f5a623"

FG_PRIMARY   = "#e8eaf6"
FG_SECONDARY = "#8892b0"
FG_MUTED     = "#4a5270"

BORDER    = "#2a3050"
RADIUS    = 8

FONT_MONO  = ("Courier New", 11)
FONT_MONO_SM = ("Courier New", 9)
FONT_TITLE = ("Georgia", 22, "bold")
FONT_HEAD  = ("Georgia", 14, "bold")
FONT_LABEL = ("Helvetica", 10)
FONT_LABEL_BOLD = ("Helvetica", 10, "bold")
FONT_SMALL = ("Helvetica", 9)
FONT_NUM   = ("Courier New", 28, "bold")


# ─────────────────────────────────────────────────────────────────
#  Reusable widgets
# ─────────────────────────────────────────────────────────────────

class Card(tk.Frame):
    def __init__(self, parent, **kw):
        kw.setdefault("bg", BG_CARD)
        kw.setdefault("bd", 0)
        kw.setdefault("relief", "flat")
        kw.setdefault("padx", 16)
        kw.setdefault("pady", 14)
        super().__init__(parent, **kw)


class StatCard(Card):
    def __init__(self, parent, label: str, value: str, color: str = ACCENT, **kw):
        super().__init__(parent, **kw)
        tk.Label(self, text=label, font=FONT_SMALL, fg=FG_SECONDARY, bg=BG_CARD).pack(anchor="w")
        self.val_lbl = tk.Label(self, text=value, font=FONT_NUM, fg=color, bg=BG_CARD)
        self.val_lbl.pack(anchor="w")

    def set(self, value: str):
        self.val_lbl.config(text=value)


class SidebarBtn(tk.Frame):
    """Sidebar navigation button with hover + active states."""

    def __init__(self, parent, icon: str, text: str, command, **kw):
        super().__init__(parent, bg=BG_PANEL, cursor="hand2", **kw)
        self._cmd     = command
        self._active  = False
        self._icon    = icon
        self._text    = text

        self._inner = tk.Frame(self, bg=BG_PANEL, padx=14, pady=10)
        self._inner.pack(fill="x")
        self._icon_lbl = tk.Label(self._inner, text=icon, font=("Helvetica", 16),
                                   fg=FG_SECONDARY, bg=BG_PANEL, width=2)
        self._icon_lbl.pack(side="left")
        self._text_lbl = tk.Label(self._inner, text=text, font=FONT_LABEL_BOLD,
                                   fg=FG_SECONDARY, bg=BG_PANEL, anchor="w")
        self._text_lbl.pack(side="left", padx=8)

        for w in (self, self._inner, self._icon_lbl, self._text_lbl):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>",    self._hover_on)
            w.bind("<Leave>",    self._hover_off)

    def _set_colors(self, bg, fg, accent_bg=None):
        ab = accent_bg or bg
        self.config(bg=bg)
        self._inner.config(bg=ab)
        self._icon_lbl.config(bg=ab, fg=fg)
        self._text_lbl.config(bg=ab, fg=fg)

    def _click(self, _=None):
        self._cmd()

    def _hover_on(self, _=None):
        if not self._active:
            self._set_colors(BG_HOVER, FG_PRIMARY, BG_HOVER)

    def _hover_off(self, _=None):
        if not self._active:
            self._set_colors(BG_PANEL, FG_SECONDARY)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._set_colors(BG_CARD, ACCENT, BG_CARD)
        else:
            self._set_colors(BG_PANEL, FG_SECONDARY)


class ActionBtn(tk.Button):
    def __init__(self, parent, text: str, command, style="primary", **kw):
        colors = {
            "primary": (ACCENT,   "#1a1200", ACCENT),
            "success": (SUCCESS,  "#001a0d", SUCCESS),
            "danger":  (DANGER,   "#1a0000", DANGER),
            "ghost":   (BG_CARD,  FG_PRIMARY, BORDER),
        }
        bg, fg, ab = colors.get(style, colors["primary"])
        kw.setdefault("relief", "flat")
        kw.setdefault("bd", 0)
        kw.setdefault("padx", 18)
        kw.setdefault("pady", 8)
        kw.setdefault("cursor", "hand2")
        kw.setdefault("font", FONT_LABEL_BOLD)
        super().__init__(parent, text=text, command=command,
                         bg=bg, fg=fg, activebackground=ab, activeforeground=fg, **kw)

    def disable(self):
        self.config(state="disabled", bg=FG_MUTED, fg=BG_BASE)

    def enable(self):
        self.config(state="normal")


class Table(tk.Frame):
    """Scrollable treeview table with dark theme."""

    def __init__(self, parent, columns: list, **kw):
        super().__init__(parent, bg=BG_CARD, **kw)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=BG_CARD, foreground=FG_PRIMARY,
                        fieldbackground=BG_CARD, rowheight=28,
                        font=FONT_MONO_SM)
        style.configure("Dark.Treeview.Heading",
                        background=BG_BASE, foreground=ACCENT,
                        font=FONT_LABEL_BOLD, relief="flat")
        style.map("Dark.Treeview",
                  background=[("selected", BG_HOVER)],
                  foreground=[("selected", ACCENT)])

        self.tree = ttk.Treeview(self, columns=columns, show="headings",
                                  style="Dark.Treeview")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

    def clear(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

    def insert(self, values: tuple, tag: str = ""):
        self.tree.insert("", "end", values=values, tags=(tag,))
        if tag == "present":
            self.tree.tag_configure("present", foreground=SUCCESS)
        elif tag == "absent":
            self.tree.tag_configure("absent", foreground=DANGER)


# ─────────────────────────────────────────────────────────────────
#  Section pages
# ─────────────────────────────────────────────────────────────────

class DashboardPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG_BASE, pady=24, padx=32)
        hdr.pack(fill="x")
        tk.Label(hdr, text="FACE ATTENDANCE SYSTEM",
                 font=FONT_TITLE, fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")
        tk.Label(hdr, text="v4 · Pi 5 Optimised",
                 font=FONT_SMALL, fg=FG_MUTED, bg=BG_BASE).pack(side="left", padx=12, pady=6)

        self._time_lbl = tk.Label(hdr, text="", font=FONT_MONO, fg=ACCENT, bg=BG_BASE)
        self._time_lbl.pack(side="right")

        # Stat cards row
        row = tk.Frame(self, bg=BG_BASE, padx=32)
        row.pack(fill="x")

        self._stat_students  = StatCard(row, "ENROLLED STUDENTS", "—", ACCENT)
        self._stat_today     = StatCard(row, "PRESENT TODAY", "—", SUCCESS)
        self._stat_absent    = StatCard(row, "ABSENT TODAY", "—", DANGER)
        self._stat_rate      = StatCard(row, "ATTENDANCE RATE", "—", ACCENT2)

        for card in (self._stat_students, self._stat_today,
                     self._stat_absent,  self._stat_rate):
            card.pack(side="left", padx=(0, 12), pady=8, ipadx=8, ipady=4)

        # Divider
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=32, pady=8)

        # Today's attendance table
        mid = tk.Frame(self, bg=BG_BASE, padx=32, pady=8)
        mid.pack(fill="both", expand=True)

        lbl_row = tk.Frame(mid, bg=BG_BASE)
        lbl_row.pack(fill="x", pady=(0, 8))
        tk.Label(lbl_row, text="TODAY'S ATTENDANCE", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")
        ActionBtn(lbl_row, "↺  Refresh", self.refresh, style="ghost").pack(side="right")

        cols = ("#", "Student ID", "Name", "Time", "Status")
        self._table = Table(mid, cols)
        self._table.pack(fill="both", expand=True)
        self._table.tree.column("#",          width=40,  minwidth=40)
        self._table.tree.column("Student ID", width=120, minwidth=100)
        self._table.tree.column("Name",       width=200, minwidth=140)
        self._table.tree.column("Time",       width=100, minwidth=80)
        self._table.tree.column("Status",     width=90,  minwidth=70)

        self._tick_clock()
        self.refresh()

    def _tick_clock(self):
        self._time_lbl.config(text=datetime.now().strftime("%a %d %b %Y   %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def refresh(self):
        total    = database.get_student_count()
        records  = database.get_attendance_by_date()
        present  = len(records)
        absent   = max(0, total - present)
        rate_str = f"{present/total*100:.0f}%" if total else "—"

        self._stat_students.set(str(total))
        self._stat_today.set(str(present))
        self._stat_absent.set(str(absent))
        self._stat_rate.set(rate_str)

        self._table.clear()
        for i, r in enumerate(records, 1):
            self._table.insert(
                (i, r["student_id"], r["student_name"], r["time"], r["status"]),
                tag="present",
            )


class EnrollPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._running = False
        self._build()

    def _build(self):
        tk.Frame(self, bg=BG_BASE, height=24).pack()

        hdr = tk.Frame(self, bg=BG_BASE, padx=32)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Enroll Student", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")

        body = tk.Frame(self, bg=BG_BASE, padx=32, pady=24)
        body.pack(fill="both", expand=True)

        # Form card
        form = Card(body, padx=28, pady=24)
        form.pack(side="left", fill="y", ipadx=8)

        tk.Label(form, text="NEW ENROLLMENT", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD, pady=4).pack(anchor="w")
        tk.Frame(form, bg=BORDER, height=1).pack(fill="x", pady=(4, 16))

        def field(parent, label, var):
            tk.Label(parent, text=label, font=FONT_LABEL, fg=FG_SECONDARY,
                     bg=BG_CARD).pack(anchor="w")
            e = tk.Entry(parent, textvariable=var, font=FONT_MONO,
                         bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                         relief="flat", bd=6)
            e.pack(fill="x", pady=(2, 12))
            return e

        self._id_var   = tk.StringVar()
        self._name_var = tk.StringVar()
        field(form, "Student ID", self._id_var)
        field(form, "Full Name",  self._name_var)

        tk.Frame(form, bg=BG_CARD, height=8).pack()
        self._enroll_btn = ActionBtn(form, "▶  Start Enrollment Camera",
                                     self._start_enroll, style="primary")
        self._enroll_btn.pack(fill="x", pady=(4, 0))

        # Info card
        info = Card(body, padx=24, pady=24)
        info.pack(side="left", fill="both", expand=True, padx=(16, 0))

        tk.Label(info, text="HOW IT WORKS", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")
        tk.Frame(info, bg=BORDER, height=1).pack(fill="x", pady=(4, 16))

        steps = [
            ("1", "Enter the student ID and full name above."),
            ("2", "Click Start — an OpenCV camera window opens."),
            ("3", "Position the student's face in the frame."),
            ("4", "Press SPACE to begin capturing."),
            (f"5", f"System captures {config.ENROLL_IMAGES_COUNT} ArcFace shots and averages them."),
            ("6", "Embedding is stored in SQLite; student is immediately\n    available for recognition."),
        ]
        for num, txt in steps:
            row = tk.Frame(info, bg=BG_CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=num, font=FONT_MONO, fg=ACCENT, bg=BG_CARD,
                     width=2).pack(side="left")
            tk.Label(row, text=txt, font=FONT_LABEL, fg=FG_SECONDARY,
                     bg=BG_CARD, justify="left", anchor="w").pack(side="left", padx=8)

        tk.Frame(info, bg=BORDER, height=1).pack(fill="x", pady=12)

        # Log area
        tk.Label(info, text="ACTIVITY LOG", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")
        self._log = tk.Text(info, height=7, font=FONT_MONO_SM,
                             bg=BG_BASE, fg=SUCCESS, relief="flat",
                             state="disabled", bd=0)
        self._log.pack(fill="both", expand=True, pady=(4, 0))

    def _log_write(self, msg: str):
        self._log.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}]  {msg}\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _start_enroll(self):
        if self._running:
            return
        sid  = self._id_var.get().strip()
        name = self._name_var.get().strip()
        if not sid:
            messagebox.showerror("Missing Field", "Please enter a Student ID.", parent=self)
            return
        if not name:
            messagebox.showerror("Missing Field", "Please enter a Student Name.", parent=self)
            return

        self._running = True
        self._enroll_btn.disable()
        self._log_write(f"Starting enrollment for {name} ({sid}) …")

        def _run():
            try:
                import enroll as enroll_mod
                ok = enroll_mod.enroll_student(student_id=sid, student_name=name)
                if ok:
                    self.after(0, lambda: self._log_write(
                        f"✓  {name} enrolled successfully."))
                    self.after(0, lambda: self.app.dashboard.refresh())
                else:
                    self.after(0, lambda: self._log_write(
                        "✗  Enrollment aborted or failed."))
            except Exception as exc:
                self.after(0, lambda: self._log_write(f"ERROR: {exc}"))
            finally:
                self._running = False
                self.after(0, self._enroll_btn.enable)

        threading.Thread(target=_run, daemon=True).start()


class AttendancePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._running = False
        self._build()

    def _build(self):
        tk.Frame(self, bg=BG_BASE, height=24).pack()

        hdr = tk.Frame(self, bg=BG_BASE, padx=32)
        hdr.pack(fill="x", pady=(0, 8))
        tk.Label(hdr, text="Take Attendance", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")

        body = tk.Frame(self, bg=BG_BASE, padx=32)
        body.pack(fill="both", expand=True)

        # Control card
        ctrl = Card(body, padx=24, pady=24)
        ctrl.pack(side="left", fill="y", ipadx=8)

        tk.Label(ctrl, text="RECOGNITION SESSION", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=(4, 18))

        self._status_var = tk.StringVar(value="IDLE")
        tk.Label(ctrl, textvariable=self._status_var,
                 font=("Courier New", 18, "bold"), fg=FG_MUTED, bg=BG_CARD).pack(pady=8)

        self._start_btn = ActionBtn(ctrl, "▶  Launch Camera",
                                    self._start_recognition, style="success")
        self._start_btn.pack(fill="x", pady=(0, 8))

        tk.Label(ctrl, text="Press Q or ESC in the camera\nwindow to end the session.",
                 font=FONT_SMALL, fg=FG_MUTED, bg=BG_CARD, justify="left").pack(anchor="w")

        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=14)
        tk.Label(ctrl, text="HOTKEYS (in camera window)",
                 font=FONT_SMALL, fg=ACCENT, bg=BG_CARD).pack(anchor="w", pady=(0, 6))
        for key, desc in [("Q / ESC", "End session"), ("S", "Save snapshot")]:
            r = tk.Frame(ctrl, bg=BG_CARD)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=key, font=FONT_MONO_SM, fg=ACCENT2,
                     bg=BG_CARD, width=8, anchor="w").pack(side="left")
            tk.Label(r, text=desc, font=FONT_SMALL, fg=FG_SECONDARY,
                     bg=BG_CARD).pack(side="left")

        # Marked list card
        marked = Card(body, padx=20, pady=20)
        marked.pack(side="left", fill="both", expand=True, padx=(16, 0))

        r2 = tk.Frame(marked, bg=BG_CARD)
        r2.pack(fill="x", pady=(0, 8))
        tk.Label(r2, text="MARKED THIS SESSION", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(side="left")
        ActionBtn(r2, "Clear", self._clear_marked, style="ghost").pack(side="right")

        tk.Frame(marked, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))

        cols = ("Time", "Student ID", "Name")
        self._table = Table(marked, cols)
        self._table.pack(fill="both", expand=True)
        self._table.tree.column("Time",       width=90)
        self._table.tree.column("Student ID", width=110)
        self._table.tree.column("Name",       width=200)

    def _clear_marked(self):
        self._table.clear()

    def _start_recognition(self):
        if self._running:
            return
        if database.get_student_count() == 0:
            messagebox.showwarning("No Students",
                "No students are enrolled yet.\nPlease enroll students first.", parent=self)
            return

        self._running = True
        self._start_btn.disable()
        self._status_var.set("RUNNING")

        def _poll_marked(pipeline):
            """Poll the pipeline's marked_queue and insert rows into the table."""
            if not self._running:
                return
            for rec in pipeline.drain_marked():
                ts = datetime.now().strftime("%H:%M:%S")
                self._table.insert(
                    (ts, rec["student_id"], rec["student_name"]),
                    tag="present",
                )
                self.app.dashboard.refresh()
            self.after(400, lambda: _poll_marked(pipeline))

        def _run():
            try:
                import Recognition as rec_mod
                # Monkey-patch to get pipeline reference for polling
                import camera as cam_mod
                import embeddings as emb_module
                import cv2, time

                emb_db = emb_module.get_embedding_db()
                pipeline = cam_mod.AttendancePipeline(config.CAMERA_INDEX)
                pipeline.start()

                # Start polling from GUI thread
                self.after(0, lambda: _poll_marked(pipeline))

                last_frame = None
                while True:
                    frame = pipeline.get_display_frame(timeout=0.05)
                    if frame is None:
                        if not pipeline.is_alive():
                            break
                        if last_frame is not None:
                            cv2.imshow("Attendance System", last_frame)
                    else:
                        last_frame = frame
                        cv2.imshow("Attendance System", frame)

                    key = cv2.waitKey(20) & 0xFF
                    if key in (ord("q"), 27):
                        break

                pipeline.stop()
                import utils as utils_mod
                utils_mod.gpio_cleanup()

            except Exception as exc:
                self.after(0, lambda: messagebox.showerror(
                    "Recognition Error", str(exc), parent=self))
            finally:
                self._running = False
                self.after(0, self._start_btn.enable)
                self.after(0, lambda: self._status_var.set("IDLE"))
                self.after(0, self.app.dashboard.refresh)

        threading.Thread(target=_run, daemon=True).start()


class StudentsPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._build()

    def _build(self):
        tk.Frame(self, bg=BG_BASE, height=24).pack()

        hdr = tk.Frame(self, bg=BG_BASE, padx=32)
        hdr.pack(fill="x", pady=(0, 8))
        tk.Label(hdr, text="Student Management", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")

        btn_row = tk.Frame(hdr, bg=BG_BASE)
        btn_row.pack(side="right")
        ActionBtn(btn_row, "↺  Refresh", self.refresh, style="ghost").pack(side="left", padx=4)
        ActionBtn(btn_row, "✕  Remove Selected", self._remove, style="danger").pack(side="left")

        body = tk.Frame(self, bg=BG_BASE, padx=32)
        body.pack(fill="both", expand=True)

        cols = ("#", "Student ID", "Name", "Enrolled At", "Images", "Embedding")
        self._table = Table(body, cols)
        self._table.pack(fill="both", expand=True)
        self._table.tree.column("#",           width=40,  minwidth=40)
        self._table.tree.column("Student ID",  width=120, minwidth=100)
        self._table.tree.column("Name",        width=200, minwidth=140)
        self._table.tree.column("Enrolled At", width=160, minwidth=140)
        self._table.tree.column("Images",      width=70,  minwidth=60)
        self._table.tree.column("Embedding",   width=90,  minwidth=70)

        self.refresh()

    def refresh(self):
        self._table.clear()
        students = database.get_all_students()
        for i, s in enumerate(students, 1):
            has_emb = "✓" if s.get("embedding") is not None else "✗"
            self._table.insert(
                (i, s["id"], s["name"], s["enrolled_at"], s["image_count"], has_emb)
            )

    def _remove(self):
        sel = self._table.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a student row to remove.", parent=self)
            return
        vals = self._table.tree.item(sel[0])["values"]
        sid, name = str(vals[1]), str(vals[2])

        if not messagebox.askyesno(
            "Confirm Removal",
            f"Permanently remove '{name}' (ID: {sid})?\n\nThis cannot be undone.",
            parent=self,
        ):
            return

        # Delete dataset images
        import shutil
        folder  = f"{sid}_{name.replace(' ', '_')}"
        img_dir = os.path.join(config.DATASET_DIR, folder)
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)

        database.delete_student(sid)

        import embeddings as emb_module
        emb_module.get_embedding_db().reload()

        self.refresh()
        self.app.dashboard.refresh()
        messagebox.showinfo("Removed", f"'{name}' has been removed.", parent=self)


class ReportsPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._build()

    def _build(self):
        tk.Frame(self, bg=BG_BASE, height=24).pack()

        hdr = tk.Frame(self, bg=BG_BASE, padx=32)
        hdr.pack(fill="x", pady=(0, 16))
        tk.Label(hdr, text="Reports", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")

        body = tk.Frame(self, bg=BG_BASE, padx=32)
        body.pack(fill="both", expand=True)

        # Left: report controls
        ctrl = Card(body, padx=24, pady=24)
        ctrl.pack(side="left", fill="y", ipadx=8, pady=(0, 16))

        def section(title):
            tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=10)
            tk.Label(ctrl, text=title, font=FONT_SMALL, fg=ACCENT,
                     bg=BG_CARD).pack(anchor="w", pady=(0, 8))

        tk.Label(ctrl, text="GENERATE REPORTS", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")

        section("DATE RANGE ATTENDANCE")
        date_row1 = tk.Frame(ctrl, bg=BG_CARD)
        date_row1.pack(fill="x")
        tk.Label(date_row1, text="Start:", font=FONT_LABEL, fg=FG_SECONDARY,
                 bg=BG_CARD, width=5).pack(side="left")
        self._start_var = tk.StringVar(
            value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        tk.Entry(date_row1, textvariable=self._start_var, font=FONT_MONO_SM,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=4, width=12).pack(side="left", padx=4)

        date_row2 = tk.Frame(ctrl, bg=BG_CARD)
        date_row2.pack(fill="x", pady=4)
        tk.Label(date_row2, text="End:", font=FONT_LABEL, fg=FG_SECONDARY,
                 bg=BG_CARD, width=5).pack(side="left")
        self._end_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(date_row2, textvariable=self._end_var, font=FONT_MONO_SM,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=4, width=12).pack(side="left", padx=4)

        ActionBtn(ctrl, "Export Full Attendance CSV",
                  self._gen_full, style="primary").pack(fill="x", pady=(8, 0))

        section("ABSENT REPORT")
        date_row3 = tk.Frame(ctrl, bg=BG_CARD)
        date_row3.pack(fill="x")
        tk.Label(date_row3, text="Date:", font=FONT_LABEL, fg=FG_SECONDARY,
                 bg=BG_CARD, width=5).pack(side="left")
        self._absent_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(date_row3, textvariable=self._absent_date_var, font=FONT_MONO_SM,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=4, width=12).pack(side="left", padx=4)
        ActionBtn(ctrl, "Export Absent Report CSV",
                  self._gen_absent, style="danger").pack(fill="x", pady=(8, 0))

        section("DAILY SUMMARY")
        date_row4 = tk.Frame(ctrl, bg=BG_CARD)
        date_row4.pack(fill="x")
        tk.Label(date_row4, text="Date:", font=FONT_LABEL, fg=FG_SECONDARY,
                 bg=BG_CARD, width=5).pack(side="left")
        self._summary_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(date_row4, textvariable=self._summary_date_var, font=FONT_MONO_SM,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=4, width=12).pack(side="left", padx=4)
        ActionBtn(ctrl, "Export Daily Summary CSV",
                  self._gen_summary, style="ghost").pack(fill="x", pady=(8, 0))

        # Right: preview table
        preview = Card(body, padx=20, pady=20)
        preview.pack(side="left", fill="both", expand=True, padx=(16, 0))

        r = tk.Frame(preview, bg=BG_CARD)
        r.pack(fill="x", pady=(0, 8))
        tk.Label(r, text="ATTENDANCE PREVIEW", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(side="left")
        ActionBtn(r, "↺  Load Preview", self._load_preview, style="ghost").pack(side="right")

        tk.Frame(preview, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        cols = ("Date", "Student ID", "Name", "Time", "Status")
        self._table = Table(preview, cols)
        self._table.pack(fill="both", expand=True)

        self._status_lbl = tk.Label(preview, text="", font=FONT_SMALL,
                                     fg=SUCCESS, bg=BG_CARD)
        self._status_lbl.pack(anchor="w", pady=(6, 0))

        self._load_preview()

    def _load_preview(self):
        self._table.clear()
        try:
            start = self._start_var.get().strip()
            end   = self._end_var.get().strip()
            rows  = database.get_attendance_between(start, end)
        except Exception:
            rows = database.get_attendance_by_date()
        for r in rows:
            self._table.insert(
                (r["date"], r["student_id"], r["student_name"], r["time"], r["status"]),
                tag="present",
            )
        self._status_lbl.config(text=f"{len(rows)} record(s) loaded.")

    def _gen_full(self):
        try:
            import Reports as rep
            path = rep.generate_attendance_report(
                start_date=self._start_var.get().strip() or None,
                end_date  =self._end_var.get().strip()   or None,
            )
            messagebox.showinfo("Report Saved", f"Full attendance report saved to:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)

    def _gen_absent(self):
        try:
            import Reports as rep
            path = rep.generate_absent_report(
                target_date=self._absent_date_var.get().strip() or None,
                send_email =False,
            )
            messagebox.showinfo("Report Saved", f"Absent report saved to:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)

    def _gen_summary(self):
        try:
            import Reports as rep
            path = rep.generate_daily_summary(
                self._summary_date_var.get().strip() or None
            )
            messagebox.showinfo("Report Saved", f"Daily summary saved to:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self)


class AttendanceByDatePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._build()

    def _build(self):
        tk.Frame(self, bg=BG_BASE, height=24).pack()

        hdr = tk.Frame(self, bg=BG_BASE, padx=32)
        hdr.pack(fill="x", pady=(0, 8))
        tk.Label(hdr, text="View Attendance by Date", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")

        ctrl = tk.Frame(self, bg=BG_BASE, padx=32)
        ctrl.pack(fill="x", pady=(0, 8))

        tk.Label(ctrl, text="Date (YYYY-MM-DD):", font=FONT_LABEL,
                 fg=FG_SECONDARY, bg=BG_BASE).pack(side="left")
        self._date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(ctrl, textvariable=self._date_var, font=FONT_MONO,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=6, width=14).pack(side="left", padx=8)
        ActionBtn(ctrl, "Search", self.refresh, style="primary").pack(side="left")

        self._summary_lbl = tk.Label(ctrl, text="", font=FONT_LABEL,
                                      fg=FG_SECONDARY, bg=BG_BASE)
        self._summary_lbl.pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG_BASE, padx=32)
        body.pack(fill="both", expand=True)

        cols = ("#", "Student ID", "Name", "Time", "Status")
        self._table = Table(body, cols)
        self._table.pack(fill="both", expand=True)
        self._table.tree.column("#",          width=40)
        self._table.tree.column("Student ID", width=130)
        self._table.tree.column("Name",       width=220)
        self._table.tree.column("Time",       width=100)
        self._table.tree.column("Status",     width=90)

        self.refresh()

    def refresh(self):
        self._table.clear()
        date    = self._date_var.get().strip() or None
        records = database.get_attendance_by_date(date)
        for i, r in enumerate(records, 1):
            self._table.insert(
                (i, r["student_id"], r["student_name"], r["time"], r["status"]),
                tag="present",
            )
        total   = database.get_student_count()
        present = len(records)
        self._summary_lbl.config(
            text=f"Present: {present}  /  Total: {total}  |  Absent: {max(0, total-present)}"
        )



class CampusMonitoringPage(tk.Frame):
    """Small UI for the simulated campus-monitoring extension."""

    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_BASE)
        self.app = app
        self._build()

    def _build(self):
        tk.Frame(self, bg=BG_BASE, height=24).pack()

        hdr = tk.Frame(self, bg=BG_BASE, padx=32)
        hdr.pack(fill="x", pady=(0, 8))
        tk.Label(hdr, text="Campus Monitoring Simulation", font=FONT_HEAD,
                 fg=FG_PRIMARY, bg=BG_BASE).pack(side="left")

        body = tk.Frame(self, bg=BG_BASE, padx=32)
        body.pack(fill="both", expand=True)

        ctrl = Card(body, padx=24, pady=24)
        ctrl.pack(side="left", fill="y", ipadx=8)

        tk.Label(ctrl, text="ZONE-BASED SIMULATION", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=(4, 16))

        tk.Label(ctrl, text="Date (YYYY-MM-DD)", font=FONT_LABEL,
                 fg=FG_SECONDARY, bg=BG_CARD).pack(anchor="w")
        self._date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        tk.Entry(ctrl, textvariable=self._date_var, font=FONT_MONO_SM,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=5, width=14).pack(fill="x", pady=(2, 12))

        tk.Label(ctrl, text="Anomaly rate (0.0–1.0)", font=FONT_LABEL,
                 fg=FG_SECONDARY, bg=BG_CARD).pack(anchor="w")
        self._rate_var = tk.StringVar(value="0.30")
        tk.Entry(ctrl, textvariable=self._rate_var, font=FONT_MONO_SM,
                 bg=BG_INPUT, fg=FG_PRIMARY, insertbackground=ACCENT,
                 relief="flat", bd=5, width=14).pack(fill="x", pady=(2, 12))

        ActionBtn(ctrl, "Seed Demo Timetable", self._seed_timetable,
                  style="ghost").pack(fill="x", pady=(0, 8))
        ActionBtn(ctrl, "▶  Run Simulation", self._run_simulation,
                  style="primary").pack(fill="x", pady=(0, 8))
        ActionBtn(ctrl, "Export CSV Report", self._export_report,
                  style="success").pack(fill="x", pady=(0, 8))
        ActionBtn(ctrl, "↺  Refresh", self.refresh,
                  style="ghost").pack(fill="x", pady=(0, 8))

        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x", pady=14)
        tk.Label(ctrl,
                 text="This is a simulation layer. It compares\nexpected zones from a timetable with\nsimulated detections and raises alerts\nwhen they do not match.",
                 font=FONT_SMALL, fg=FG_MUTED, bg=BG_CARD,
                 justify="left").pack(anchor="w")

        preview = Card(body, padx=20, pady=20)
        preview.pack(side="left", fill="both", expand=True, padx=(16, 0))

        top = tk.Frame(preview, bg=BG_CARD)
        top.pack(fill="x", pady=(0, 8))
        tk.Label(top, text="SIMULATED ZONE EVENTS", font=FONT_SMALL,
                 fg=ACCENT, bg=BG_CARD).pack(side="left")
        self._summary_lbl = tk.Label(top, text="", font=FONT_SMALL,
                                     fg=FG_SECONDARY, bg=BG_CARD)
        self._summary_lbl.pack(side="right")

        tk.Frame(preview, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        cols = ("Time", "Student ID", "Name", "Detected", "Expected", "Alert")
        self._table = Table(preview, cols)
        self._table.pack(fill="both", expand=True)
        self._table.tree.column("Time", width=80)
        self._table.tree.column("Student ID", width=110)
        self._table.tree.column("Name", width=180)
        self._table.tree.column("Detected", width=130)
        self._table.tree.column("Expected", width=130)
        self._table.tree.column("Alert", width=70)

        self.refresh()

    def _date(self):
        return self._date_var.get().strip() or datetime.now().strftime("%Y-%m-%d")

    def _seed_timetable(self):
        try:
            import campus_monitoring as cm
            count = cm.seed_demo_timetable(overwrite=False)
            messagebox.showinfo("Timetable Seeded",
                                f"Inserted {count} timetable row(s). Existing rows were kept.",
                                parent=self)
        except Exception as exc:
            messagebox.showerror("Campus Simulation Error", str(exc), parent=self)

    def _run_simulation(self):
        try:
            import campus_monitoring as cm
            rate = float(self._rate_var.get().strip() or "0.30")
            result = cm.simulate_day(date=self._date(), anomaly_rate=rate, seed=42)
            self.refresh()
            messagebox.showinfo(
                "Simulation Complete",
                f"Events: {result['events']}\nAlerts: {result['alerts']}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Campus Simulation Error", str(exc), parent=self)

    def _export_report(self):
        try:
            import campus_monitoring as cm
            path = cm.generate_monitoring_report(self._date())
            messagebox.showinfo("Report Saved",
                                f"Campus monitoring report saved to:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Campus Simulation Error", str(exc), parent=self)

    def refresh(self):
        self._table.clear()
        try:
            import campus_monitoring as cm
            date = self._date()
            events = cm.get_zone_events(date)
            alerts = cm.get_alerts(date)
            alert_keys = {
                (a["student_id"], a["date"], a["time"], a["detected_zone"])
                for a in alerts
            }
            for e in events:
                expected = cm.get_expected_zone(e["student_id"], e["date"], e["time"])
                is_alert = (e["student_id"], e["date"], e["time"], e["detected_zone"]) in alert_keys
                self._table.insert(
                    (e["time"], e["student_id"], e["student_name"],
                     e["detected_zone"], expected or "No class", "YES" if is_alert else ""),
                    tag="absent" if is_alert else "present",
                )
            self._summary_lbl.config(text=f"Events: {len(events)}  |  Alerts: {len(alerts)}")
        except Exception as exc:
            self._summary_lbl.config(text=f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────
#  Main application window
# ─────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance System v4")
        self.configure(bg=BG_BASE)
        self.geometry("1200x720")
        self.minsize(960, 620)

        database.init_db()

        self._pages   = {}
        self._nav_btns = {}
        self._current  = None

        self._build_layout()
        self.navigate("dashboard")

    def _build_layout(self):
        # ── sidebar ───────────────────────────────────────────────
        self.sidebar = tk.Frame(self, bg=BG_PANEL, width=210)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo area
        logo = tk.Frame(self.sidebar, bg=BG_PANEL, pady=22, padx=16)
        logo.pack(fill="x")
        tk.Label(logo, text="◈", font=("Helvetica", 28), fg=ACCENT,
                 bg=BG_PANEL).pack(side="left")
        lbl_col = tk.Frame(logo, bg=BG_PANEL)
        lbl_col.pack(side="left", padx=8)
        tk.Label(lbl_col, text="ATTEND", font=("Georgia", 13, "bold"),
                 fg=FG_PRIMARY, bg=BG_PANEL).pack(anchor="w")
        tk.Label(lbl_col, text="Face System v4", font=FONT_SMALL,
                 fg=FG_MUTED, bg=BG_PANEL).pack(anchor="w")

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=12)

        nav_items = [
            ("dashboard",  "⊞", "Dashboard"),
            ("enroll",     "＋", "Enroll Student"),
            ("attendance", "◉", "Take Attendance"),
            ("bydate",     "▦", "View by Date"),
            ("students",   "◈", "Students"),
            ("campus",     "⌖", "Campus Sim"),
            ("reports",    "▤", "Reports"),
        ]
        for key, icon, label in nav_items:
            btn = SidebarBtn(
                self.sidebar, icon, label,
                command=lambda k=key: self.navigate(k),
            )
            btn.pack(fill="x")
            self._nav_btns[key] = btn

        tk.Frame(self.sidebar, bg=BG_PANEL).pack(fill="both", expand=True)

        # Pi info footer
        footer = tk.Frame(self.sidebar, bg=BG_PANEL, padx=14, pady=12)
        footer.pack(fill="x")
        tk.Frame(footer, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        tk.Label(footer, text="Raspberry Pi 5",
                 font=FONT_SMALL, fg=FG_MUTED, bg=BG_PANEL).pack(anchor="w")
        tk.Label(footer, text="YuNet · ArcFace · SQLite",
                 font=FONT_SMALL, fg=FG_MUTED, bg=BG_PANEL).pack(anchor="w")

        # ── content area ──────────────────────────────────────────
        self.content = tk.Frame(self, bg=BG_BASE)
        self.content.pack(side="left", fill="both", expand=True)

        # Build pages
        self.dashboard  = DashboardPage(self.content, self)
        self._pages["dashboard"]  = self.dashboard
        self._pages["enroll"]     = EnrollPage(self.content, self)
        self._pages["attendance"] = AttendancePage(self.content, self)
        self._pages["bydate"]     = AttendanceByDatePage(self.content, self)
        self._pages["students"]   = StudentsPage(self.content, self)
        self._pages["campus"]     = CampusMonitoringPage(self.content, self)
        self._pages["reports"]    = ReportsPage(self.content, self)

    def navigate(self, key: str):
        if self._current:
            self._pages[self._current].pack_forget()
            self._nav_btns[self._current].set_active(False)
        self._pages[key].pack(fill="both", expand=True)
        self._nav_btns[key].set_active(True)
        self._current = key
        # Refresh data-driven pages on visit
        if key == "students":
            self._pages["students"].refresh()
        elif key == "dashboard":
            self.dashboard.refresh()
        elif key == "bydate":
            self._pages["bydate"].refresh()
        elif key == "campus":
            self._pages["campus"].refresh()


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    import multiprocessing
    multiprocessing.freeze_support()   # needed for PyInstaller on Pi
    utils.setup_logging()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
