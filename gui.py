"""
gui.py — Tkinter desktop GUI for Face Attendance System v5 (Pi 5 optimised).

Layout
------
┌──────────────┬──────────────────────────────────────────────────┐
│   Sidebar    │  Swappable content frames                        │
│  Navigation  │  Dashboard / Students / Attendance / Reports     │
│  Quick Acts  │                                                  │
│              ├──────────────────────────────────────────────────│
│              │  Status bar                                       │
└──────────────┴──────────────────────────────────────────────────┘

Design decisions
----------------
• All OpenCV windows (enrollment, recognition) are launched in daemon
  threads so the Tk main-loop is never blocked.
• Dashboard auto-refreshes every 30 s via Tk's .after() scheduler.
• The Treeview helper returns (container_frame, tree) so scrollbars are
  always included without duplicating boilerplate.
• Reports run in threads and post results back via .after(0, callback).
"""

import logging
import os
import sys
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import config
import database
import Reports as reports
import embeddings as emb_module

logger = logging.getLogger("attendance_system.gui")

# ─────────────────────────────────────────────────────────────────
#  Colour palette  (Catppuccin Mocha)
# ─────────────────────────────────────────────────────────────────
BG       = "#1e1e2e"
BG_PANEL = "#2a2a3e"
BG_CARD  = "#313150"
FG       = "#cdd6f4"
FG_DIM   = "#6c7086"
ACCENT   = "#89b4fa"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"
BORDER   = "#45475a"
CYAN     = "#89dceb"


# ─────────────────────────────────────────────────────────────────
#  Helper: scrollable Treeview
# ─────────────────────────────────────────────────────────────────

def _make_tree(
    parent,
    cols:     Tuple[str,  ...],
    headings: Tuple[str,  ...],
    widths:   Tuple[int,  ...],
    height:   int = 14,
) -> Tuple[tk.Frame, ttk.Treeview]:
    """
    Return (container_frame, treeview) with vertical + horizontal scrollbars.
    Pack or grid the container_frame; interact with the treeview directly.
    """
    frame = tk.Frame(parent, bg=BG)
    tree  = ttk.Treeview(frame, columns=cols, show="headings",
                         selectmode="browse", height=height)

    for col, head, w in zip(cols, headings, widths):
        tree.heading(col, text=head, anchor="w")
        tree.column(col, width=w, minwidth=60, anchor="w", stretch=True)

    vsb = ttk.Scrollbar(frame, orient="vertical",   command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid( row=0, column=1, sticky="ns")
    hsb.grid( row=1, column=0, sticky="ew")
    frame.rowconfigure(0,    weight=1)
    frame.columnconfigure(0, weight=1)
    return frame, tree


# ─────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────

class AttendanceApp(tk.Tk):
    """Root Tk window — builds the sidebar + four swappable content frames."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Face Attendance System  v5")
        self.geometry("1160x700")
        self.minsize(900, 580)
        self.configure(bg=BG)

        self._apply_styles()
        self._build_ui()

        # Initial data load
        self._show_frame("dashboard")
        self._schedule_refresh()

    # ── styles ────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")

        # Base
        s.configure(".",
                    background=BG, foreground=FG,
                    fieldbackground=BG_PANEL,
                    bordercolor=BORDER,
                    darkcolor=BG, lightcolor=BG,
                    troughcolor=BG_PANEL, selectbackground=ACCENT,
                    selectforeground=BG)

        # Frames
        s.configure("Sidebar.TFrame", background=BG_PANEL)

        # Labels
        for name, fg, size, bold in [
            ("TLabel",       FG,     10, False),
            ("Dim.TLabel",   FG_DIM,  9, False),
            ("Head.TLabel",  FG,     16, True),
            ("Accent.TLabel",ACCENT, 11, True),
            ("Card.TLabel",  FG,     10, False),
            ("Stat.TLabel",  FG,     28, True),
        ]:
            font = ("", size, "bold") if bold else ("", size)
            s.configure(name, background=BG, foreground=fg, font=font)

        s.configure("CardBg.TLabel", background=BG_CARD, foreground=FG)
        s.configure("CardDim.TLabel", background=BG_CARD, foreground=FG_DIM, font=("", 9))
        s.configure("CardAccent.TLabel", background=BG_CARD,
                    foreground=ACCENT, font=("", 11, "bold"))

        # Buttons
        s.configure("TButton",
                    background=BG_CARD, foreground=FG,
                    bordercolor=BORDER, focuscolor="none",
                    relief="flat", padding=(12, 6))
        s.map("TButton",
              background=[("active", ACCENT)],
              foreground=[("active", BG)])

        for name, bg, hover in [
            ("Accent.TButton",  ACCENT, "#74a8e8"),
            ("Danger.TButton",  RED,    "#d96f88"),
            ("Success.TButton", GREEN,  "#8ac98a"),
        ]:
            s.configure(name, background=bg, foreground=BG,
                        font=("", 10, "bold"), relief="flat",
                        bordercolor=bg, focuscolor="none", padding=(12, 6))
            s.map(name, background=[("active", hover)])

        # Treeview
        s.configure("Treeview",
                    background=BG_CARD, fieldbackground=BG_CARD,
                    foreground=FG, rowheight=30, bordercolor=BORDER)
        s.configure("Treeview.Heading",
                    background=BG_PANEL, foreground=ACCENT,
                    relief="flat", font=("", 10, "bold"))
        s.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", BG)])

        # Entry / Combobox
        s.configure("TEntry",
                    fieldbackground=BG_PANEL, foreground=FG,
                    insertcolor=FG, bordercolor=BORDER)
        s.configure("TCombobox",
                    fieldbackground=BG_PANEL, foreground=FG)

        # Scrollbar
        s.configure("Vertical.TScrollbar",
                    background=BG_PANEL, troughcolor=BG, bordercolor=BG)
        s.configure("Horizontal.TScrollbar",
                    background=BG_PANEL, troughcolor=BG, bordercolor=BG)

        # Separator
        s.configure("TSeparator", background=BORDER)

    # ── layout ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Sidebar (fixed 210 px) ────────────────────────────────
        self._sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=210)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)
        self._build_sidebar(self._sidebar)

        # ── Content area ─────────────────────────────────────────
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Status bar (bottom of right)
        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(right, textvariable=self._status_var,
                 bg=BG_PANEL, fg=FG_DIM, anchor="w",
                 font=("", 9), padx=14, pady=4).pack(side="bottom", fill="x")

        # Content canvas (stacked Place geometry)
        self._canvas = tk.Frame(right, bg=BG)
        self._canvas.pack(fill="both", expand=True)

        # Build all frames
        self._frames: Dict[str, tk.Frame] = {
            "dashboard":  self._build_dashboard(),
            "students":   self._build_students(),
            "attendance": self._build_attendance(),
            "reports":    self._build_reports(),
        }
        for frame in self._frames.values():
            frame.place(relx=0, rely=0, relwidth=1, relheight=1, in_=self._canvas)

    def _build_sidebar(self, sidebar: ttk.Frame) -> None:
        # Logo area
        tk.Label(sidebar, text="🎓", font=("", 30),
                 bg=BG_PANEL, fg=ACCENT).pack(pady=(22, 2))
        tk.Label(sidebar, text="Attendance System",
                 font=("", 11, "bold"), bg=BG_PANEL, fg=FG).pack()
        tk.Label(sidebar, text="v5  ·  Pi 5 Optimised",
                 font=("", 8), bg=BG_PANEL, fg=FG_DIM).pack(pady=(2, 18))
        ttk.Separator(sidebar).pack(fill="x", padx=18, pady=2)

        # Navigation
        tk.Label(sidebar, text="NAVIGATION",
                 font=("", 8), bg=BG_PANEL, fg=FG_DIM).pack(
            anchor="w", padx=18, pady=(14, 4))

        self._nav_btns: Dict[str, tk.Button] = {}
        nav = [
            ("dashboard",  "⬛  Dashboard"),
            ("students",   "👤  Students"),
            ("attendance", "📋  Attendance"),
            ("reports",    "📊  Reports"),
        ]
        for key, label in nav:
            btn = tk.Button(
                sidebar, text=label, anchor="w",
                font=("", 10), bg=BG_PANEL, fg=FG,
                activebackground=ACCENT, activeforeground=BG,
                bd=0, pady=9, padx=20, cursor="hand2",
                command=lambda k=key: self._show_frame(k),
            )
            btn.pack(fill="x")
            self._nav_btns[key] = btn

        ttk.Separator(sidebar).pack(fill="x", padx=18, pady=14)

        # Quick actions
        tk.Label(sidebar, text="QUICK ACTIONS",
                 font=("", 8), bg=BG_PANEL, fg=FG_DIM).pack(
            anchor="w", padx=18, pady=(0, 6))

        ttk.Button(sidebar, text="▶  Take Attendance",
                   style="Success.TButton",
                   command=self._start_recognition).pack(
            fill="x", padx=16, pady=3)
        ttk.Button(sidebar, text="➕  Enroll Student",
                   style="Accent.TButton",
                   command=self._enroll_dialog).pack(
            fill="x", padx=16, pady=3)

        # Bottom info
        tk.Label(sidebar,
                 text="YuNet · ArcFace · SQLite",
                 font=("", 8), bg=BG_PANEL, fg=FG_DIM).pack(
            side="bottom", pady=12)

    # ─────────────────────────────────────────────────────────────
    #  Frame: Dashboard
    # ─────────────────────────────────────────────────────────────

    def _build_dashboard(self) -> tk.Frame:
        f = tk.Frame(self._canvas, bg=BG)

        # Page header
        hdr = tk.Frame(f, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(22, 4))
        ttk.Label(hdr, text="Dashboard", style="Head.TLabel").pack(side="left")
        ttk.Button(hdr, text="↻ Refresh",
                   command=self._refresh_dashboard).pack(side="right")

        # ── Stat cards ───────────────────────────────────────────
        cards_row = tk.Frame(f, bg=BG)
        cards_row.pack(fill="x", padx=18, pady=6)
        cards_row.columnconfigure((0, 1, 2, 3), weight=1, uniform="card")

        self._stat_enrolled = self._stat_card(
            cards_row, col=0, title="Enrolled", accent=ACCENT)
        self._stat_present  = self._stat_card(
            cards_row, col=1, title="Present Today", accent=GREEN)
        self._stat_absent   = self._stat_card(
            cards_row, col=2, title="Absent Today", accent=RED)
        self._stat_date     = self._stat_card(
            cards_row, col=3, title="Today's Date", accent=YELLOW, big=False)

        # ── Today's attendance table ──────────────────────────────
        ttk.Label(f, text="Today's Attendance",
                  style="Accent.TLabel").pack(anchor="w", padx=24, pady=(14, 4))

        tree_frame, self._dash_tree = _make_tree(
            f,
            cols     = ("time", "id", "name", "status"),
            headings = ("Time", "Student ID", "Name", "Status"),
            widths   = (100, 140, 300, 110),
            height   = 12,
        )
        tree_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        return f

    def _stat_card(self, parent, col: int, title: str,
                   accent: str, big: bool = True) -> tk.Label:
        card = tk.Frame(parent, bg=BG_CARD)
        card.grid(row=0, column=col, sticky="nsew", padx=6, pady=4, ipady=10)
        tk.Label(card, text=title, bg=BG_CARD,
                 fg=FG_DIM, font=("", 9)).pack(pady=(10, 2))
        val = tk.Label(card, text="—", bg=BG_CARD, fg=accent,
                       font=("", 26 if big else 14, "bold"))
        val.pack(pady=(0, 10))
        return val

    def _refresh_dashboard(self) -> None:
        today    = datetime.now().strftime("%Y-%m-%d")
        enrolled = database.get_student_count()
        records  = database.get_attendance_by_date(today)
        present  = len(records)
        absent   = max(0, enrolled - present)

        self._stat_enrolled.config(text=str(enrolled))
        self._stat_present.config( text=str(present))
        self._stat_absent.config(  text=str(absent))
        self._stat_date.config(    text=datetime.now().strftime("%b %d\n%Y"))

        self._dash_tree.delete(*self._dash_tree.get_children())
        for r in records:
            self._dash_tree.insert("", "end", values=(
                r["time"], r["student_id"], r["student_name"], r["status"]
            ))

    # ─────────────────────────────────────────────────────────────
    #  Frame: Students
    # ─────────────────────────────────────────────────────────────

    def _build_students(self) -> tk.Frame:
        f = tk.Frame(self._canvas, bg=BG)

        hdr = tk.Frame(f, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(22, 8))
        ttk.Label(hdr, text="Enrolled Students",
                  style="Head.TLabel").pack(side="left")
        ttk.Button(hdr, text="↻ Refresh",
                   command=self._refresh_students).pack(side="right", padx=4)
        ttk.Button(hdr, text="🗑 Remove Selected",
                   style="Danger.TButton",
                   command=self._remove_selected).pack(side="right", padx=4)
        ttk.Button(hdr, text="➕ Enroll New",
                   style="Accent.TButton",
                   command=self._enroll_dialog).pack(side="right", padx=4)

        tree_frame, self._students_tree = _make_tree(
            f,
            cols     = ("id", "name", "images", "emb", "enrolled_at"),
            headings = ("ID", "Name", "Images", "Emb ✓", "Enrolled At"),
            widths   = (120, 260, 70, 70, 200),
            height   = 18,
        )
        tree_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        return f

    def _refresh_students(self) -> None:
        self._students_tree.delete(*self._students_tree.get_children())
        for s in database.get_all_students():
            has_emb = "✓" if s.get("embedding") is not None else "✗"
            self._students_tree.insert("", "end", iid=s["id"], values=(
                s["id"], s["name"], s["image_count"], has_emb, s["enrolled_at"]
            ))

    def _remove_selected(self) -> None:
        import shutil
        sel = self._students_tree.selection()
        if not sel:
            messagebox.showinfo("Remove Student",
                                "Select a student row first.", parent=self)
            return
        sid     = sel[0]
        student = database.get_student(sid)
        if not student:
            return
        if not messagebox.askyesno(
            "Confirm Removal",
            f"Permanently remove '{student['name']}' (ID: {sid})?\n\n"
            "All images and attendance history will be kept.",
            parent=self,
        ):
            return

        folder  = f"{sid}_{student['name'].replace(' ', '_')}"
        img_dir = os.path.join(config.DATASET_DIR, folder)
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)

        database.delete_student(sid)
        emb_module.get_embedding_db().reload()
        self._refresh_students()
        self._refresh_dashboard()
        self._set_status(f"Removed: {student['name']} ({sid})")

    # ─────────────────────────────────────────────────────────────
    #  Frame: Attendance
    # ─────────────────────────────────────────────────────────────

    def _build_attendance(self) -> tk.Frame:
        f = tk.Frame(self._canvas, bg=BG)

        ttk.Label(f, text="Attendance Records",
                  style="Head.TLabel").pack(anchor="w", padx=24, pady=(22, 10))

        # Filter bar
        bar = tk.Frame(f, bg=BG)
        bar.pack(fill="x", padx=20, pady=(0, 8))

        tk.Label(bar, text="Date:", bg=BG, fg=FG).pack(side="left", padx=(0, 6))
        self._att_date_var = tk.StringVar(
            value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(bar, textvariable=self._att_date_var,
                  width=14).pack(side="left")
        ttk.Button(bar, text="View",
                   command=self._refresh_attendance).pack(side="left", padx=8)
        ttk.Button(bar, text="Today",
                   command=self._att_today).pack(side="left")

        self._att_count_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self._att_count_var,
                  style="Accent.TLabel").pack(side="right", padx=8)

        tree_frame, self._att_tree = _make_tree(
            f,
            cols     = ("time", "id", "name", "status"),
            headings = ("Time", "Student ID", "Name", "Status"),
            widths   = (100, 140, 320, 110),
            height   = 18,
        )
        tree_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        return f

    def _att_today(self) -> None:
        self._att_date_var.set(datetime.now().strftime("%Y-%m-%d"))
        self._refresh_attendance()

    def _refresh_attendance(self) -> None:
        d = (self._att_date_var.get().strip()
             or datetime.now().strftime("%Y-%m-%d"))
        records = database.get_attendance_by_date(d)
        self._att_tree.delete(*self._att_tree.get_children())
        for r in records:
            self._att_tree.insert("", "end", values=(
                r["time"], r["student_id"], r["student_name"], r["status"]
            ))
        self._att_count_var.set(f"Total present: {len(records)}")

    # ─────────────────────────────────────────────────────────────
    #  Frame: Reports
    # ─────────────────────────────────────────────────────────────

    def _build_reports(self) -> tk.Frame:
        f = tk.Frame(self._canvas, bg=BG)

        ttk.Label(f, text="Reports",
                  style="Head.TLabel").pack(anchor="w", padx=24, pady=(22, 8))

        # ── Full Attendance Report ────────────────────────────────
        r1_frame, r1_vars = self._report_card(
            f,
            title       = "Full Attendance Report",
            desc        = "Exports all attendance records between two dates to CSV.",
            fields      = [("Start Date", "start", "YYYY-MM-DD"),
                           ("End Date",   "end",   "YYYY-MM-DD")],
            btn_label   = "Generate CSV",
        )
        r1_frame.pack(fill="x", padx=20, pady=6)
        ttk.Button(r1_frame, text="Generate CSV", style="Accent.TButton",
                   command=lambda: self._run_report(
                       lambda: reports.generate_attendance_report(
                           start_date=r1_vars["start"].get() or None,
                           end_date  =r1_vars["end"].get()   or None,
                       )
                   )).grid(row=4, column=0, columnspan=2,
                           sticky="w", padx=14, pady=(6, 14))

        # ── Absent Students Report ────────────────────────────────
        r2_frame, r2_vars = self._report_card(
            f,
            title     = "Absent Students Report",
            desc      = ("Lists students absent on a given date. "
                         + ("Email enabled." if config.EMAIL_ENABLED
                            else "Set EMAIL_ENABLED=True in config.py to email.")),
            fields    = [("Date", "date", "YYYY-MM-DD (blank = today)")],
            btn_label = "Generate",
        )
        r2_frame.pack(fill="x", padx=20, pady=6)
        ttk.Button(r2_frame, text="Generate & Email" if config.EMAIL_ENABLED
                   else "Generate CSV",
                   style="Accent.TButton",
                   command=lambda: self._run_report(
                       lambda: reports.generate_absent_report(
                           target_date=r2_vars["date"].get() or None,
                           send_email =config.EMAIL_ENABLED,
                       )
                   )).grid(row=3, column=0, columnspan=2,
                           sticky="w", padx=14, pady=(6, 14))

        # ── Daily Summary ─────────────────────────────────────────
        r3_frame, r3_vars = self._report_card(
            f,
            title     = "Daily Summary",
            desc      = "Compact present/absent summary for one day.",
            fields    = [("Date", "date", "YYYY-MM-DD (blank = today)")],
            btn_label = "Generate",
        )
        r3_frame.pack(fill="x", padx=20, pady=6)
        ttk.Button(r3_frame, text="Generate CSV", style="Accent.TButton",
                   command=lambda: self._run_report(
                       lambda: reports.generate_daily_summary(
                           target_date=r3_vars["date"].get() or None,
                       )
                   )).grid(row=3, column=0, columnspan=2,
                           sticky="w", padx=14, pady=(6, 14))

        # Output log
        ttk.Label(f, text="Output Log",
                  style="Accent.TLabel").pack(anchor="w", padx=24, pady=(10, 4))
        self._report_log = tk.Text(
            f, height=5, bg=BG_PANEL, fg=FG_DIM,
            font=("Courier", 9), bd=0, insertbackground=FG,
            state="disabled",
        )
        self._report_log.pack(fill="x", padx=20, pady=(0, 20))
        return f

    def _report_card(
        self, parent, title: str, desc: str,
        fields: List[Tuple[str, str, str]], btn_label: str,
    ) -> Tuple[tk.Frame, Dict[str, tk.StringVar]]:
        card = tk.Frame(parent, bg=BG_CARD)
        card.columnconfigure(1, weight=1)

        tk.Label(card, text=title, bg=BG_CARD, fg=ACCENT,
                 font=("", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 2))
        tk.Label(card, text=desc, bg=BG_CARD, fg=FG_DIM,
                 font=("", 9), wraplength=600, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 8))

        vars_dict: Dict[str, tk.StringVar] = {}
        for i, (lbl, key, placeholder) in enumerate(fields):
            tk.Label(card, text=lbl + ":", bg=BG_CARD,
                     fg=FG, font=("", 10)).grid(
                row=2 + i, column=0, sticky="w", padx=14, pady=4)
            var = tk.StringVar()
            vars_dict[key] = var
            entry = ttk.Entry(card, textvariable=var, width=28)
            entry.grid(row=2 + i, column=1, sticky="w", padx=6, pady=4)
            # Grey placeholder text
            entry.insert(0, placeholder)
            entry.configure(foreground=FG_DIM)
            entry.bind("<FocusIn>",  lambda e, en=entry, ph=placeholder: (
                (en.delete(0, "end"), en.configure(foreground=FG))
                if en.get() == ph else None))
            entry.bind("<FocusOut>", lambda e, en=entry, v=var, ph=placeholder: (
                (en.insert(0, ph), en.configure(foreground=FG_DIM))
                if not v.get() else None))

        return card, vars_dict

    def _run_report(self, fn: Callable) -> None:
        def _task():
            try:
                path = fn()
                msg  = f"✓  Saved → {path}"
            except Exception as exc:
                msg  = f"✗  Error: {exc}"
            self.after(0, lambda: self._log_report(msg))
            self.after(0, lambda: self._set_status(msg[:80]))
        threading.Thread(target=_task, daemon=True).start()

    def _log_report(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._report_log.configure(state="normal")
        self._report_log.insert("end", f"[{ts}]  {msg}\n")
        self._report_log.see("end")
        self._report_log.configure(state="disabled")

    # ─────────────────────────────────────────────────────────────
    #  Enrollment dialog
    # ─────────────────────────────────────────────────────────────

    def _enroll_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Enroll New Student")
        dlg.geometry("380x240")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        tk.Label(dlg, text="Enroll New Student", bg=BG, fg=ACCENT,
                 font=("", 14, "bold")).pack(pady=(20, 14))

        form = tk.Frame(dlg, bg=BG)
        form.pack(padx=34, fill="x")
        form.columnconfigure(1, weight=1)

        tk.Label(form, text="Student ID:", bg=BG,
                 fg=FG, font=("", 10)).grid(row=0, column=0, sticky="w", pady=8)
        id_var = tk.StringVar()
        ttk.Entry(form, textvariable=id_var, width=24).grid(
            row=0, column=1, padx=12, pady=8, sticky="ew")

        tk.Label(form, text="Full Name:", bg=BG,
                 fg=FG, font=("", 10)).grid(row=1, column=0, sticky="w", pady=8)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(form, textvariable=name_var, width=24)
        name_entry.grid(row=1, column=1, padx=12, pady=8, sticky="ew")

        tk.Label(dlg,
                 text=f"Camera will open and capture {config.ENROLL_IMAGES_COUNT} face shots.",
                 bg=BG, fg=FG_DIM, font=("", 9)).pack(pady=(0, 10))

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack()

        def _start():
            sid  = id_var.get().strip()
            name = name_var.get().strip()
            if not sid:
                messagebox.showwarning("Missing ID",
                    "Student ID is required.", parent=dlg)
                return
            if not name:
                messagebox.showwarning("Missing Name",
                    "Student Name is required.", parent=dlg)
                return
            dlg.destroy()

            def _enroll():
                import enroll
                ok = enroll.enroll_student(sid, name)
                self.after(0, self._refresh_students)
                self.after(0, self._refresh_dashboard)
                msg = f"Enrolled: {name} ({sid})" if ok else "Enrollment aborted or failed."
                self.after(0, lambda: self._set_status(msg))

            threading.Thread(target=_enroll, daemon=True).start()

        # Allow Enter key to submit
        dlg.bind("<Return>", lambda _e: _start())

        ttk.Button(btn_row, text="Start Camera  ▶",
                   style="Accent.TButton",
                   command=_start).pack(side="left", padx=8)
        ttk.Button(btn_row, text="Cancel",
                   command=dlg.destroy).pack(side="left", padx=8)

        name_entry.focus_set()

    # ─────────────────────────────────────────────────────────────
    #  Recognition
    # ─────────────────────────────────────────────────────────────

    def _start_recognition(self) -> None:
        if database.get_student_count() == 0:
            messagebox.showinfo("No Students Enrolled",
                "Enroll at least one student before taking attendance.",
                parent=self)
            return

        self._set_status(
            "Recognition session running — press  Q  in the camera window to stop.")

        def _run():
            import Recognition as recognition
            recognition.run_recognition()
            self.after(0, self._refresh_dashboard)
            self.after(0, self._refresh_attendance)
            self.after(0, lambda: self._set_status("Recognition session ended."))

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────────
    #  Navigation + auto-refresh
    # ─────────────────────────────────────────────────────────────

    def _show_frame(self, key: str) -> None:
        self._frames[key].lift()

        # Update nav button highlights
        for k, btn in self._nav_btns.items():
            btn.configure(
                bg=ACCENT if k == key else BG_PANEL,
                fg=BG     if k == key else FG,
            )

        # Refresh the newly visible frame's data
        {
            "dashboard":  self._refresh_dashboard,
            "students":   self._refresh_students,
            "attendance": self._refresh_attendance,
        }.get(key, lambda: None)()

    def _schedule_refresh(self) -> None:
        """Auto-refresh dashboard every 30 s."""
        self._refresh_dashboard()
        self.after(30_000, self._schedule_refresh)

    # ─────────────────────────────────────────────────────────────
    #  Status bar
    # ─────────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._status_var.set(f"[{ts}]  {msg}")


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

def launch() -> None:
    """Bootstrap directories + DB, then open the GUI window."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(config.DATASET_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    database.init_db()

    app = AttendanceApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
