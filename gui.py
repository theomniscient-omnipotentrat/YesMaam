"""
gui.py — Optional Tkinter-based launcher for the attendance system.
         Provides button-driven access to core features without a
         terminal.  Run with:  python gui.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils
import database
import attendance as att_module
import enroll as enroll_module
import recognize as rec_module

# ──────────────────────────────────────────────────────────────────
#  Bootstrap
# ──────────────────────────────────────────────────────────────────
utils.ensure_dir(config.DATASET_DIR)
utils.ensure_dir(config.REPORT_DIR)
database.init_db()
att_module.init_attendance_file()


# ──────────────────────────────────────────────────────────────────
#  Theme colours
# ──────────────────────────────────────────────────────────────────
BG         = "#1e1e2e"
BG_CARD    = "#2a2a3d"
FG         = "#cdd6f4"
ACCENT     = "#89b4fa"
SUCCESS    = "#a6e3a1"
DANGER     = "#f38ba8"
WARN       = "#f9e2af"
FONT_HEAD  = ("Helvetica", 18, "bold")
FONT_BODY  = ("Helvetica", 11)
FONT_MONO  = ("Courier", 10)


# ──────────────────────────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────────────────────────

class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("880x640")
        self.resizable(True, True)
        self.configure(bg=BG)

        self._build_header()
        self._build_body()
        self._build_status_bar()
        self._refresh_stats()

    # ── Header ────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=ACCENT, pady=8)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="🎓  Face Recognition Attendance System",
            font=FONT_HEAD,
            bg=ACCENT,
            fg=BG,
        ).pack()
        tk.Label(
            hdr,
            text="Raspberry Pi  |  MediaPipe + OpenCV LBPH",
            font=("Helvetica", 9),
            bg=ACCENT,
            fg=BG,
        ).pack()

    # ── Body ──────────────────────────────────────────────────────

    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left column — action buttons
        left = tk.Frame(body, bg=BG, width=220)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._build_buttons(left)

        # Right column — tabbed log + attendance table
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_notebook(right)

    def _build_buttons(self, parent):
        tk.Label(parent, text="Actions", font=("Helvetica", 13, "bold"),
                 bg=BG, fg=ACCENT).pack(pady=(0, 10))

        buttons = [
            ("👤  Enroll Student",       SUCCESS, self._enroll),
            ("📷  Take Attendance",      ACCENT,  self._take_attendance),
            ("📋  View Today",           FG,      self._view_today),
            ("📅  View by Date",         FG,      self._view_by_date),
            ("👥  All Students",         FG,      self._view_students),
            ("📊  Daily Report",         WARN,    self._daily_report),
            ("🔄  Retrain Model",        WARN,    self._retrain),
            ("❌  Exit",                 DANGER,  self.quit),
        ]

        for text, color, cmd in buttons:
            btn = tk.Button(
                parent,
                text=text,
                font=FONT_BODY,
                bg=BG_CARD,
                fg=color,
                activebackground=color,
                activeforeground=BG,
                relief="flat",
                bd=0,
                padx=10,
                pady=8,
                cursor="hand2",
                command=cmd,
            )
            btn.pack(fill="x", pady=3)

        # Stats card
        self.stats_frame = tk.Frame(parent, bg=BG_CARD, padx=10, pady=10)
        self.stats_frame.pack(fill="x", pady=(20, 0))
        tk.Label(self.stats_frame, text="Statistics", font=("Helvetica", 10, "bold"),
                 bg=BG_CARD, fg=ACCENT).pack(anchor="w")
        self.lbl_enrolled   = tk.Label(self.stats_frame, text="Enrolled:  0", font=FONT_BODY, bg=BG_CARD, fg=FG)
        self.lbl_enrolled.pack(anchor="w")
        self.lbl_present    = tk.Label(self.stats_frame, text="Present today:  0", font=FONT_BODY, bg=BG_CARD, fg=SUCCESS)
        self.lbl_present.pack(anchor="w")
        self.lbl_date       = tk.Label(self.stats_frame, text=utils.current_date(), font=FONT_BODY, bg=BG_CARD, fg=WARN)
        self.lbl_date.pack(anchor="w")

    def _build_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        # Log tab
        log_frame = tk.Frame(self.notebook, bg=BG_CARD)
        self.notebook.add(log_frame, text="  System Log  ")
        self.log_text = scrolledtext.ScrolledText(
            log_frame, bg=BG_CARD, fg=FG, font=FONT_MONO,
            insertbackground=FG, state="disabled", wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Attendance tab
        att_frame = tk.Frame(self.notebook, bg=BG_CARD)
        self.notebook.add(att_frame, text="  Attendance  ")
        self._build_attendance_table(att_frame)

    def _build_attendance_table(self, parent):
        cols = ("ID", "Name", "Date", "Time", "Status")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=BG_CARD, foreground=FG,
                        fieldbackground=BG_CARD, rowheight=26)
        style.configure("Treeview.Heading", background=ACCENT, foreground=BG, font=("Helvetica", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

        self.att_tree = ttk.Treeview(parent, columns=cols, show="headings", style="Treeview")
        widths = [90, 180, 100, 90, 80]
        for col, w in zip(cols, widths):
            self.att_tree.heading(col, text=col)
            self.att_tree.column(col, width=w, anchor="center")
        self.att_tree.pack(fill="both", expand=True, padx=4, pady=4)

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.att_tree.yview)
        self.att_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    # ── Status bar ────────────────────────────────────────────────

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        bar = tk.Label(self, textvariable=self.status_var, bg=BG_CARD, fg=FG,
                       font=("Helvetica", 9), anchor="w", padx=10)
        bar.pack(fill="x", side="bottom")

    # ── Helpers ───────────────────────────────────────────────────

    def _log(self, msg: str, color: str = FG):
        self.log_text.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def _refresh_stats(self):
        enrolled = database.get_student_count()
        present  = len(att_module.get_attendance_by_date())
        self.lbl_enrolled.config(text=f"Enrolled:  {enrolled}")
        self.lbl_present.config(text=f"Present today:  {present}")
        self.after(5000, self._refresh_stats)

    def _refresh_attendance_tab(self, records):
        self.att_tree.delete(*self.att_tree.get_children())
        for r in records:
            self.att_tree.insert("", "end", values=(
                r["student_id"], r["student_name"], r["date"], r["time"], r["status"]
            ))

    # ── Button handlers ───────────────────────────────────────────

    def _enroll(self):
        dlg = EnrollDialog(self)
        self.wait_window(dlg)
        sid   = dlg.student_id
        sname = dlg.student_name
        if not sid or not sname:
            return
        self._set_status(f"Enrolling {sname} …")
        self._log(f"Enrolling student: {sid} — {sname}")
        threading.Thread(
            target=self._enroll_worker,
            args=(sid, sname),
            daemon=True,
        ).start()

    def _enroll_worker(self, sid, sname):
        ok = enroll_module.enroll_student(sid, sname)
        if ok:
            if os.path.isfile(config.MODEL_FILE):
                os.remove(config.MODEL_FILE)
            self.after(0, lambda: self._log(f"✓ Enrolled {sname}", SUCCESS))
            self.after(0, lambda: self._set_status("Enrolment complete."))
        else:
            self.after(0, lambda: self._log(f"✗ Enrolment failed for {sname}", DANGER))
            self.after(0, lambda: self._set_status("Enrolment failed."))
        self.after(0, self._refresh_stats)

    def _take_attendance(self):
        if database.get_student_count() == 0:
            messagebox.showwarning("No Students", "Enroll at least one student first.")
            return
        self._set_status("Recognition running … close the camera window to stop.")
        self._log("Starting recognition session …", ACCENT)
        threading.Thread(target=self._recognition_worker, daemon=True).start()

    def _recognition_worker(self):
        rec_module.run_recognition()
        self.after(0, lambda: self._set_status("Recognition session ended."))
        self.after(0, lambda: self._log("Recognition session ended.", WARN))
        self.after(0, self._refresh_stats)
        records = att_module.get_attendance_by_date()
        self.after(0, lambda: self._refresh_attendance_tab(records))

    def _view_today(self):
        records = att_module.get_attendance_by_date()
        self._refresh_attendance_tab(records)
        self.notebook.select(1)
        self._log(f"Today's attendance loaded ({len(records)} records).")

    def _view_by_date(self):
        date_str = _ask_date(self)
        if not date_str:
            return
        records = att_module.get_attendance_by_date(date_str)
        self._refresh_attendance_tab(records)
        self.notebook.select(1)
        self._log(f"Attendance for {date_str} loaded ({len(records)} records).")

    def _view_students(self):
        students = database.get_all_students()
        self._log(f"Enrolled students ({len(students)}):", ACCENT)
        for s in students:
            self._log(f"  • {s['id']:12} {s['name']:25} images={s['image_count']}")

    def _daily_report(self):
        date_str = _ask_date(self)
        if not date_str:
            date_str = utils.current_date()
        path = att_module.generate_daily_report(date_str)
        self._log(f"Daily report saved: {path}", SUCCESS)
        messagebox.showinfo("Report Saved", f"Report written to:\n{path}")

    def _retrain(self):
        self._set_status("Retraining model …")
        self._log("Retraining recognition model …", WARN)
        threading.Thread(target=self._retrain_worker, daemon=True).start()

    def _retrain_worker(self):
        if os.path.isfile(config.MODEL_FILE):
            os.remove(config.MODEL_FILE)
        model = rec_module.train_model(force=True)
        msg = "Model trained successfully." if model else "Training failed — enroll students first."
        color = SUCCESS if model else DANGER
        self.after(0, lambda: self._log(msg, color))
        self.after(0, lambda: self._set_status(msg))


# ──────────────────────────────────────────────────────────────────
#  Helper dialogs
# ──────────────────────────────────────────────────────────────────

class EnrollDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Enroll Student")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.student_id   = ""
        self.student_name = ""
        self._build()
        self.grab_set()
        self.transient(parent)

    def _build(self):
        f = tk.Frame(self, bg=BG, padx=24, pady=20)
        f.pack()
        tk.Label(f, text="Student ID:",   font=FONT_BODY, bg=BG, fg=FG).grid(row=0, column=0, sticky="e", pady=5)
        tk.Label(f, text="Student Name:", font=FONT_BODY, bg=BG, fg=FG).grid(row=1, column=0, sticky="e", pady=5)
        self._id_var   = tk.StringVar()
        self._name_var = tk.StringVar()
        tk.Entry(f, textvariable=self._id_var,   font=FONT_BODY, bg=BG_CARD, fg=FG, insertbackground=FG, width=24).grid(row=0, column=1, padx=8)
        tk.Entry(f, textvariable=self._name_var, font=FONT_BODY, bg=BG_CARD, fg=FG, insertbackground=FG, width=24).grid(row=1, column=1, padx=8)
        btns = tk.Frame(f, bg=BG)
        btns.grid(row=2, column=0, columnspan=2, pady=14)
        tk.Button(btns, text="Enroll", bg=SUCCESS, fg=BG, font=FONT_BODY, command=self._ok, padx=16).pack(side="left", padx=8)
        tk.Button(btns, text="Cancel", bg=BG_CARD, fg=FG, font=FONT_BODY, command=self.destroy, padx=16).pack(side="left")

    def _ok(self):
        self.student_id   = self._id_var.get().strip()
        self.student_name = self._name_var.get().strip()
        if not self.student_id or not self.student_name:
            messagebox.showwarning("Missing Info", "Both ID and Name are required.", parent=self)
            return
        self.destroy()


def _ask_date(parent) -> str:
    dlg = tk.Toplevel(parent)
    dlg.title("Select Date")
    dlg.configure(bg=BG)
    dlg.resizable(False, False)
    dlg.grab_set()
    result = {"date": ""}

    f = tk.Frame(dlg, bg=BG, padx=20, pady=16)
    f.pack()
    tk.Label(f, text="Date (YYYY-MM-DD):", font=FONT_BODY, bg=BG, fg=FG).pack(anchor="w")
    var = tk.StringVar(value=utils.current_date())
    tk.Entry(f, textvariable=var, font=FONT_BODY, bg=BG_CARD, fg=FG, insertbackground=FG, width=18).pack(pady=6)

    def ok():
        result["date"] = var.get().strip()
        dlg.destroy()

    tk.Button(f, text="OK", bg=ACCENT, fg=BG, font=FONT_BODY, command=ok, padx=14).pack(pady=4)
    parent.wait_window(dlg)
    return result["date"]


# ──────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
