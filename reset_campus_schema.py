"""
Reset only the campus simulation tables and recreate them with the GUI-ready schema.
This does NOT delete enrolled students, embeddings, or attendance records.
Run from the same folder as main.py/gui.py:
    python reset_campus_schema.py
"""

import database
import campus_monitoring

if __name__ == "__main__":
    database.init_db()
    campus_monitoring.reset_campus_tables()
    campus_monitoring.init_monitoring_db()
    print("Campus simulation schema reset successfully.")
    print("Students, embeddings and normal attendance records were not deleted.")
