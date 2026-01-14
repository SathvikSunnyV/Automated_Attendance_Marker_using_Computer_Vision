import os
import csv
from datetime import datetime
from config import ATTENDANCE_FOLDER

def mark_attendance(identity):
    date = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(ATTENDANCE_FOLDER, f"{date}.csv")

    already_marked = set()

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            already_marked = {row[0] for row in reader}

    if identity in already_marked:
        return False

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([identity, datetime.now().strftime("%H:%M:%S")])

    return True
