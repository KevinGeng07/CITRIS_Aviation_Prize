# The two jobs that run once a day (see dags/atc_replay_dag.py):
#
#   replay_recordings()  runs saved recordings through the pipeline
#   write_qc_report()    writes a CSV of one day's escalated cases, for quality control and retraining

import csv
import os
import shutil
from datetime import date

import config
import database
import pipeline


def find_new_recordings():
    """Returns the paths of the .wav files waiting in the recordings folder."""
    os.makedirs(config.RECORDINGS_FOLDER, exist_ok=True)

    paths = []
    for file_name in sorted(os.listdir(config.RECORDINGS_FOLDER)):
        if file_name.endswith(".wav"):
            paths.append(os.path.join(config.RECORDINGS_FOLDER, file_name))
    return paths


def replay_recordings():
    """Replays every new recording, then moves it to recordings/processed so it is never replayed twice.
    Name recordings like "unit-001_morning_call.wav": the unit ID is the part before the first "_".
    Returns how many were replayed."""
    paths = find_new_recordings()

    transmissions = []
    for path in paths:
        unit_id = os.path.basename(path).split("_")[0]
        transmissions.append(pipeline.Transmission(unit_id=unit_id, audio_path=path))

    pipeline.process_transmissions(transmissions)

    # Only move the files once the whole batch has been processed.
    os.makedirs(config.PROCESSED_RECORDINGS_FOLDER, exist_ok=True)
    for path in paths:
        shutil.move(path, config.PROCESSED_RECORDINGS_FOLDER)

    return len(paths)


def write_qc_report(day=None):
    """Writes the cases escalated on one day. `day` looks like "2026-11-09". Default is today.
    Returns the path of the CSV file."""
    if day is None:
        day = str(date.today())

    database.create_tables()
    # created_at looks like "2026-11-09T14:03:22", so this matches every case from that day.
    cases = database.fetch_all("SELECT * FROM escalations WHERE created_at LIKE ? ORDER BY id", (day + "%",))

    columns = ["id", "created_at", "reason", "unit_id", "transcript", "draft_reply", "status", "human_reply"]

    os.makedirs(config.QC_REPORT_FOLDER, exist_ok=True)
    file_path = os.path.join(config.QC_REPORT_FOLDER, "qc_" + day + ".csv")
    with open(file_path, "w", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(columns)
        for case in cases:
            writer.writerow([case[column] for column in columns])

    return file_path
