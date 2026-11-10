# Tests for the daily replay job and the QC report. They use fake models and a
# temporary folder, so no GPU, Ollama, Airflow, or real audio is needed.

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
import jobs
import models
from test_pipeline import use_fake_models


@pytest.fixture(autouse=True)
def fresh_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_FILE", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "RECORDINGS_FOLDER", str(tmp_path / "recordings"))
    monkeypatch.setattr(config, "PROCESSED_RECORDINGS_FOLDER", str(tmp_path / "recordings" / "processed"))
    monkeypatch.setattr(config, "QC_REPORT_FOLDER", str(tmp_path / "qc_reports"))
    database.create_tables()
    database.update_position("N123AB", 4500)

    # Every "recording" sounds like a clear radio check.
    use_fake_models(monkeypatch)
    monkeypatch.setattr(models, "transcribe", lambda path: ("Tower, N123AB, radio check.", 0.95))


def add_recording(file_name):
    os.makedirs(config.RECORDINGS_FOLDER, exist_ok=True)
    open(os.path.join(config.RECORDINGS_FOLDER, file_name), "w").close()


def test_replay_runs_each_recording_and_moves_it():
    add_recording("unit-001_morning.wav")
    add_recording("unit-002_noon.wav")
    add_recording("notes.txt")

    count = jobs.replay_recordings()

    assert count == 2
    assert len(database.get_events("transcribed")) == 2
    # The unit ID comes from the part of the file name before the first "_".
    unit_ids = [event["unit_id"] for event in database.get_events("transcribed")]
    assert sorted(unit_ids) == ["unit-001", "unit-002"]
    # Replayed recordings are moved; other files are left alone.
    assert sorted(os.listdir(config.PROCESSED_RECORDINGS_FOLDER)) == ["unit-001_morning.wav", "unit-002_noon.wav"]
    assert os.path.exists(os.path.join(config.RECORDINGS_FOLDER, "notes.txt"))


def test_replay_never_repeats_a_recording():
    add_recording("unit-001_morning.wav")

    jobs.replay_recordings()
    second_count = jobs.replay_recordings()

    assert second_count == 0
    assert len(database.get_events("transcribed")) == 1


def test_replay_with_no_recordings_does_nothing():
    assert jobs.replay_recordings() == 0
    assert database.get_events("transcribed") == []


def test_qc_report_only_has_cases_from_that_day():
    old_case = database.escalate("garbled", "unit-001", "krrr")
    database.escalate("distress", "unit-002", "mayday")

    # Pretend the first case happened last week.
    connection = database.get_connection()
    connection.execute("UPDATE escalations SET created_at = '2026-11-01T09:00:00' WHERE id = ?", (old_case,))
    connection.commit()
    connection.close()

    today = database.now()[:10]
    file_path = jobs.write_qc_report(today)

    with open(file_path) as report_file:
        rows = list(csv.DictReader(report_file))
    assert [row["reason"] for row in rows] == ["distress"]
