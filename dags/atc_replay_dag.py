# An Airflow job that runs once a day:
#   1. check_ollama:      stop early if the language model is not running
#   2. replay_recordings: run every new recording in the recordings folder through the pipeline
#   3. export_qc_report:  write a CSV of that day's cases that went to the human operator
#
# The live pipeline does not run inside Airflow, because Airflow is built for scheduled
# jobs and would add delay to every radio call. Live calls use pipeline.py directly.

import os
import sys
from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

# Let Airflow find the project files that sit one folder above this one.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jobs
import models


def check_ollama():
    models.check_ollama()


def replay_recordings():
    count = jobs.replay_recordings()
    print("Replayed " + str(count) + " recordings")


def export_qc_report(ds):
    # Airflow fills in `ds` with the run's date, like "2026-11-09".
    file_path = jobs.write_qc_report(ds)
    print("Wrote " + file_path)


with DAG(
    dag_id="atc_replay_and_qc",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    # One run at a time: SQLite and the GPU can only handle one job at once.
    max_active_runs=1,
    # If a task fails (for example Ollama was restarting), try again twice, 5 minutes apart.
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
) as dag:
    check_task = PythonOperator(task_id="check_ollama", python_callable=check_ollama)
    replay_task = PythonOperator(task_id="replay_recordings", python_callable=replay_recordings)
    report_task = PythonOperator(task_id="export_qc_report", python_callable=export_qc_report)

    check_task >> replay_task >> report_task
