# Everything that is saved lives here, in one SQLite file. There are four tables:
#
#   events       a log of everything that happened, for tuning and review
#   registry     which unit ID belongs to which callsign
#   aircraft     what we know about each aircraft in range (the "tracker")
#   escalations  the human operator's inbox
#
# Nothing about an escalated case is ever written to the aircraft table.

import json
import sqlite3
from datetime import datetime

import config


def get_connection():
    connection = sqlite3.connect(config.DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def now():
    return datetime.now().isoformat(timespec="seconds")


def run(sql, values=()):
    """Runs one change (INSERT, UPDATE, ...) and saves it. Returns the new row's ID."""
    connection = get_connection()
    cursor = connection.execute(sql, values)
    connection.commit()
    connection.close()
    return cursor.lastrowid


def fetch_one(sql, values=()):
    connection = get_connection()
    row = connection.execute(sql, values).fetchone()
    connection.close()
    return row


def fetch_all(sql, values=()):
    connection = get_connection()
    rows = connection.execute(sql, values).fetchall()
    connection.close()
    return rows


def create_tables():
    run(
        "CREATE TABLE IF NOT EXISTS events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, "
        "kind TEXT, unit_id TEXT, details TEXT)"
    )
    run("CREATE TABLE IF NOT EXISTS registry (unit_id TEXT PRIMARY KEY, callsign TEXT, bound_at TEXT)")
    run(
        "CREATE TABLE IF NOT EXISTS aircraft ("
        "callsign TEXT PRIMARY KEY, altitude_ft INTEGER, "
        "last_seen TEXT, last_clearance TEXT)"
    )
    run(
        "CREATE TABLE IF NOT EXISTS escalations ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, reason TEXT, "
        "unit_id TEXT, transcript TEXT, draft_reply TEXT, status TEXT, human_reply TEXT)"
    )


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


def log_event(kind, unit_id, details):
    run(
        "INSERT INTO events (created_at, kind, unit_id, details) VALUES (?, ?, ?, ?)",
        (now(), kind, unit_id, json.dumps(details)),
    )


def get_events(kind):
    return fetch_all("SELECT * FROM events WHERE kind = ? ORDER BY id", (kind,))


# ---------------------------------------------------------------------------
# Registry: which unit ID belongs to which callsign
# ---------------------------------------------------------------------------


def lookup_callsign(unit_id):
    """Returns the callsign for this unit ID, or None if we have never bound one."""
    row = fetch_one("SELECT callsign FROM registry WHERE unit_id = ?", (unit_id,))
    if row is None:
        return None
    return row["callsign"]


def callsign_is_bound(callsign):
    return fetch_one("SELECT unit_id FROM registry WHERE callsign = ?", (callsign,)) is not None


def bind_callsign(unit_id, callsign):
    run(
        "INSERT OR REPLACE INTO registry (unit_id, callsign, bound_at) VALUES (?, ?, ?)",
        (unit_id, callsign, now()),
    )
    log_event("registry_bound", unit_id, {"callsign": callsign})


# ---------------------------------------------------------------------------
# Aircraft tracker: what we know about each aircraft in range.
# In real use, update_position() would be called by an ADS-B / radar feed.
# ---------------------------------------------------------------------------


def get_aircraft(callsign):
    row = fetch_one("SELECT * FROM aircraft WHERE callsign = ?", (callsign,))
    if row is None:
        return None
    return dict(row)


def is_in_range(callsign):
    return get_aircraft(callsign) is not None


def update_position(callsign, altitude_ft):
    if get_aircraft(callsign) is None:
        run(
            "INSERT INTO aircraft (callsign, altitude_ft, last_seen) VALUES (?, ?, ?)",
            (callsign, altitude_ft, now()),
        )
    else:
        run(
            "UPDATE aircraft SET altitude_ft = ?, last_seen = ? WHERE callsign = ?",
            (altitude_ft, now(), callsign),
        )


def record_clearance(callsign, text):
    run("UPDATE aircraft SET last_clearance = ? WHERE callsign = ?", (text, callsign))


# ---------------------------------------------------------------------------
# Human operator's inbox
# ---------------------------------------------------------------------------


def escalate(reason, unit_id, transcript, draft_reply=""):
    """Puts a case in the human operator's inbox. Returns the case ID."""
    case_id = run(
        "INSERT INTO escalations (created_at, reason, unit_id, transcript, draft_reply, status) "
        "VALUES (?, ?, ?, ?, ?, 'open')",
        (now(), reason, unit_id, transcript, draft_reply),
    )
    log_event("escalated", unit_id, {"case_id": case_id, "reason": reason})
    print("ESCALATED to human (" + reason + "): " + unit_id + ": " + transcript)
    return case_id


def get_case(case_id):
    return fetch_one("SELECT * FROM escalations WHERE id = ?", (case_id,))


def get_open_cases():
    return fetch_all("SELECT * FROM escalations WHERE status = 'open' ORDER BY id")


def mark_case_answered(case_id, reply_text):
    run("UPDATE escalations SET status = 'answered', human_reply = ? WHERE id = ?", (reply_text, case_id))
