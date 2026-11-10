# These tests use fake models, so they run without a GPU, Ollama, or any downloads.

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
import models
import pipeline
from models import ATCResponse, CPDLCMessage, IntentResult
from pipeline import IntentQueue, Transmission


@pytest.fixture(autouse=True)
def fresh_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_FILE", str(tmp_path / "test.db"))
    database.create_tables()
    database.update_position("N123AB", 4500)
    database.update_position("DAL482", 11000)


def fake_intent(urgency="green", callsign="N123AB", confidence=0.95):
    def extract_intent(transcript):
        return IntentResult(
            urgency=urgency, intent="radio check", spoken_callsign=callsign, confidence=confidence
        )

    return extract_intent


def fake_response(confidence=0.95):
    def write_response(callsign, intent, aircraft):
        return ATCResponse(
            callsign=callsign, phraseology=callsign + ", radio check loud and clear.", confidence=confidence
        )

    return write_response


def use_fake_models(monkeypatch, intent=None, response=None):
    monkeypatch.setattr(models, "extract_intent", intent or fake_intent())
    monkeypatch.setattr(models, "write_response", response or fake_response())
    monkeypatch.setattr(models, "write_cpdlc", lambda r: CPDLCMessage(element="TEST", text="TEST MESSAGE"))
    monkeypatch.setattr(models, "speak", lambda text, name: "")


def reasons_escalated():
    return [case["reason"] for case in database.get_open_cases()]


def test_yellow_is_served_before_green():
    queue = IntentQueue()
    queue.add("green", "first green")
    queue.add("yellow", "yellow")
    queue.add("green", "second green")

    assert queue.pop() == "yellow"
    assert queue.pop() == "first green"
    assert queue.pop() == "second green"


def test_good_call_is_sent_and_saved(monkeypatch):
    use_fake_models(monkeypatch)
    call = Transmission(unit_id="unit-1", text="Tower, N123AB, radio check.")

    pipeline.process_transmissions([call])

    assert len(database.get_events("transmitted")) == 1
    assert reasons_escalated() == []
    assert "radio check" in database.get_aircraft("N123AB")["last_clearance"]


def test_unit_id_is_remembered_after_first_call(monkeypatch):
    use_fake_models(monkeypatch)
    pipeline.process_transmissions([Transmission(unit_id="unit-1", text="Tower, N123AB, radio check.")])
    assert database.lookup_callsign("unit-1") == "N123AB"

    # Second call: the pilot does not say a callsign, but the registry knows who it is.
    monkeypatch.setattr(models, "extract_intent", fake_intent(callsign=""))
    pipeline.process_transmissions([Transmission(unit_id="unit-1", text="Tower, radio check again.")])

    assert len(database.get_events("transmitted")) == 2
    assert reasons_escalated() == []


def test_unknown_callsign_gets_a_say_callsign_reply(monkeypatch):
    use_fake_models(monkeypatch, intent=fake_intent(callsign="N000XX"))

    pipeline.process_transmissions([Transmission(unit_id="unit-9", text="Tower, N000XX, radio check.")])

    assert database.lookup_callsign("unit-9") is None
    assert len(database.get_events("transmitted")) == 1
    assert reasons_escalated() == []


def test_garbled_transcript_goes_to_human(monkeypatch):
    use_fake_models(monkeypatch)
    monkeypatch.setattr(models, "transcribe", lambda path: ("krrr tower krrr", 0.2))

    pipeline.process_transmissions([Transmission(unit_id="unit-1", audio_path="fake.wav")])

    assert reasons_escalated() == ["garbled"]
    assert len(database.get_events("transmitted")) == 0


def test_unclear_intent_goes_to_human(monkeypatch):
    use_fake_models(monkeypatch, intent=fake_intent(confidence=0.3))

    pipeline.process_transmissions([Transmission(unit_id="unit-1", text="Tower, uh, requesting")])

    assert reasons_escalated() == ["unclear_intent"]
    assert len(database.get_events("transmitted")) == 0


def test_distress_goes_to_human(monkeypatch):
    use_fake_models(monkeypatch, intent=fake_intent(urgency="red"))

    pipeline.process_transmissions([Transmission(unit_id="unit-1", text="Mayday mayday, N123AB")])

    assert reasons_escalated() == ["distress"]
    assert len(database.get_events("transmitted")) == 0


def test_low_confidence_reply_goes_to_human_and_is_not_saved(monkeypatch):
    use_fake_models(monkeypatch, response=fake_response(confidence=0.4))

    pipeline.process_transmissions([Transmission(unit_id="unit-1", text="Tower, N123AB, radio check.")])

    assert reasons_escalated() == ["low_confidence"]
    assert len(database.get_events("transmitted")) == 0
    assert database.get_aircraft("N123AB")["last_clearance"] is None


def test_human_reply_is_sent_but_not_saved_to_tracker(monkeypatch):
    use_fake_models(monkeypatch, intent=fake_intent(urgency="red"))
    pipeline.process_transmissions([Transmission(unit_id="unit-1", text="Mayday mayday, N123AB")])
    case = database.get_open_cases()[0]

    pipeline.answer_case(case["id"], "N123AB, cleared to land runway 28.")

    sent = database.get_events("transmitted")
    assert len(sent) == 1
    assert database.get_aircraft("N123AB")["last_clearance"] is None
    assert database.get_open_cases() == []
