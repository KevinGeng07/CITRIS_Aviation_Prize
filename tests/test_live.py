# Tests for the live tower. They use fake models, so no GPU or Ollama is needed.

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database
import live
import models
from models import IntentResult
from pipeline import IntentQueue, Transmission
from test_pipeline import use_fake_models


@pytest.fixture(autouse=True)
def fresh_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_FILE", str(tmp_path / "test.db"))
    database.create_tables()
    database.update_position("N123AB", 4500)
    database.update_position("DAL482", 11000)


def fake_intent_from_text(transcript):
    """Landing requests are yellow, everything else is green. The callsign is taken from the text."""
    callsign = "DAL482" if "DAL482" in transcript else "N123AB"
    urgency = "yellow" if "landing" in transcript else "green"
    return IntentResult(urgency=urgency, intent=transcript, spoken_callsign=callsign, confidence=0.95)


def sent_to():
    return [event["unit_id"] for event in database.get_events("transmitted")]


def test_parse_line_reads_typed_text():
    call = live.parse_line("unit-001: Tower, N123AB, radio check.\n")
    assert call.unit_id == "unit-001"
    assert call.text == "Tower, N123AB, radio check."
    assert call.audio_path is None


def test_parse_line_reads_a_recording_path():
    call = live.parse_line("unit-002: recordings/call.wav")
    assert call.unit_id == "unit-002"
    assert call.audio_path == "recordings/call.wav"


def test_parse_line_rejects_lines_without_a_unit_id():
    assert live.parse_line("Tower, N123AB, radio check.") is None
    assert live.parse_line("unit-001:") is None


def test_waits_for_a_call_then_answers_it(monkeypatch):
    use_fake_models(monkeypatch)

    async def scenario():
        incoming = asyncio.Queue()
        worker = asyncio.create_task(live.answer_waiting_calls(incoming, IntentQueue()))

        # Nothing has been said yet, so the tower is still waiting.
        await asyncio.sleep(0.05)
        assert not worker.done()

        await incoming.put(Transmission(unit_id="unit-001", text="Tower, N123AB, radio check."))
        return await worker

    keep_going = asyncio.run(scenario())

    assert keep_going is True
    assert sent_to() == ["unit-001"]


def test_calls_that_pile_up_are_answered_yellow_first(monkeypatch):
    use_fake_models(monkeypatch)
    monkeypatch.setattr(models, "extract_intent", fake_intent_from_text)

    async def scenario():
        incoming = asyncio.Queue()
        # Both calls arrive before the tower gets to them: the radio check first, the landing second.
        await incoming.put(Transmission(unit_id="unit-001", text="Tower, N123AB, radio check."))
        await incoming.put(Transmission(unit_id="unit-002", text="Tower, DAL482, requesting landing."))
        return await live.answer_waiting_calls(incoming, IntentQueue())

    asyncio.run(scenario())

    assert sent_to() == ["unit-002", "unit-001"]


def test_stops_when_the_input_closes(monkeypatch):
    use_fake_models(monkeypatch)

    async def scenario():
        incoming = asyncio.Queue()
        await incoming.put(Transmission(unit_id="unit-001", text="Tower, N123AB, radio check."))
        await incoming.put(None)
        await live.answer_calls(incoming)  # would hang forever if it did not stop

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))

    # The call that came in before the stop is still answered.
    assert sent_to() == ["unit-001"]
