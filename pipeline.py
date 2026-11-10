# The whole flow, in order. Read this file top to bottom to see what happens
# to a pilot's transmission:
#
#   1. stage_one:  speech to text, then the pilot's intent. Garbled and distress calls go to the human.
#   2. the queue:  yellow (time-sensitive) requests are answered before green (routine) ones.
#   3. stage_two:  work out who is calling, draft a reply, check its confidence, then send it.
#
# A human operator is called in for exactly three reasons:
#   "garbled" / "unclear_intent"  we could not understand the call
#   "distress"                    mayday or pan-pan
#   "low_confidence"              the drafted reply scored too low to send

import heapq
import time
from typing import Optional

from pydantic import BaseModel

import config
import database
import models
from models import ATCResponse, IntentResult


# ---------------------------------------------------------------------------
# The shapes of the data that moves through the pipeline
# ---------------------------------------------------------------------------


class Transmission(BaseModel):
    unit_id: str
    audio_path: Optional[str] = None  # a recording, or...
    text: Optional[str] = None  # ...typed text for testing


class QueuedItem(BaseModel):
    transmission: Transmission
    transcript: str
    intent: IntentResult


# ---------------------------------------------------------------------------
# The queue: yellow before green. Within the same colour, first come, first served.
# ---------------------------------------------------------------------------

PRIORITY = {"yellow": 0, "green": 1}


class IntentQueue:
    def __init__(self):
        self.items = []
        self.arrival_number = 0

    def add(self, urgency, item):
        heapq.heappush(self.items, (PRIORITY[urgency], self.arrival_number, item))
        self.arrival_number = self.arrival_number + 1

    def pop(self):
        priority, arrival_number, item = heapq.heappop(self.items)
        return item

    def is_empty(self):
        return len(self.items) == 0


# ---------------------------------------------------------------------------
# The arbiter: only sends when the channel is free.
# There is no real radio or datalink yet. send() prints and logs what would go out.
# ---------------------------------------------------------------------------


class Arbiter:
    def __init__(self):
        # The radio receiver would set this to True while a pilot is keyed up.
        self.pilot_is_transmitting = False

    def wait_for_clear_channel(self, max_wait_seconds=5):
        waited = 0
        while self.pilot_is_transmitting and waited < max_wait_seconds:
            time.sleep(0.2)
            waited = waited + 0.2
        return not self.pilot_is_transmitting

    def send(self, unit_id, callsign, spoken_text, voice_file, cpdlc_text, source="model"):
        if not self.wait_for_clear_channel():
            database.log_event("send_failed", unit_id, {"reason": "channel busy"})
            return False

        # Real radio and datalink sending goes here.
        print("VOICE to " + unit_id + ": " + spoken_text)
        print("CPDLC to " + unit_id + ": " + cpdlc_text)

        database.log_event(
            "transmitted",
            unit_id,
            {
                "callsign": callsign,
                "spoken_text": spoken_text,
                "voice_file": voice_file,
                "cpdlc_text": cpdlc_text,
                "source": source,
            },
        )

        # Only the model's own replies are saved to the tracker.
        # Replies from the human operator are logged above but never stored there.
        if source == "model" and callsign != "":
            database.record_clearance(callsign, spoken_text)

        return True


arbiter = Arbiter()


# ---------------------------------------------------------------------------
# Stage one: what did the pilot say, and what do they want?
# ---------------------------------------------------------------------------


def read_transmission(transmission):
    if transmission.audio_path is not None:
        transcript, confidence = models.transcribe(transmission.audio_path)
    else:
        transcript = transmission.text
        confidence = 1.0

    database.log_event("transcribed", transmission.unit_id, {"transcript": transcript, "confidence": confidence})
    return transcript, confidence


def stage_one(transmission, queue):
    """Speech to text and intent. Garbled and distress calls go to the human. The rest are queued."""
    unit_id = transmission.unit_id
    transcript, transcript_confidence = read_transmission(transmission)

    if transcript == "" or transcript_confidence < config.MIN_TRANSCRIPT_CONFIDENCE:
        database.escalate("garbled", unit_id, transcript)
        return

    try:
        intent = models.extract_intent(transcript)
    except ValueError:
        # The model answered, but not in the shape we asked for.
        database.escalate("unclear_intent", unit_id, transcript)
        return

    database.log_event("intent_extracted", unit_id, intent.model_dump())

    if intent.urgency == "red":
        database.escalate("distress", unit_id, transcript)
        return

    if intent.confidence < config.MIN_INTENT_CONFIDENCE:
        database.escalate("unclear_intent", unit_id, transcript)
        return

    queue.add(intent.urgency, QueuedItem(transmission=transmission, transcript=transcript, intent=intent))


# ---------------------------------------------------------------------------
# Stage two: who is calling, what do we say back, and is it safe to send?
# ---------------------------------------------------------------------------


def clean_callsign(text):
    # "n123 ab" and "N123AB" should count as the same callsign
    return text.replace(" ", "").replace("-", "").upper()


def find_callsign(unit_id, spoken_callsign):
    """Returns the callsign for this unit ID, or an empty string if we can't tell."""
    spoken = clean_callsign(spoken_callsign)

    known_callsign = database.lookup_callsign(unit_id)
    if known_callsign is not None:
        if spoken != "" and spoken != known_callsign:
            database.log_event("callsign_disagrees", unit_id, {"registry": known_callsign, "spoken": spoken})
        return known_callsign

    # First contact: only trust the spoken callsign if the tracker agrees.
    if spoken == "":
        return ""
    if not database.is_in_range(spoken):
        return ""
    if database.callsign_is_bound(spoken):
        return ""

    database.bind_callsign(unit_id, spoken)
    return spoken


def stage_two(item):
    """Identity, reply, confidence check, then send."""
    unit_id = item.transmission.unit_id
    callsign = find_callsign(unit_id, item.intent.spoken_callsign)

    if callsign == "":
        # We don't know who this is, so ask. No model needed for this.
        response = ATCResponse(callsign="", phraseology="Aircraft calling tower, say your callsign.", confidence=1.0)
    else:
        aircraft = database.get_aircraft(callsign)
        response = models.write_response(callsign, item.intent, aircraft)

    database.log_event("response_drafted", unit_id, response.model_dump())

    if response.confidence < config.MIN_RESPONSE_CONFIDENCE:
        database.escalate("low_confidence", unit_id, item.transcript, draft_reply=response.phraseology)
        return

    cpdlc = models.write_cpdlc(response)
    voice_file = models.speak(response.phraseology, unit_id)
    arbiter.send(unit_id, response.callsign, response.phraseology, voice_file, cpdlc.text)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def process_transmissions(transmissions):
    """Runs a batch of calls: stage one for all of them, then stage two in queue order."""
    database.create_tables()

    queue = IntentQueue()
    for transmission in transmissions:
        stage_one(transmission, queue)

    while not queue.is_empty():
        stage_two(queue.pop())


def answer_case(case_id, reply_text):
    """The human operator answers a case. The reply goes out like any other, but is not saved to the tracker."""
    case = database.get_case(case_id)
    database.mark_case_answered(case_id, reply_text)
    arbiter.send(case["unit_id"], "", reply_text, "", "", source="human")
