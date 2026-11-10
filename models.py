# All the open-source models are used here and only here.
# Whisper for speech to text, an Ollama LLM for the thinking, Kokoro for the voice.

import math
import os
import time
from typing import Literal

import numpy
import ollama
from pydantic import BaseModel

import config


# The shapes the LLM must answer in. Ollama is told to match these exactly.


class IntentResult(BaseModel):
    urgency: Literal["green", "yellow", "red"]
    intent: str
    spoken_callsign: str
    confidence: float


class ATCResponse(BaseModel):
    callsign: str
    phraseology: str
    confidence: float


class CPDLCMessage(BaseModel):
    element: str
    text: str


# Models are loaded the first time they are needed, then reused.
whisper_model = None
tts_pipeline = None


def check_ollama():
    """Raises an error if Ollama is not running or the model in config.py is not installed."""
    installed = [model.model for model in ollama.list().models]
    if config.LLM_MODEL not in installed:
        raise RuntimeError(
            config.LLM_MODEL + " is not installed in Ollama. Run: ollama pull " + config.LLM_MODEL
        )


def get_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return whisper_model


def transcribe(audio_path):
    """Returns the text and a confidence between 0 and 1."""
    model = get_whisper()
    segments, info = model.transcribe(audio_path, language="en", vad_filter=True)

    texts = []
    confidences = []
    for segment in segments:
        texts.append(segment.text.strip())
        confidences.append(math.exp(segment.avg_logprob))

    if len(texts) == 0:
        return "", 0.0

    text = " ".join(texts)
    confidence = sum(confidences) / len(confidences)
    return text, confidence


def ask_llm(prompt, schema_class):
    """Asks the LLM and forces the answer to match the schema."""
    reply = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=schema_class.model_json_schema(),
        options={"temperature": 0},
    )
    return schema_class.model_validate_json(reply.message.content)


def extract_intent(transcript):
    prompt = (
        "You are reading a pilot's radio call to an airport tower.\n"
        "Transcript: " + transcript + "\n\n"
        "Fill in these fields:\n"
        "- urgency: red for a distress call (mayday, pan-pan, emergency). "
        "yellow for a time-sensitive request (landing, takeoff, runway, altitude change). "
        "green for anything routine (radio check, information, taxi).\n"
        "- intent: a short phrase for what the pilot wants.\n"
        "- spoken_callsign: the callsign written compactly, like N123AB or DAL482. "
        "Use an empty string if no callsign was said.\n"
        "- confidence: a number from 0 to 1 for how sure you are that you understood the call."
    )
    return ask_llm(prompt, IntentResult)


def write_response(callsign, intent, aircraft):
    if aircraft is None:
        aircraft_text = "No tracker data."
    else:
        last_clearance = aircraft["last_clearance"] or "none"
        aircraft_text = (
            "Altitude " + str(aircraft["altitude_ft"]) + " ft. "
            "Last instruction given: " + last_clearance + "."
        )

    prompt = (
        "You are an air traffic controller at a tower.\n"
        "Aircraft: " + callsign + "\n"
        "Aircraft info: " + aircraft_text + "\n"
        "Pilot request urgency: " + intent.urgency + "\n"
        "Pilot request: " + intent.intent + "\n\n"
        "Fill in these fields:\n"
        "- callsign: the aircraft callsign.\n"
        "- phraseology: the reply in standard ATC phraseology. "
        "Start with the callsign and keep it short.\n"
        "- confidence: a number from 0 to 1 for how sure you are that this reply is "
        "correct and safe to send."
    )
    return ask_llm(prompt, ATCResponse)


def write_cpdlc(response):
    prompt = (
        "Turn this air traffic control radio reply into a CPDLC text message.\n"
        "Reply: " + response.phraseology + "\n\n"
        "Fill in these fields:\n"
        "- element: the name of the standard CPDLC message element that fits, "
        "for example CLIMB TO [altitude] or CONTACT [unit name] [frequency].\n"
        "- text: the full message in capital letters."
    )
    return ask_llm(prompt, CPDLCMessage)


def speak(text, name):
    """Makes a wav file of the spoken reply and returns its path. Returns '' if voice is off."""
    if not config.SPEAK_REPLIES:
        return ""

    global tts_pipeline
    if tts_pipeline is None:
        from kokoro import KPipeline

        tts_pipeline = KPipeline(lang_code="a")

    chunks = []
    for graphemes, phonemes, audio in tts_pipeline(text, voice=config.TTS_VOICE):
        chunks.append(numpy.asarray(audio))
    all_audio = numpy.concatenate(chunks)

    import soundfile

    os.makedirs(config.AUDIO_OUTPUT_FOLDER, exist_ok=True)
    file_path = os.path.join(config.AUDIO_OUTPUT_FOLDER, name + "_" + str(int(time.time())) + ".wav")
    soundfile.write(file_path, all_audio, 24000) # returns audio file.
    return file_path
