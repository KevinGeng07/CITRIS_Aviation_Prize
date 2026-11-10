# All settings live here. Change these instead of hunting through the code.

import os

# Every file and folder lives inside the project folder, no matter where the code is run from.
# (Airflow runs tasks from a different folder, so plain names like "recordings" would break.)
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

DATABASE_FILE = os.path.join(PROJECT_FOLDER, "atc_data.db")
AUDIO_OUTPUT_FOLDER = os.path.join(PROJECT_FOLDER, "outgoing_audio")
RECORDINGS_FOLDER = os.path.join(PROJECT_FOLDER, "recordings")
PROCESSED_RECORDINGS_FOLDER = os.path.join(PROJECT_FOLDER, "recordings", "processed")
QC_REPORT_FOLDER = os.path.join(PROJECT_FOLDER, "qc_reports")

# Speech to text (faster-whisper has Silero VAD built in)
WHISPER_MODEL = "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper"
WHISPER_DEVICE = "cuda"  # use "cpu" if you have no GPU
WHISPER_COMPUTE_TYPE = "float16"  # use "int8" on cpu

# Language model, run through Ollama. Run "ollama list" to see your installed tags.
LLM_MODEL = "gemma4:12b"

# Text to speech (Kokoro). Keep this False while testing with typed text.
SPEAK_REPLIES = False
TTS_VOICE = "af_heart"

# Confidence thresholds. These are starting guesses: tune them using the logged data.
MIN_TRANSCRIPT_CONFIDENCE = 0.60
MIN_INTENT_CONFIDENCE = 0.60
MIN_RESPONSE_CONFIDENCE = 0.80
