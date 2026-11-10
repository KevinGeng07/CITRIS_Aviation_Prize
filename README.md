# ATC pipeline starter

A small, all-open-source version of the pilot-to-tower pipeline. A pilot's transmission is turned into text, the intent is extracted, requests are ordered by urgency, a reply is drafted and checked for confidence, and the reply goes out as both voice and a CPDLC-style text message. A human operator is called in for exactly three reasons.

## The three reasons a human is called

| Reason in the code | What happened |
|---|---|
| `garbled`/`unclear_intent` | The transmission could not be understood (low Whisper confidence, or the LLM was unsure of the intent) <- can be tuned for the model to re-request pilot communication w/ 1 retry. |
| `distress` | The LLM marked the call as urgency "red" (mayday, pan-pan) |
| `low_confidence` | The drafted reply scored below the set confidence threshold |

Escalated cases are saved in the `escalations` table and never written to the aircraft tracker.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | The whole flow in order: understand the call, queue it (yellow before green), reply, send. Also holds the queue and the arbiter (sends only when the channel is free). **Start reading here.** |
| `models.py` | Whisper, the Ollama LLM, and Kokoro, plus the structured outputs the LLMs must use. |
| `database.py` | Everything that is saved: the event log, the callsign registry, the aircraft tracker, and the human operator's inbox. |
| `config.py` | Model names, thresholds, folders. |
| `live.py` | The live tower: waits for pilot calls and answers each one as it arrives. |
| `jobs.py` | The daily jobs: replay saved recordings, and write the QC report of escalated cases. |
| `dags/atc_replay_dag.py` | Airflow schedule for the daily jobs: check Ollama, replay recordings, then export that day's QC report. |
| `run_demo.py` | Try it with a few typed calls. |
| `tests/` | Tests that use fake models. No GPU needed. |

## Quick start (typed text, no audio)

```
pip install -r requirements.txt
ollama pull gemma4:12b
python run_demo.py
```

To play the pilot yourself, run `python live.py` and type calls like `unit-001: Tower, N123AB, radio check.` (or `unit-002: path/to/call.wav`). It waits for each call and answers it right away. Press Ctrl+D to stop.

Run `ollama list` to check the model name, and change `LLM_MODEL` in `config.py` if you want a different one. Run the tests with `pytest`.

## Turning on voice

1. Install the espeak-ng system package (Kokoro needs it), for example `sudo apt install espeak-ng`.
2. Set `SPEAK_REPLIES = True` in `config.py`.
3. Give a transmission an audio file instead of text: `Transmission(unit_id="unit-001", audio_path="call.wav")`.
4. No GPU? Set `WHISPER_DEVICE = "cpu"` and `WHISPER_COMPUTE_TYPE = "int8"`.

## Airflow

Put the project folder somewhere Airflow can see it and point your dags folder at `dags/`. Name recordings like `unit-001_morning_call.wav` (the part before the first `_` is the unit ID) and drop them in `recordings/`. Each recording is replayed once and then moved to `recordings/processed/`. The QC report for a day only lists cases escalated on that day.

Try the job once without the scheduler: `airflow dags test atc_replay_and_qc`.

Live calls do not run through Airflow. It is built for scheduled jobs and would add delay to every radio call. Live calls go through `live.py`, which runs all the time and waits for calls.

## What is not built yet

- Real VHF audio input (`read_terminal()` in `live.py` is the stand-in to replace), real CPDLC sending, and real ADS-B input. The arbiter only prints and logs, and `database.update_position()` is where a feed would plug in.
- Any check beyond the confidence thresholds. There is no rules check, no readback check, and no CPDLC message validator, by design for this first stage.
- Callsign matching beyond simple cleanup. A spoken "Delta four eight two" has to become `DAL482` somewhere. Right now the LLM is asked to do it.
- A health check for live calls. If Ollama or the GPU goes down, the code raises an error and no call reaches the human. (The Airflow job does check Ollama before it starts.)

## Things to know

- The confidence numbers are the biggest safety lever in this design. The two LLM scores are self-reported, which models are often wrong about. Treat the thresholds in `config.py` as starting guesses and tune them with the logged data in the `events` table.
- The CPDLC step asks the LLM for the closest standard message. For real use, put the actual CPDLC message set in the prompt.
- Test on real ATC recordings before trusting any of it, and run in shadow mode (a human controller does all real transmitting) before connecting it to anything live.
