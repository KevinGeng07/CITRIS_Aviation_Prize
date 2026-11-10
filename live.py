# The live tower: waits for pilot calls and answers each one as soon as it arrives.
# Run it with:  python live.py   (you need Ollama running with the model from config.py)
#
# There is no radio yet, so you play the pilot by typing in the terminal:
#     unit-001: Tower, N123AB, radio check.
#     unit-002: recordings/some_call.wav        (a .wav path is run through Whisper)
# Press Ctrl+D (or Ctrl+C) to stop.
#
# Two things run side by side:
#   - the listener puts each pilot call into the `incoming` queue
#   - answer_calls() waits for that queue, then runs the pipeline on the call
# While the pipeline is busy, new calls wait in `incoming`, and are answered yellow before green.

import asyncio
import sys
import threading

import database
import pipeline
from pipeline import IntentQueue, Transmission


def parse_line(line):
    """Turns "unit-001: Tower, N123AB, radio check." into a Transmission. Returns None if it can't."""
    if ":" not in line:
        return None

    unit_id, message = line.split(":", 1)
    unit_id = unit_id.strip()
    message = message.strip()
    if unit_id == "" or message == "":
        return None

    if message.endswith(".wav"):
        return Transmission(unit_id=unit_id, audio_path=message)
    return Transmission(unit_id=unit_id, text=message)


def read_terminal(loop, incoming):
    """The stand-in for the radio. It runs in its own thread, because reading
    the terminal blocks while it waits for you to type. A real radio receiver
    would hand over calls the same way: loop.call_soon_threadsafe(incoming.put_nowait, ...)."""
    for line in sys.stdin:
        transmission = parse_line(line)
        if transmission is None:
            print('Type a call like:  unit-001: Tower, N123AB, radio check.')
            continue
        loop.call_soon_threadsafe(incoming.put_nowait, transmission)

    # The terminal was closed (Ctrl+D). None tells answer_calls() to stop.
    loop.call_soon_threadsafe(incoming.put_nowait, None)


async def answer_waiting_calls(incoming, queue):
    """Waits for at least one call, then answers every call that is waiting.
    Returns False when it is time to stop."""
    # Wait here, using no CPU, until a pilot calls.
    first = await incoming.get()
    waiting = [first]

    # If more calls came in while we were busy, take them too, so yellow can go before green.
    while not incoming.empty():
        waiting.append(incoming.get_nowait())

    keep_going = None not in waiting
    calls = [call for call in waiting if call is not None]

    # Whisper and the LLM are slow, so they run in a background thread.
    # That way the listener can keep taking new calls in the meantime.
    for transmission in calls:
        await asyncio.to_thread(pipeline.stage_one, transmission, queue)
    while not queue.is_empty():
        await asyncio.to_thread(pipeline.stage_two, queue.pop())

    return keep_going


async def answer_calls(incoming):
    queue = IntentQueue()
    keep_going = True
    while keep_going:
        keep_going = await answer_waiting_calls(incoming, queue)


async def main():
    database.create_tables()

    # Pretend ADS-B says these two aircraft are in range (same as run_demo.py).
    database.update_position("N123AB", 4500)
    database.update_position("DAL482", 11000)

    incoming = asyncio.Queue()
    loop = asyncio.get_running_loop()
    threading.Thread(target=read_terminal, args=(loop, incoming), daemon=True).start()

    print("Tower is listening. Type a call like:  unit-001: Tower, N123AB, radio check.")
    await answer_calls(incoming)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    print("Tower stopped.")
