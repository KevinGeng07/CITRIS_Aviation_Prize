# Try the pipeline with typed text. You need Ollama running with the model from config.py.
# Run it with:  python run_demo.py

import database
import pipeline
from pipeline import Transmission

database.create_tables()

# Pretend ADS-B says these two aircraft are in range.
database.update_position("N123AB", 4500)
database.update_position("DAL482", 11000)

transmissions = [
    Transmission(unit_id="unit-001", text="Tower, N123AB, radio check."),
    Transmission(unit_id="unit-002", text="Tower, DAL482, ten miles east, requesting landing clearance."),
    Transmission(unit_id="unit-003", text="Mayday mayday mayday, N999ZZ, engine failure."),
    Transmission(unit_id="unit-004", text="Tower, uh, krrr, requesting, krrrr"),
]

# The landing request (yellow) should be answered before the radio check (green).
pipeline.process_transmissions(transmissions)

print("")
print("Cases waiting for the human operator:")
for case in database.get_open_cases():
    print("  #" + str(case["id"]) + " " + case["reason"] + ": " + case["transcript"])
