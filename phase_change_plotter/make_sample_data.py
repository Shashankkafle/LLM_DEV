"""
Generates a synthetic traffic-signal-phase-log JSON file that mimics the
structure described by the user, for development/testing of the visualizer.

Structure:
{
    "original_run_details": {...},     # metadata to be IGNORED
    "10": {"intersection_id": ..., "phase": "...", "phase_name": "...", ...},
    "13": {...},
    ...
}

Note: in the real dataset, keys are timesteps but multiple intersections can
share / interleave timesteps (each intersection logs its own phase-change
events independently), so a single timestep key may only describe ONE
intersection's event. We replicate that by emitting one record per
(timestep, intersection) pair, but since the top-level structure is a flat
dict keyed by "timestep", we simulate this the way it's likely produced in
practice: incrementing a global counter per logged event.
"""
import json
import random

random.seed(42)

PHASES = [
    ("rrrryyrrrrrryyrr", "ETWT_GREEN"),
    ("rrrrrrrrrrrrrrrr", "ALL_RED"),
    ("yyrrrrrryyrrrrrr", "NTST_GREEN"),
    ("rrrryyrrrrrryyrr", "ETWT_GREEN"),
    ("rrrrrryyrrrrrryy", "NTST_LEFT_GREEN"),
]

INTERSECTIONS = [f"intersection_{i}_{j}" for i in range(1, 4) for j in range(1, 4)]  # 9 intersections

def generate(num_events=20000, out_path="sample_logs.json", grid=(3, 3)):
    rows, cols = grid
    intersections = [f"intersection_{i}_{j}" for i in range(1, rows + 1) for j in range(1, cols + 1)]
    data = {
        "original_run_details": {
            "sumo_config": "hangzhou_1x1.sumocfg",
            "run_id": "exp_001",
            "notes": "synthetic sample data, ignore me",
        }
    }

    # Track current sim-time per intersection so timesteps are monotonic per-intersection
    cur_time = {i: 0 for i in intersections}
    cur_phase_idx = {i: random.randrange(len(PHASES)) for i in intersections}

    events = []
    for _ in range(num_events):
        inter = random.choice(intersections)
        # advance this intersection's clock by a random dwell time
        dwell = random.choice([3, 3, 5, 5, 5, 8, 10, 15, 20])
        cur_time[inter] += dwell

        # occasionally change phase, otherwise repeat (so we get realistic
        # short repeated runs that should be collapsed/merged by the tool,
        # plus some genuinely new contiguous blocks)
        if random.random() < 0.4:
            cur_phase_idx[inter] = random.randrange(len(PHASES))
        phase_str, phase_name = PHASES[cur_phase_idx[inter]]

        events.append((cur_time[inter], inter, phase_str, phase_name))

    # Sort by timestep to mimic a real chronological log, then assign keys.
    # Use the timestep itself as the key where possible; if collisions occur
    # (two intersections change at the exact same tick), bump by a tiny
    # fractional-looking integer offset to keep keys unique, similar to how
    # real loggers might use an event counter instead of raw tick.
    events.sort(key=lambda e: e[0])
    last_key = -1
    for t, inter, phase_str, phase_name in events:
        key = max(t, last_key + 1)
        last_key = key
        record = {
            "intersection_id": inter,
            "phase": phase_str,
            "phase_name": phase_name,
        }
        if random.random() < 0.05:
            record["phase_from_sumo"] = phase_str
        data[str(key)] = record

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(events)} events across {len(intersections)} intersections to {out_path}")


if __name__ == "__main__":
    generate()
