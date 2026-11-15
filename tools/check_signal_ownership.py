"""Check that the controller owned every signal of a run from step 0.

    python tools/check_signal_ownership.py <run_dir>

Reads only the run's own files, so it works on a run still in progress.
Fails (exit 1) if
  - an intersection has no phase_sequences file, or its first record is not
    the step-0 record (prev_phase null, new_phase a green state);
  - any logged state is not one of the config's greens, yellows or all-red
    (the net file's program shows up as right-turn 'G' and 's' states);
  - oversaturation_indices.csv has a green on a right-turn lane (index 0), or
    a green window in which lanes _1 and _2 of one edge are green together.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_initial_record(event, green_states):
    return (event["step"] == 0 and event["prev_phase"] is None
            and event["new_phase"] in green_states)


def phase_sequence_problems(run_dir, manifest):
    phases = manifest["environment"]["phases"]
    green_states = {phase["green"] for phase in phases.values()}
    known_states = green_states | {phase["yellow"] for phase in phases.values()}
    known_states.add(manifest["environment"]["signal_timing"]["all_red_state"])

    problems = defaultdict(int)
    records = 0
    for tls in manifest["sumo"]["intersection_ids"]:
        path = run_dir / "phase_sequences" / f"{tls}_phase_sequence.jsonl"
        if not path.exists():
            problems["intersections without a phase_sequences file"] += 1
            continue
        events = load_events(path)
        records += len(events)
        if not is_initial_record(events[0], green_states):
            problems["files whose first record is not the step-0 green"] += 1
        for event in events[1:]:
            if event["prev_phase"] not in known_states:
                problems["records with an unknown prev_phase"] += 1
        for event in events:
            if event["new_phase"] not in known_states:
                problems["records with an unknown new_phase"] += 1
    return records, problems


def green_window_problems(run_dir):
    with open(run_dir / "oversaturation_indices.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    lanes_green_together = defaultdict(set)
    right_turn_rows = 0
    for row in rows:
        edge, lane_index = row["lane"].rsplit("_", 1)
        if lane_index == "0":
            right_turn_rows += 1
        lanes_green_together[(row["tls"], edge, row["t_green_start"])].add(lane_index)
    both_green = sum(1 for lanes in lanes_green_together.values() if {"1", "2"} <= lanes)
    problems = {}
    if right_turn_rows:
        problems["green rows on right-turn lanes (_0)"] = right_turn_rows
    if both_green:
        problems["green windows with lanes _1 and _2 together"] = both_green
    return len(rows), problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    run_dir = Path(parser.parse_args().run_dir)

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    records, problems = phase_sequence_problems(run_dir, manifest)
    green_rows, green_problems = green_window_problems(run_dir)
    problems = {**problems, **green_problems}
    if not manifest.get("signals_owned_from_start"):
        problems["manifest signals_owned_from_start is not true"] = 1

    print(f"signal ownership: {len(manifest['sumo']['intersection_ids'])} intersections, "
          f"{records} phase records, {green_rows} green rows")
    for name, count in problems.items():
        print(f"  {name}: {count}")
    if problems:
        sys.exit("FAIL: the controller did not own every signal")
    print("PASS: the controller owned every signal from step 0")


if __name__ == "__main__":
    main()
