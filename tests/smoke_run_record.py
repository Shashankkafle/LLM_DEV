"""End-to-end run-record check: a real baseline run must leave a complete,
self-describing record on disk.

Runs FixedTime in-process on the toy net with a blockage crossing the horizon,
then asserts every record file exists with the fields the tooling depends on:
run_manifest.json, per-intersection decisions.jsonl, blockage_events.jsonl,
and final_summary.json (schema v2).

Needs SUMO on PATH. Run: PYTHONPATH=<repo root> python tests/smoke_run_record.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import runner_baselines
from configurations import LOGS_DIR_NAME, RUN_RECORD_SCHEMA_VERSION

SUMOCFG = "simulations/single_intersection/run.sumocfg"
SCENARIO = "simulations/single_intersection/scenarios/accident_single_lane.json"
TEST_NAME = "smoke_run_record"
STEPS = 400  # crosses the blockage's start_step=300

failures = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    runner_baselines.main(argparse.Namespace(
        controller="fixedtime", test_name=TEST_NAME, simulation_steps=STEPS,
        simulation_config=SUMOCFG, intersection_config="single_intersection",
        seed=7, blockage_scenario=SCENARIO, use_gui=False))

    run_dir = sorted(Path(LOGS_DIR_NAME).glob(f"{TEST_NAME}_*"))[-1]
    print(f"\nInspecting {run_dir}\n")

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    check(manifest["schema_version"] == RUN_RECORD_SCHEMA_VERSION,
          "manifest schema_version")
    check(manifest["controller"] == "fixedtime", "manifest controller")
    check(manifest["status"] == "completed", "manifest status completed")
    check(manifest["wall_clock_duration_s"] is not None, "manifest duration stamped")
    check(bool(manifest["sumo"]["version"]), "manifest records the SUMO version")
    check("--seed" in manifest["sumo"]["cmd"], "manifest SUMO cmd shows the seed")
    check(manifest["environment"]["seed"] == 7, "manifest environment seed")
    check(bool(manifest["environment"]["input_files"]),
          "manifest fingerprints the input files")
    check(manifest["blockage"]["scenario_name"] == "accident_single_lane",
          "manifest blockage block")

    decisions = read_jsonl(run_dir / "TLS" / "decisions.jsonl")
    check(len(decisions) > 0, f"baseline wrote decisions.jsonl ({len(decisions)} records)")
    first = decisions[0]
    check(first["controller"] == "fixedtime" and first["intersection_id"] == "TLS",
          "decision records carry controller + intersection_id")
    check("cycle_index" in first["controller_state"],
          "decision records keep the controller's working state")
    check(first["phase_action"]["activated_phase"] in
          ("ETWT", "NTST", "ELWL", "NLSL"),
          "decision records a valid activated phase")
    check(first["traffic_state"] is not None,
          "decision records the observed traffic state")

    events = read_jsonl(run_dir / "blockage_events.jsonl")
    kinds = [e["event"] for e in events]
    check("activated" in kinds, f"blockage activation logged ({kinds})")
    check(any(e["event"] == "obstacle_placed" and "lane_position_m" in e
              for e in events),
          "obstacle placement logged with its actual position")

    summary = json.loads((run_dir / "final_summary.json").read_text())
    check(summary["schema_version"] == RUN_RECORD_SCHEMA_VERSION,
          "final summary schema_version")
    check(summary["controller"] == "fixedtime" and summary["seed"] == 7,
          "final summary carries controller + seed")
    check(summary["blockage_scenario"] == "accident_single_lane",
          "final summary names the blockage scenario")
    check(summary["step_length_s"] == 1.0, "final summary records the step length")
    check(summary["run_wall_clock_s"] is not None, "final summary wall clock")
    check(summary["total_decisions"] == len(decisions),
          "summary decision count matches the decision log")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
