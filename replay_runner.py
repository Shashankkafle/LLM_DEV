"""
run_phase_schedule_sim.py

Runs a SUMO simulation via TraCI, applying a schedule of traffic-light
phase overrides at specific simulation steps, and recording metrics
with the existing MetricsRecorder.

The schedule comes from a JSONL event stream (replay_record.jsonl), one event
per line:

    {"step": 10, "intersection_id": "intersection_1_1", "phase": "rrrryyrr...", "phase_name": "ETWT_GREEN"}
    {"step": 10, "intersection_id": "intersection_1_4", "phase": "GGGrrrrr...", "phase_name": "NTST_GREEN"}
    ...

Run metadata lives in a sidecar `replay_meta.json` next to it:

    {
        "test_name": "metrics test2",
        "simulation_steps": 3600,
        "simulation_config": "dataset/.../roadnet.sumocfg",
        "llm_path": "..."   # ignored
    }

Events are grouped by step on load. At each step, every listed intersection's
traffic light state is force-set to `phase` via
traci.trafficlight.setRedYellowGreenState. The SUMO config path and total step
count are read from the sidecar -- no path/step overrides needed.

Legacy records (a single replay_record.json with step keys and an in-band
`original_run_details` block) are still accepted.

Usage:
    python run_phase_schedule_sim.py path/to/replay_record.jsonl
    python run_phase_schedule_sim.py path/to/replay_record.jsonl --no-gui
    python run_phase_schedule_sim.py path/to/replay_record.json    # legacy
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import traci

from configurations import (
    SUMO_BINARY,
    SUMO_GUI_BINARY,
    RERUNS_DIR_NAME,
    REPLAY_META_FILENAME,
    sumo_metrics_args,
)
from utils.metrics_recorder import MetricsRecorder

# metrics_recorder.py imports configurations.MIN_SPEED directly, so
# configurations.py just needs to be importable (e.g. run this script
# from the repo root, or have it on PYTHONPATH).


def load_schedule(schedule_path: Path):
    """
    Loads a replay record, returning:
      - phase_schedule: {int_step: [event, ...]}
      - run_details: the run metadata (config path, step count, ...)

    A .jsonl path is read as the new streamed format (with a sidecar
    replay_meta.json); anything else is read as the legacy single-JSON format.
    """
    schedule_path = Path(schedule_path)
    if schedule_path.suffix == ".jsonl":
        return _load_jsonl_schedule(schedule_path)
    return _load_legacy_schedule(schedule_path)


def _load_jsonl_schedule(events_path: Path):
    """New format: one event per line, run metadata in a sidecar file."""
    meta_path = events_path.parent / REPLAY_META_FILENAME
    if not meta_path.exists():
        raise ValueError(
            f"Expected metadata sidecar next to {events_path.name}: {meta_path}"
        )
    with open(meta_path, "r") as f:
        run_details = json.load(f)

    phase_schedule = {}
    with open(events_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            phase_schedule.setdefault(event["step"], []).append(event)

    return phase_schedule, run_details


def _load_legacy_schedule(record_path: Path):
    """Old format: a single JSON object keyed by step, metadata in-band."""
    with open(record_path, "r") as f:
        raw = json.load(f)

    if "original_run_details" not in raw:
        raise ValueError(
            "Legacy record is missing 'original_run_details' "
            "(needed for simulation_config and simulation_steps)."
        )
    run_details = raw.pop("original_run_details")

    phase_schedule = {}
    for key, value in raw.items():
        try:
            step = int(key)
        except ValueError:
            print(f"Skipping non-numeric schedule key: {key!r}")
            continue
        # Old records stored either a single event or (after the first fix) a
        # list per step; normalize both to a list.
        phase_schedule[step] = value if isinstance(value, list) else [value]

    return phase_schedule, run_details


def build_sumo_cmd(sumocfg_path: str, use_gui: bool, output_dir=None):
    binary = SUMO_GUI_BINARY if use_gui else SUMO_BINARY
    cmd = [binary, "-c", sumocfg_path]
    # Same flags as the original live run (see SumoEnv): identical simulation
    # behavior (teleport disabled) plus the same metric outputs, so the replay
    # reproduces the run it re-scores and is comparable to every controller.
    if output_dir is not None:
        cmd += sumo_metrics_args(output_dir)
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay_record",
        type=str,
        help="Path to the replay record (replay_record.jsonl, or a legacy .json)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless (sumo) instead of sumo-gui. Default is GUI, since you want to observe it.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory to write metrics output to. Defaults to a timestamped folder under ./runs/",
    )
    args = parser.parse_args()

    schedule_path = Path(args.replay_record)
    if not schedule_path.exists():
        print(f"Schedule file not found: {schedule_path}", file=sys.stderr)
        sys.exit(1)

    phase_schedule, run_details = load_schedule(schedule_path)

    sumocfg_path = run_details["simulation_config"]
    total_steps = run_details["simulation_steps"]

    if not Path(sumocfg_path).exists():
        print(
            f"WARNING: simulation_config path does not exist relative to cwd: {sumocfg_path}\n"
            f"         Run this script from the directory that makes that path resolve correctly.",
            file=sys.stderr,
        )

    use_gui = not args.no_gui

    test_name = run_details.get("test_name", "phase_schedule_run")
    safe_name = test_name.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir =  Path(schedule_path).parent / RERUNS_DIR_NAME / f"{safe_name}_{timestamp}"

    # Build the command after run_dir is known so SUMO writes its statistics
    # into this run's output directory.
    sumo_cmd = build_sumo_cmd(sumocfg_path, use_gui, output_dir=run_dir)

    # Auto-start the GUI so you don't have to click "play" manually.
    # if use_gui:
    #     sumo_cmd.append("--start")

    print(f"Run name      : {run_details.get('test_name', '(unnamed)')}")
    print(f"SUMO config   : {sumocfg_path}")
    print(f"Total steps   : {total_steps}")
    print(f"GUI           : {use_gui}")
    print(f"Output dir    : {run_dir}")
    print(f"Phase events  : {sorted(phase_schedule.keys())}")
    print(f"SUMO command  : {' '.join(sumo_cmd)}")
    print("-" * 50)

    recorder = MetricsRecorder(run_dir=run_dir, verbose=True,
                               sumo_config=sumocfg_path)

    traci.start(sumo_cmd)

    try:
        for step in range(total_steps):
            traci.simulationStep()

            # Apply any scheduled phase overrides for this exact step. A step may
            # hold several events -- one per intersection that switched on it.
            for event in phase_schedule.get(step, []):
                intersection_id = event["intersection_id"]
                phase_state = event["phase"]
                phase_name = event.get("phase_name", "")

                try:
                    traci.trafficlight.setRedYellowGreenState(
                        intersection_id, phase_state
                    )
                    print(
                        f"[step {step}] Applied phase '{phase_name}' "
                        f"({phase_state}) to '{intersection_id}'"
                    )
                except traci.exceptions.TraCIException as exc:
                    print(
                        f"[step {step}] FAILED to set phase on "
                        f"'{intersection_id}': {exc}",
                        file=sys.stderr,
                    )

            # Record per-step metrics every step.
            recorder.record_step_summary(step)

            # Stop early if SUMO has no more vehicles expected.
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected at step {step}, stopping early.")
                break

    finally:
        # Close SUMO first so it flushes the statistics file, then summarize
        # (save_final_summary parses that file for population-faithful metrics).
        traci.close()
        recorder.save_final_summary()

    


if __name__ == "__main__":
    main()