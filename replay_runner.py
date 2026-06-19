"""
run_phase_schedule_sim.py

Runs a SUMO simulation via TraCI, applying a schedule of traffic-light
phase overrides at specific simulation steps, and recording metrics
with the existing MetricsRecorder.

The schedule comes from a JSON file shaped like:

{
    "10": {
        "intersection_id": "intersection_1_1",
        "phase": "rrrryyrrrrrryyrr",
        "phase_name": "ETWT_GREEN",
        "phase_from_sumo": "rrrryyrrrrrryyrr"
    },
    ...
    "original_run_details": {
        "test_name": "metrics test2",
        "simulation_steps": 3600,
        "simulation_config": "dataset/sumo_version/hangzhou_1x1_bc-tyc_18041608_1h/roadnet.sumocfg",
        "llm_path": "..."   # ignored
    }
}

Numeric keys are simulation steps. At each such step, the named
intersection's traffic light state is force-set to `phase_from_sumo`
via traci.trafficlight.setRedYellowGreenState.

Both the SUMO config path and the total step count are read directly
from original_run_details -- no path/step overrides needed.

Usage:
    python run_phase_schedule_sim.py schedule.json
    python run_phase_schedule_sim.py schedule.json --no-gui
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import traci

from utils.metrics_recorder import MetricsRecorder

# metrics_recorder.py imports configurations.MIN_SPEED directly, so
# configurations.py just needs to be importable (e.g. run this script
# from the repo root, or have it on PYTHONPATH).


def load_schedule(schedule_path: Path):
    """
    Splits the JSON into:
      - phase_schedule: {int_step: {...phase info...}}
      - run_details: the 'original_run_details' block
    """
    with open(schedule_path, "r") as f:
        raw = json.load(f)

    if "original_run_details" not in raw:
        raise ValueError(
            "Schedule JSON is missing 'original_run_details' "
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
        phase_schedule[step] = value

    return phase_schedule, run_details


def build_sumo_cmd(sumocfg_path: str, use_gui: bool):
    binary = "sumo-gui" if use_gui else "sumo"
    return [binary, "-c", sumocfg_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_record", type=str, help="Path to the replay record JSON file")
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
    sumo_cmd = build_sumo_cmd(sumocfg_path, use_gui)

    # Auto-start the GUI so you don't have to click "play" manually.
    if use_gui:
        sumo_cmd.append("--start")

    test_name = run_details.get("test_name", "phase_schedule_run")
    safe_name = test_name.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       
    run_dir =  Path(schedule_path).parent / "reruns" / f"{safe_name}_{timestamp}"

    print(f"Run name      : {run_details.get('test_name', '(unnamed)')}")
    print(f"SUMO config   : {sumocfg_path}")
    print(f"Total steps   : {total_steps}")
    print(f"GUI           : {use_gui}")
    print(f"Output dir    : {run_dir}")
    print(f"Phase events  : {sorted(phase_schedule.keys())}")
    print(f"SUMO command  : {' '.join(sumo_cmd)}")
    print("-" * 50)

    recorder = MetricsRecorder(run_dir=run_dir, verbose=True)

    traci.start(sumo_cmd)

    try:
        for step in range(total_steps):
            traci.simulationStep()

            # Apply any scheduled phase override for this exact step.
            event = phase_schedule.get(step)
            if event is not None:
                intersection_id = event["intersection_id"]
                phase_state = event["phase_from_sumo"]
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
        recorder.save_final_summary()
        traci.close()

    


if __name__ == "__main__":
    main()