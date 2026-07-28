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
    RUN_MANIFEST_FILENAME,
    FINAL_SUMMARY_FILENAME,
    BLOCKAGE_SCENARIO_COPY_FILENAME,
    BLOCKAGE_EVENTS_FILENAME,
    sumo_metrics_args,
)
from utils.blockage_manager import BlockageManager, load_scenario
from utils.metrics_recorder import MetricsRecorder
from utils.run_manifest import (
    build_manifest,
    save_manifest,
    finalize_manifest,
    add_sumo_runtime,
    input_fingerprints,
)

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


def load_blockage_manager(schedule_path: Path, run_details):
    """Rebuild the original run's blockage manager for the rerun.

    Without this, a blockage run would silently replay blockage-free physics
    under the recorded phase schedule, defeating the re-scoring purpose.
    Prefers the scenario copy inside the run dir (reproducible even if the
    original scenario file changed); falls back to the recorded path.
    """
    scenario_copy = schedule_path.parent / BLOCKAGE_SCENARIO_COPY_FILENAME
    if scenario_copy.exists():
        source = scenario_copy
    elif run_details.get("blockage_scenario"):
        source = run_details["blockage_scenario"]
    else:
        return None
    scenario = load_scenario(source)
    print(f"Blockage      : {scenario['scenario_name']} (from {source})")
    return BlockageManager(scenario["blockages"],
                           scenario_name=scenario["scenario_name"])


def load_original_record(original_dir: Path):
    """The original run's recorded input-file hashes and SUMO version, from
    run_manifest.json (new runs) or final_summary.json (older runs).
    Returns (input_files, sumo_version); either may be None."""
    manifest_path = original_dir / RUN_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text())
        return (manifest.get("environment", {}).get("input_files"),
                (manifest.get("sumo") or {}).get("version"))
    except Exception:
        pass
    try:
        summary = json.loads((original_dir / FINAL_SUMMARY_FILENAME).read_text())
        return summary.get("input_files"), summary.get("sumo_version")
    except Exception:
        return None, None


def warn_on_input_mismatch(original_inputs, sumocfg_path):
    """Warn (never fail) when the current net/route files differ from the
    hashes the original run recorded -- a replay on changed inputs silently
    re-scores a different experiment."""
    if not original_inputs:
        return
    current = input_fingerprints(sumocfg_path)
    if not current:
        return
    for name, recorded_sha in original_inputs.items():
        current_sha = current.get(name)
        if current_sha is None:
            print(f"WARNING: input file {name} from the original run is no "
                  f"longer referenced by {sumocfg_path}", file=sys.stderr)
        elif current_sha != recorded_sha:
            print(f"WARNING: input file {name} differs from the original run "
                  f"(hash {current_sha[:12]} vs recorded {recorded_sha[:12]})",
                  file=sys.stderr)


def warn_on_sumo_version_mismatch(original_version):
    """Warn when replaying under a different SUMO than the original run
    (METRICS.md documents measurable cross-version drift)."""
    if not original_version:
        return
    current = traci.getVersion()[1]
    if current != original_version:
        print(f"WARNING: replaying under {current}, original ran "
              f"{original_version} -- results may drift", file=sys.stderr)


def build_sumo_cmd(sumocfg_path: str, use_gui: bool, output_dir=None, seed=None):
    binary = SUMO_GUI_BINARY if use_gui else SUMO_BINARY
    cmd = [binary, "-c", sumocfg_path]
    # Same flags as the original live run (see SumoEnv): identical simulation
    # behavior (teleport disabled) plus the same metric outputs, so the replay
    # reproduces the run it re-scores and is comparable to every controller.
    if output_dir is not None:
        cmd += sumo_metrics_args(output_dir)
    # None keeps SUMO's fixed default seed, matching an unseeded original run.
    if seed is not None:
        cmd += ["--seed", str(seed)]
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
    # into this run's output directory. The original run's seed (if any) is
    # replayed too, so the simulation reproduces the same traffic.
    sumo_cmd = build_sumo_cmd(sumocfg_path, use_gui, output_dir=run_dir,
                              seed=run_details.get("seed"))

    # Auto-start the GUI so you don't have to click "play" manually.
    # if use_gui:
    #     sumo_cmd.append("--start")

    print(f"Run name      : {run_details.get('test_name', '(unnamed)')}")
    print(f"SUMO config   : {sumocfg_path}")
    print(f"Total steps   : {total_steps}")
    print(f"GUI           : {use_gui}")
    print(f"Output dir    : {run_dir}")
    total_events = sum(len(events) for events in phase_schedule.values())
    print(f"Phase events  : {total_events} events across {len(phase_schedule)} steps")
    print(f"SUMO command  : {' '.join(sumo_cmd)}")
    print("-" * 50)

    original_inputs, original_sumo_version = load_original_record(schedule_path.parent)
    warn_on_input_mismatch(original_inputs, sumocfg_path)

    manifest = build_manifest("replay", args, None, None, extra={
        "replayed_from": str(schedule_path.parent),
        "source_run_details": run_details,
    })
    manifest["test_name"] = test_name
    manifest["environment"].update({
        "simulation_config": sumocfg_path,
        "simulation_config_abs": (str(Path(sumocfg_path).resolve())
                                  if Path(sumocfg_path).exists() else None),
        "input_files": input_fingerprints(sumocfg_path),
        "seed": run_details.get("seed"),
        "simulation_steps": total_steps,
    })
    save_manifest(run_dir, manifest)

    blockage_manager = load_blockage_manager(schedule_path, run_details)
    if blockage_manager is not None:
        blockage_manager.set_event_log(run_dir / BLOCKAGE_EVENTS_FILENAME)
    run_info = {
        "test_name": test_name,
        "controller": "replay",
        "replayed_from": str(schedule_path.parent),
        "source_controller": run_details.get("controller"),
        "simulation_config": sumocfg_path,
        "seed": run_details.get("seed"),
        "blockage_scenario": run_details.get("blockage_scenario"),
    }
    recorder = MetricsRecorder(run_dir=run_dir, verbose=True,
                               sumo_config=sumocfg_path,
                               blockage_manager=blockage_manager,
                               run_info=run_info)

    traci.start(sumo_cmd)
    recorder.record_initial_load()
    warn_on_sumo_version_mismatch(original_sumo_version)
    add_sumo_runtime(manifest, sumo_cmd)
    save_manifest(run_dir, manifest)
    if blockage_manager is not None:
        blockage_manager.validate_against_network()

    status = "crashed"
    try:
        for step in range(total_steps):
            traci.simulationStep()
            # Same clock as the live runs: SumoEnv ticks the manager with
            # sim seconds, not the 0-based loop variable.
            if blockage_manager is not None:
                blockage_manager.step(int(traci.simulation.getTime()))

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
        status = "completed"

    finally:
        # Close SUMO first so it flushes the statistics file, then summarize
        # (save_final_summary parses that file for population-faithful metrics).
        try:
            traci.close()
            recorder.save_final_summary()
        finally:
            finalize_manifest(run_dir, status)

    


if __name__ == "__main__":
    main()