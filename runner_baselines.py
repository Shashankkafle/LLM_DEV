"""
runner_baselines.py

Transportation-engineering baselines (FixedTime, MaxPressure) on the SAME SUMO
env / PhaseHandler / MetricsRecorder seam as the LLM (runner.py) and CoLight
(runner_colight.py) runners. Running every controller through this one path is
what makes the comparison fair: identical scenario, transitions, and
population-faithful metrics -- only the decision rule changes.

Usage:
    python runner_baselines.py --controller fixedtime
    python runner_baselines.py --controller maxpressure \
        --simulation_config dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.sumocfg \
        --intersection_config three_lane --simulation_steps 3600 \
        --test_name maxpressure_hz4x4
"""
import argparse
from datetime import datetime
from pathlib import Path

import traci

from sumo_env import SumoEnv
from utils.phase_handler import PhaseHandler
from utils.metrics_recorder import MetricsRecorder
from utils.replay_recorder import ReplayRecorder
from configurations import (
    INTERSECTION_CONFIGS,
    DEFAULT_INTERSECTION_CONFIG_NAME,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_START_PHASE,
    LOGS_DIR_NAME,
    PHASE_SEQUENCES_DIR_NAME,
)


# =============================================================================
# Controllers. Each exposes choose(intersection_id, handler) -> phase name.
# A decision is made only when a green window ends (handler.switch_phase), so
# both controllers run on the LLM/CoLight cadence (30s green = one decision).
# =============================================================================

class FixedTimeController:
    """Round-robin: cycle the phases in a fixed order, equal green each.
    Ignores traffic state entirely -- the classic FixedTime baseline."""

    def __init__(self, conf):
        self.cycle = list(conf["phases"].keys())

    def choose(self, intersection_id, handler):
        idx = self.cycle.index(handler.current_phase)
        return self.cycle[(idx + 1) % len(self.cycle)]


class MaxPressureController:
    """Greedily activate the highest-pressure phase (Varaiya max-pressure).

    A phase's pressure is the sum over its protected-green movements of
    (upstream stopped - downstream stopped) vehicles. Stopped counts come from
    SUMO halting numbers (speed < 0.1 m/s), matching MIN_SPEED. On a tie the
    current phase is kept, so a still-dominant phase extends instead of
    needlessly switching (and paying the yellow+all-red transition)."""

    def __init__(self, conf, intersection_ids):
        self.conf = conf
        # per intersection: {phase_name: {"in": {lanes}, "out": {lanes}}}
        self.phase_links = {
            iid: self._build_phase_links(iid) for iid in intersection_ids
        }

    def _build_phase_links(self, intersection_id):
        """Map each phase to the incoming and outgoing lanes of its protected
        (capital 'G') movements, read from SUMO's controlled-link table."""
        controlled_links = traci.trafficlight.getControlledLinks(intersection_id)
        phase_links = {}
        for phase_name, phase_cfg in self.conf["phases"].items():
            green = phase_cfg["green"]
            in_lanes, out_lanes = set(), set()
            for i, char in enumerate(green):
                if char == "G":  # protected green only, not permissive 'g'
                    for from_lane, to_lane, _via in controlled_links[i]:
                        in_lanes.add(from_lane)
                        out_lanes.add(to_lane)
            phase_links[phase_name] = {"in": in_lanes, "out": out_lanes}
        return phase_links

    def choose(self, intersection_id, handler):
        links = self.phase_links[intersection_id]

        def halting(lanes):
            return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

        pressures = {
            phase: halting(d["in"]) - halting(d["out"])
            for phase, d in links.items()
        }
        max_pressure = max(pressures.values())
        # Tie-break toward holding the current phase to avoid needless switching.
        if pressures[handler.current_phase] == max_pressure:
            return handler.current_phase
        for phase, pressure in pressures.items():
            if pressure == max_pressure:
                return phase
        return handler.current_phase  # unreachable; keeps the type checker happy


def build_controller(name, conf, intersection_ids):
    if name == "fixedtime":
        return FixedTimeController(conf)
    if name == "maxpressure":
        return MaxPressureController(conf, intersection_ids)
    raise ValueError(f"Unknown controller: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["fixedtime", "maxpressure"], required=True)
    parser.add_argument("--test_name", type=str, default=None,
                        help="Output folder name prefix. Defaults to the controller name.")
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="SUMO random seed. Default keeps SUMO's fixed "
                             "default (deterministic reruns).")
    parser.add_argument(
        "--intersection_config",
        type=str,
        choices=list(INTERSECTION_CONFIGS.keys()),
        default=DEFAULT_INTERSECTION_CONFIG_NAME,
    )
    return parser.parse_args()


def main(args):
    conf = INTERSECTION_CONFIGS[args.intersection_config]
    test_name = args.test_name or args.controller
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = Path(LOGS_DIR_NAME) / f"{test_name}_{timestamp}"
    phase_sequence_dir = records_dir / PHASE_SEQUENCES_DIR_NAME
    phase_sequence_dir.mkdir(parents=True, exist_ok=True)

    run_details = {
        "test_name": test_name,
        "controller": args.controller,
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "intersection_config": args.intersection_config,
        "seed": args.seed,
    }
    replay_recorder = ReplayRecorder(record_dir=records_dir, meta=run_details)
    recorder = MetricsRecorder(run_dir=records_dir, verbose=False,
                               phase_names=list(conf["phases"].keys()),
                               sumo_config=args.simulation_config)
    env = SumoEnv(
        sumo_config=args.simulation_config, use_gui=args.use_gui,
        phase_sequence_dir=phase_sequence_dir, intersection_config=conf,
        output_dir=records_dir, seed=args.seed,
    )
    env.start_simulation()

    intersection_ids = env.get_intersections()
    controller = build_controller(args.controller, conf, intersection_ids)
    handlers = {
        iid: PhaseHandler(env=env, conf=conf, intersection_id=iid,
                          start_phase=DEFAULT_START_PHASE,
                          replay_recorder=replay_recorder)
        for iid in intersection_ids
    }

    print(f"Controller    : {args.controller}")
    print(f"SUMO config   : {args.simulation_config}")
    print(f"Intersections : {len(intersection_ids)}")
    print(f"Output dir    : {records_dir}")
    print("-" * 50)

    try:
        for step in range(args.simulation_steps):
            env.step()
            recorder.record_step_summary(step)
            for intersection_id, handler in handlers.items():
                handler.step()
                if handler.switch_phase:
                    next_phase = controller.choose(intersection_id, handler)
                    recorder.record_decision_wait()
                    handler.activate_phase(next_phase)

            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected at step {step}, stopping early.")
                break
    finally:
        # Close SUMO first so it flushes the statistics file, then summarize.
        env.close()
        recorder.save_final_summary()


if __name__ == "__main__":
    main(parse_args())
