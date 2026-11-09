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
    python runner_baselines.py --controller ruletable --movement_flags   # see ruletable_controller.py
"""
import argparse
from dataclasses import dataclass

import traci

from runner_common import (
    setup_run, run_control_loop, build_blockage_manager, _phase_and_green,
    add_movement_flag_args, movement_flag_config_from_args, regime_cycle_s_from_args,
)
from utils.run_manifest import build_manifest
from configurations import (
    INTERSECTION_CONFIGS,
    DEFAULT_INTERSECTION_CONFIG_NAME,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_SIMULATION_CONFIG,
)


# =============================================================================
# Controllers. Each exposes choose(intersection_id, handler) ->
# (phase name, controller_state dict). controller_state is whatever the rule
# computed to decide (recorded per decision, so any run is explainable).
# A decision is made only when a green window ends (handler.switch_phase), so
# both controllers run on the LLM/CoLight cadence (30s green = one decision).
# =============================================================================

class FixedTimeController:
    """Round-robin: cycle the phases in a fixed order, equal green each.
    Ignores traffic state entirely -- the classic FixedTime baseline."""

    def __init__(self, conf):
        self.cycle = list(conf["phases"].keys())

    def choose(self, intersection_id, handler):
        idx = (self.cycle.index(handler.current_phase) + 1) % len(self.cycle)
        return self.cycle[idx], {"cycle_index": idx}


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
        info = {"pressures": pressures}
        max_pressure = max(pressures.values())
        # Tie-break toward holding the current phase to avoid needless switching.
        if pressures[handler.current_phase] == max_pressure:
            return handler.current_phase, info
        for phase, pressure in pressures.items():
            if pressure == max_pressure:
                return phase, info
        return handler.current_phase, info  # unreachable; keeps the type checker happy


@dataclass
class CyclicPlan:
    """A fixed cycle: phases served in `order`, each for splits[phase] green
    seconds. offset is stored and reported but not yet applied."""
    order: list[str]
    splits: dict[str, int]
    offset: int
    yellow_duration: int
    red_duration: int

    @property
    def cycle_length(self):
        return sum(self.splits[p] + self.yellow_duration + self.red_duration
                   for p in self.order)


def default_cyclic_plan(conf):
    settings = conf["global_settings"]
    order = list(conf["phases"])
    return CyclicPlan(
        order=order,
        splits={phase: settings["default_green_duration"] for phase in order},
        offset=0,
        yellow_duration=settings["yellow_duration"],
        red_duration=settings["red_duration"],
    )


def plan_transition_type(old_plan, new_plan):
    if old_plan.cycle_length != new_plan.cycle_length:
        return "cycle_change"
    if old_plan.splits != new_plan.splits:
        return "split_change"
    return "plan_change"


class CyclicController:
    """Serve each intersection's CyclicPlan in order, with the plan's green
    split per phase. Every intersection starts on the plan given here; a plan
    counts as installed for an intersection once set_plan or choose has used it."""

    def __init__(self, conf, plan):
        self.conf = conf
        self.default_plan = plan
        self.plans = {}

    def set_plan(self, intersection_id, new_plan, t, executor_log):
        """Install new_plan. Only replacing a different, already-installed plan
        is a plan change; the initial install logs nothing."""
        old_plan = self.plans.get(intersection_id)
        self.plans[intersection_id] = new_plan
        if old_plan is None or old_plan == new_plan:
            return
        executor_log.log_plan_change(
            t, intersection_id, old_plan.cycle_length, new_plan.cycle_length,
            old_plan.splits, new_plan.splits, plan_transition_type(old_plan, new_plan))

    def choose(self, intersection_id, handler):
        plan = self.plans.setdefault(intersection_id, self.default_plan)
        # A phase outside the order (e.g. the start phase) restarts the cycle.
        if handler.current_phase in plan.order:
            cycle_index = (plan.order.index(handler.current_phase) + 1) % len(plan.order)
        else:
            cycle_index = 0
        next_phase = plan.order[cycle_index]
        green = plan.splits[next_phase]
        controller_state = {
            "cycle_index": cycle_index,
            "green": green,
            "cycle_length": plan.cycle_length,
            "offset": plan.offset,
        }
        return (next_phase, green), controller_state


def build_controller(name, conf, intersection_ids):
    if name == "fixedtime":
        return FixedTimeController(conf)
    if name == "maxpressure":
        return MaxPressureController(conf, intersection_ids)
    if name == "cyclic":
        return CyclicController(conf, default_cyclic_plan(conf))
    raise ValueError(f"Unknown controller: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller",
                        choices=["fixedtime", "maxpressure", "cyclic", "ruletable"],
                        required=True,
                        help="ruletable: the cyclic plan with the rule-table selector "
                             "changing its splits; needs --movement_flags.")
    parser.add_argument("--test_name", type=str, default=None,
                        help="Output folder name prefix. Defaults to the controller name.")
    parser.add_argument("--run_group", type=str, default=None,
                        help="Nest the run under logs/<run_group>/ (used by "
                             "run_matrix.py to group a sweep's runs).")
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="SUMO random seed. Default keeps SUMO's fixed "
                             "default (deterministic reruns).")
    parser.add_argument("--blockage_scenario", type=str, default=None,
                        help="Path to a blockage scenario JSON (see "
                             "simulations/single_intersection/scenarios/). "
                             "Omit for no blockages.")
    parser.add_argument(
        "--intersection_config",
        type=str,
        choices=list(INTERSECTION_CONFIGS.keys()),
        default=DEFAULT_INTERSECTION_CONFIG_NAME,
    )
    parser.add_argument("--logs_dir", type=str, default=None,
                        help="Write the run directory under this root instead of logs/.")
    parser.add_argument("--oversat_shadow", action="store_true",
                        help="Also write oversaturation_shadow.csv: per-step "
                             "halted and exit counts for every signalized lane "
                             "(the raw inputs behind TOSI/SOSI).")
    add_movement_flag_args(parser)
    args = parser.parse_args()
    if args.controller == "ruletable" and not args.movement_flags:
        parser.error("--controller ruletable needs --movement_flags (the selector "
                     "runs on the live diagnoser)")
    return args


def build_run_controller(name, conf, ctx, flag_config):
    if name == "ruletable":
        # Imported here: ruletable_controller builds on CyclicController from
        # this module, so a top-level import would be circular.
        from ruletable_controller import RuleTableController
        return RuleTableController(conf, ctx, flag_config)
    return build_controller(name, conf, list(ctx.handlers))


def main(args):
    conf = INTERSECTION_CONFIGS[args.intersection_config]
    test_name = args.test_name or args.controller

    _, blockage_manager = build_blockage_manager(args.blockage_scenario)
    run_meta = {
        "test_name": test_name,
        "controller": args.controller,
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "intersection_config": args.intersection_config,
        "seed": args.seed,
        "blockage_scenario": args.blockage_scenario,
    }
    manifest = build_manifest(args.controller, args, args.intersection_config, conf)
    manifest["test_name"] = test_name
    flag_config = movement_flag_config_from_args(args)
    ctx = setup_run(conf, test_name, args.simulation_config, run_meta,
                    use_gui=args.use_gui, seed=args.seed,
                    blockage_manager=blockage_manager, manifest=manifest,
                    run_group=args.run_group, logs_dir=args.logs_dir,
                    oversat_shadow=getattr(args, "oversat_shadow", False),
                    movement_flag_config=flag_config,
                    regime_cycle_s=regime_cycle_s_from_args(args))
    controller = build_run_controller(args.controller, conf, ctx, flag_config)

    print(f"Controller    : {args.controller}")
    print(f"SUMO config   : {args.simulation_config}")
    print(f"Intersections : {len(ctx.handlers)}")
    print(f"Output dir    : {ctx.records_dir}")
    print("-" * 50)

    def decide(step, intersection_id, handler):
        decision, controller_state = controller.choose(intersection_id, handler)
        phase, _green = _phase_and_green(decision)
        state_data = ctx.env.get_state(intersection_id)
        ctx.recorder.record_simple_decision(
            step=step, intersection_id=intersection_id,
            previous_phase=handler.current_phase, activated_phase=phase,
            decision_type="rule_decision",
            controller_state=controller_state,
            traffic_state=state_data.get("movement_states"),
        )
        return decision

    try:
        run_control_loop(ctx, args.simulation_steps, decide)
    finally:
        if hasattr(controller, "close"):
            controller.close()


if __name__ == "__main__":
    main(parse_args())
