"""Run a real SUMO simulation with the rule-table selector in the loop and
print what goes into it and what comes out.

The cyclic controller serves every intersection; the selector (utils/selector.py)
sees every closed green and is evaluated once per step per intersection, and any
plan it returns is installed. Only the --tls intersection's closed greens are
printed; selector decisions are printed for every intersection.

    uv run python tools/watch_selector.py
    uv run python tools/watch_selector.py --until 5000 --gui

All selector rows are also written to <run dir>/selector_rows.csv.
"""
import argparse
import csv
import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from experiments import CONFIGS  # noqa: E402
from runner_baselines import CyclicController, default_cyclic_plan  # noqa: E402
from runner_common import run_control_loop, setup_run  # noqa: E402
from utils.movement_flags import FlagConfig  # noqa: E402
from utils.regime import NORMAL  # noqa: E402
from utils.selector import (ROW_COLUMNS, ExtentView, GreenRecord,  # noqa: E402
                            IntersectionLanes, Plan, RuleTableSelector, lane_branches)

INTERSECTION_CONFIG = "three_lane"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="hzreal_cross_2_3_r1200", choices=sorted(CONFIGS))
    parser.add_argument("--until", type=int, default=6000, help="Simulated seconds to run.")
    parser.add_argument("--tls", default="intersection_2_3",
                        help="Intersection whose closed greens are printed.")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


# --- building the selector's inputs from the run --------------------------------

def phase_lanes_of(tls, trackers, conf):
    """{phase: lanes} for one signal: a lane is in a phase when one of its links
    is 'G' in that phase's green string."""
    lanes = {phase: [] for phase in conf["phases"]}
    for tracker in trackers:
        if tracker.tls != tls:
            continue
        for phase, phase_conf in conf["phases"].items():
            if any(phase_conf["green"][index] == "G" for index in tracker.link_indices):
                lanes[phase].append(tracker.lane)
    return lanes


def selector_plan(cyclic_plan):
    return Plan(order=tuple(cyclic_plan.order), splits=dict(cyclic_plan.splits),
                offset=cyclic_plan.offset)


# --- printing ---------------------------------------------------------------------

def print_green(record):
    print(f"  green  {record.lane:<28} {record.green_start:>5}-{record.green_end:<5} "
          f"tosi={record.tosi}  held_over={record.held_over}  entries={record.entries}")


def print_decision(t, tls, own_lanes, extents, branches, new_plan, rows):
    print(f"\n[t={t}] {tls}")
    own_flags = {lane: branch for lane, branch in branches.items() if lane in own_lanes}
    print(f"  in : flags on this intersection={own_flags}")
    for extent in extents:
        print(f"       extent {extent.extent_id} regime={extent.regime} kind={extent.kind} "
              f"members={sorted(extent.members)}")
    for row in rows:
        print(f"  out: {row['outcome']}  phase={row['phase']}  dg_requested={row['dg_requested']} "
              f"dg_given={row['dg_given']}  short_by={row['short_by']}  donors={row['donors']}  "
              f"held {row['held_first']}->{row['held_last']}")
        print(f"       plan {row['plan_before']}  ->  {row['plan_after']}")
    print(f"  new plan returned: {new_plan.text if new_plan else 'None'}")


# --- the listener that feeds closed greens to the selector --------------------------

class SelectorFeed:
    """Sits in the OversaturationIndices listener slot in front of the
    Diagnoser: forwards every callback, and hands each closed green to the
    selector."""

    def __init__(self, diagnoser, selector, print_tls):
        self.diagnoser = diagnoser
        self.selector = selector
        self.print_tls = print_tls

    def on_green_start(self, tracker):
        self.diagnoser.on_green_start(tracker)

    def on_step(self, tracker, step):
        self.diagnoser.on_step(tracker, step)

    def on_interval_closed(self, tracker, result):
        self.diagnoser.on_interval_closed(tracker, result)
        record = GreenRecord.from_interval_result(result)
        if record.tls == self.print_tls:
            print_green(record)
        self.selector.on_green_closed(record)


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    conf = INTERSECTION_CONFIGS[INTERSECTION_CONFIG]
    simulation_config = CONFIGS[args.config]
    test_name = f"watch_selector_{args.config}"
    run_meta = {"test_name": test_name, "controller": "cyclic+selector",
                "simulation_steps": args.until, "simulation_config": simulation_config,
                "intersection_config": INTERSECTION_CONFIG, "seed": 1,
                "blockage_scenario": None}
    ctx = setup_run(conf, test_name, simulation_config, run_meta, use_gui=args.gui,
                    seed=1, run_group="watch_selector", movement_flag_config=FlagConfig())
    print(f"Run dir: {ctx.records_dir}")

    trackers = ctx.oversaturation.trackers
    intersections = [IntersectionLanes(tls=tls, phase_lanes=phase_lanes_of(tls, trackers, conf))
                     for tls in ctx.handlers]
    selector = RuleTableSelector(intersections)
    controller = CyclicController(conf, default_cyclic_plan(conf))
    cyclic_plans = {tls: controller.default_plan for tls in ctx.handlers}

    ctx.oversaturation.listener = SelectorFeed(ctx.diagnoser, selector, args.tls)

    live_extent_rows = []

    def remember_extents(cycle_start, network_row, extent_rows):
        live_extent_rows[:] = [row for row in extent_rows if row.regime != NORMAL]

    ctx.diagnoser.cycle_listener = remember_extents

    def evaluate_selector():
        t = ctx.env.get_current_step()
        extents = [ExtentView.from_extent_row(row, ctx.diagnoser.lane_tls)
                   for row in live_extent_rows]
        branches = lane_branches(ctx.movement_flags.active_branches())
        for tls in ctx.handlers:
            plan = selector_plan(cyclic_plans[tls])
            new_plan, rows = selector.evaluate(t, tls, plan, extents, branches)
            if rows or new_plan:
                own_lanes = selector.states[tls].lanes.all_lanes
                print_decision(t, tls, own_lanes, extents, branches, new_plan, rows)
            if new_plan is not None:
                cyclic_plans[tls] = dataclasses.replace(cyclic_plans[tls],
                                                        splits=dict(new_plan.splits))
                controller.set_plan(tls, cyclic_plans[tls], t, ctx.executor_log)

    update_observers = ctx.update_observers

    def update_observers_then_selector():
        update_observers()
        evaluate_selector()

    ctx.update_observers = update_observers_then_selector

    def decide(step, intersection_id, handler):
        decision, _state = controller.choose(intersection_id, handler)
        return decision

    try:
        run_control_loop(ctx, args.until, decide)
    finally:
        with open(ctx.records_dir / "selector_rows.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=ROW_COLUMNS)
            writer.writeheader()
            writer.writerows(selector.rows)
        print(f"\n{len(selector.rows)} selector rows -> {ctx.records_dir / 'selector_rows.csv'}")


if __name__ == "__main__":
    main()
