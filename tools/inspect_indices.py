"""Watch the TOSI/SOSI oversaturation indices being computed, live in sumo-gui.

Runs FixedTime on hzreal in sumo-gui and streams the index computation for
one intersection to the terminal, so the numbers can be checked against what
the GUI shows. The arithmetic is the production code
(utils/oversaturation_indices.py); this tool only listens and prints.

    python tools/inspect_indices.py --tls 2_3 --lane road_2_4_3_1 --scenario c3 --until 1800 --step-print
    python tools/inspect_indices.py --tls 2_3 --scenario none --until 600 --pause-on-green

Per step (--step-print), one line per watched movement with the raw rule
inputs: the lane's signal, vehicle and halted counts, (a) near_stop, (b) exit
count, (c) target-lane halted within 15 m, and whether the second counted as
spillback-blocked ('-' outside the counted green seconds).

At each lane's all-red, a summary block shows H, R and the TOSI/SOSI/ratio
arithmetic. GUI colours: the lane's H set turns red when its green starts,
and the vehicles still on the lane at all-red (H & R) turn magenta. Only
colours change; speeds and routes are never touched.

Times in the summary are snapshot times. H is the queue at the instant green
came on; a step line's t is the end of the second it describes.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import traci  # noqa: E402

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from experiments import BLOCKAGES, CONFIGS  # noqa: E402
from runner_baselines import FixedTimeController  # noqa: E402
from runner_common import build_blockage_manager, run_control_loop, setup_run  # noqa: E402
from utils.oversaturation_indices import HEADWAY_S, SOSI_SKIP_STEPS, compute_tosi  # noqa: E402

INTERSECTION_CONFIG = "three_lane"
RED = (255, 0, 0, 255)
MAGENTA = (255, 0, 255, 255)
IDS_SHOWN_BY_DEFAULT = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tls", required=True,
                        help="Intersection to watch, e.g. 2_3 (= intersection_2_3).")
    parser.add_argument("--lane", default=None,
                        help="Watch one incoming lane, e.g. road_2_4_3_1. "
                             "Default: every signalized lane of --tls.")
    parser.add_argument("--scenario", choices=sorted(BLOCKAGES), default="none",
                        help="Blockage alias from experiments.BLOCKAGES; 'none' "
                             "checks the indices stay ~0 in free flow.")
    parser.add_argument("--until", type=int, default=1800, help="Simulated seconds to run.")
    parser.add_argument("--step-print", action="store_true",
                        help="Print the raw rule inputs every step.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full H/R id lists instead of the first 5.")
    parser.add_argument("--delay", type=int, default=100, help="sumo-gui delay in ms per step.")
    parser.add_argument("--zoom", type=float, default=400.0, help="sumo-gui zoom (percent).")
    parser.add_argument("--pause-on-green", action="store_true",
                        help="Wait for Enter after every green summary.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logs_dir", default="logs",
                        help="The run (with its oversaturation_indices.csv) is "
                             "written under <logs_dir>/inspect_indices/.")
    return parser.parse_args()


def full_tls_id(tls):
    return tls if tls.startswith("intersection_") else f"intersection_{tls}"


# --- formatting ------------------------------------------------------------------

def format_ids(ids, verbose):
    ordered = sorted(ids)
    if verbose or len(ordered) <= IDS_SHOWN_BY_DEFAULT:
        return "[" + ", ".join(ordered) + "]"
    return "[" + ", ".join(ordered[:IDS_SHOWN_BY_DEFAULT]) + ", ...]"


def format_flag(value):
    return "-" if value is None else str(int(value))


def format_step_line(step, target):
    return (f"t={step.t} lane={step.lane} sig={step.signal} n={step.vehicle_count} "
            f"halted={step.halted_count} near_stop={int(step.halted_near_stop)} "
            f"exited={step.exit_count} tgt={target.target_lane} "
            f"tgt_halted_up15={int(target.target_halted_near_entry)} "
            f"-> blocked_step={format_flag(target.blocked_step)}")


def format_tosi_line(result):
    raw_tosi, _ = compute_tosi(result.halted_at_start, result.on_lane_at_allred, result.green_s)
    arithmetic = f"{result.held_over} × {HEADWAY_S} / {result.green_s} = {raw_tosi:.2f}"
    if result.tosi is None:
        return f"TOSI = suppressed (lane blocked during this green; raw {arithmetic})"
    return f"TOSI = {arithmetic}"


def format_sosi_line(result, target):
    counted = result.green_s - SOSI_SKIP_STEPS
    if result.sosi[target] is None:
        return f"SOSI -> {target}: None (green ≤ {SOSI_SKIP_STEPS} s)"
    return (f"SOSI -> {target}: blocked steps = {result.blocked_steps[target]} / "
            f"{counted} = {result.sosi[target]:.2f}")


def format_ratio(halted, capacity, ratio):
    if ratio is None:
        return f"{halted}/{capacity} = None"
    return f"{halted}/{capacity} = {ratio:.2f}"


def format_summary(result, verbose):
    h, r = result.halted_at_start, result.on_lane_at_allred
    lines = [
        f"=== {result.lane} green {result.green_start}–{result.green_end} "
        f"({result.green_s} s), all-red at {result.allred} ===",
        f"H (halted at green start, {result.green_start}): {len(h)} ids {format_ids(h, verbose)}",
        f"R (on lane at all-red, {result.allred}):   {len(r)} ids {format_ids(r, verbose)}",
        f"H∩R = {result.held_over}  discharged = {result.discharged}",
        format_tosi_line(result),
        *(format_sosi_line(result, target) for target in result.sosi),
        f"ratio_start = {format_ratio(result.halted_count_start, result.capacity, result.ratio_start)}"
        f"   ratio_end = {format_ratio(result.halted_count_end, result.capacity, result.ratio_end)}",
    ]
    return "\n".join(lines)


# --- GUI -------------------------------------------------------------------------

def color_vehicles(vehicle_ids, color):
    for vehicle_id in vehicle_ids:
        try:
            traci.vehicle.setColor(vehicle_id, color)
        except traci.TraCIException:
            pass  # left the network between the snapshot and this call


def centre_view(tls_id, zoom):
    x, y = traci.junction.getPosition(tls_id)
    traci.gui.setOffset(traci.gui.DEFAULT_VIEW, x, y)
    traci.gui.setZoom(traci.gui.DEFAULT_VIEW, zoom)


# --- listener ----------------------------------------------------------------------

class IndexInspector:
    """Listener for OversaturationIndices: prints and colours the watched lanes."""

    def __init__(self, tls_id, lane, step_print, verbose, pause_on_green):
        self.tls_id = tls_id
        self.lane = lane
        self.step_print = step_print
        self.verbose = verbose
        self.pause_on_green = pause_on_green

    def watches(self, tracker):
        return tracker.tls == self.tls_id and self.lane in (None, tracker.lane)

    def on_green_start(self, tracker):
        if self.watches(tracker):
            color_vehicles(tracker.open_green.halted_at_start, RED)

    def on_step(self, tracker, step):
        if self.step_print and self.watches(tracker):
            for target in step.targets:
                print(format_step_line(step, target))

    def on_interval_closed(self, tracker, result):
        if not self.watches(tracker):
            return
        color_vehicles(result.halted_at_start & result.on_lane_at_allred, MAGENTA)
        print(format_summary(result, self.verbose))
        if self.pause_on_green:
            input("[paused] Enter to continue... ")


def watched_trackers(trackers, inspector):
    watched = [tracker for tracker in trackers if inspector.watches(tracker)]
    if not watched:
        lanes = sorted(t.lane for t in trackers if t.tls == inspector.tls_id)
        raise SystemExit(f"No signalized lane matches --tls {inspector.tls_id} "
                         f"--lane {inspector.lane}. Lanes of that TLS: {lanes}")
    return watched


def print_watch_list(watched):
    print("Watching:")
    for tracker in watched:
        note = "  (right turn: always green, never gets a summary)" \
            if tracker.lane.endswith("_0") else ""
        print(f"  {tracker.lane} -> {', '.join(tracker.targets)}  "
              f"length {tracker.lane_length:.1f} m, storage {tracker.capacity}{note}")


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tls_id = full_tls_id(args.tls)
    conf = INTERSECTION_CONFIGS[INTERSECTION_CONFIG]
    scenario_path = BLOCKAGES[args.scenario]
    _, blockage_manager = build_blockage_manager(scenario_path)
    inspector = IndexInspector(tls_id, args.lane, args.step_print, args.verbose,
                               args.pause_on_green)
    run_meta = {"test_name": f"inspect_{args.tls}_{args.scenario}", "controller": "fixedtime",
                "simulation_steps": args.until, "simulation_config": CONFIGS["hzreal"],
                "intersection_config": INTERSECTION_CONFIG, "seed": args.seed,
                "blockage_scenario": scenario_path}
    ctx = setup_run(conf, run_meta["test_name"], CONFIGS["hzreal"], run_meta,
                    use_gui=True, seed=args.seed, blockage_manager=blockage_manager,
                    run_group="inspect_indices", logs_dir=args.logs_dir,
                    oversat_listener=inspector,
                    extra_sumo_args=["--start", "--delay", str(args.delay)])
    try:
        print_watch_list(watched_trackers(ctx.oversaturation.trackers, inspector))
        centre_view(tls_id, args.zoom)
    except BaseException:
        ctx.env.close()
        raise
    print(f"Run dir: {ctx.records_dir}")

    controller = FixedTimeController(conf)

    def decide(step, intersection_id, handler):
        phase, _ = controller.choose(intersection_id, handler)
        return phase

    run_control_loop(ctx, args.until, decide)


if __name__ == "__main__":
    main()
