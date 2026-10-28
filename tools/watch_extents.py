"""Watch the extents and their regimes live, with sumo-gui open.

Runs a baseline controller in sumo-gui with the movement flags on and prints,
at the end of every regime cycle (140 s by default), the network regime and
one line per extent. The flags, extents and regimes are the production code
(utils/movement_flags.py, utils/extents.py, utils/diagnoser.py); this tool
only listens and prints. The run dir with movement_flags.csv, regime.csv and
extents.csv is written as usual, so tools/verify_flags.py can check it after.

    python tools/watch_extents.py
    python tools/watch_extents.py --controller fixedtime --until 1500
    python tools/watch_extents.py --pause-on-change --color

Defaults: MaxPressure on the crossing surge hzreal_cross_2_3_r1200 (both
surges meet at intersection_2_3 from 3600 s; the first extent is born at
~3920 s). FixedTime on the same config shows extents from ~700 s.

Extent line: id, regime (with the change, when it changed this cycle), NEW
when the extent was born this cycle, GONE when it no longer has a component
(kept until it reaches N), then lanes / connections, E = member lanes that
entered, X = old member lanes that exited, and the member lanes.

--color paints the vehicles on each extent's lanes by regime at every cycle
close: L yellow, P red, R green. Only colours change.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import traci  # noqa: E402

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from experiments import CONFIGS  # noqa: E402
from runner_baselines import build_controller  # noqa: E402
from runner_common import run_control_loop, setup_run  # noqa: E402
from utils.diagnoser import DEFAULT_CYCLE_S  # noqa: E402
from utils.movement_flags import FlagConfig  # noqa: E402

INTERSECTION_CONFIG = "three_lane"
REGIME_COLORS = {
    "L": (255, 220, 0, 255),
    "P": (255, 0, 0, 255),
    "R": (0, 200, 0, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="hzreal_cross_2_3_r1200", choices=sorted(CONFIGS),
                        help="SUMO config alias from experiments.CONFIGS.")
    parser.add_argument("--controller", default="maxpressure",
                        choices=["maxpressure", "fixedtime"])
    parser.add_argument("--until", type=int, default=9000, help="Simulated seconds to run.")
    parser.add_argument("--cycle_s", type=int, default=DEFAULT_CYCLE_S)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay", type=int, default=0, help="sumo-gui delay in ms per step.")
    parser.add_argument("--tls", default="2_3", help="Intersection to centre the view on.")
    parser.add_argument("--zoom", type=float, default=150.0, help="sumo-gui zoom (percent).")
    parser.add_argument("--pause-on-change", action="store_true",
                        help="Wait for Enter whenever any regime changes.")
    parser.add_argument("--color", action="store_true",
                        help="Colour vehicles on extent lanes by regime.")
    parser.add_argument("--logs_dir", default="logs",
                        help="The run is written under <logs_dir>/watch_extents/.")
    return parser.parse_args()


# --- formatting ------------------------------------------------------------------

def format_change(previous, current):
    return current if previous in (None, current) else f"{previous}->{current}"


def format_network_line(cycle_start, cycle_s, row, previous_regime):
    return (f"[{cycle_start:>5}-{cycle_start + cycle_s - 1:>5}] network "
            f"{format_change(previous_regime, row['regime']):<4} "
            f"active {row['n_active']:>3}  +{row['entries']} -{row['exits']}")


def format_extent_line(row, previous_regime):
    tags = "NEW " if row.newborn else "GONE" if row.vanished else "    "
    members = "+".join(sorted(row.members)) or "-"
    return (f"    extent {row.extent_id:>2}  {format_change(previous_regime, row.regime):<4} "
            f"{tags}  lanes {len(row.members)} conns {row.n_conns:>2}  "
            f"E {row.entries} X {row.exits}  {members}")


# --- GUI -------------------------------------------------------------------------

def color_lane_vehicles(lanes, color):
    for lane in lanes:
        for vehicle_id in traci.lane.getLastStepVehicleIDs(lane):
            traci.vehicle.setColor(vehicle_id, color)


def centre_view(tls, zoom):
    tls_id = tls if tls.startswith("intersection_") else f"intersection_{tls}"
    x, y = traci.junction.getPosition(tls_id)
    traci.gui.setOffset(traci.gui.DEFAULT_VIEW, x, y)
    traci.gui.setZoom(traci.gui.DEFAULT_VIEW, zoom)


# --- cycle printer -------------------------------------------------------------------

class CyclePrinter:
    """Diagnoser cycle_listener: prints each closed cycle."""

    def __init__(self, cycle_s, pause_on_change, color):
        self.cycle_s = cycle_s
        self.pause_on_change = pause_on_change
        self.color = color
        self.network_regime = None
        self.extent_regimes = {}

    def __call__(self, cycle_start, network_row, extent_rows):
        changed = self.network_regime not in (None, network_row["regime"])
        print(format_network_line(cycle_start, self.cycle_s, network_row, self.network_regime))
        self.network_regime = network_row["regime"]
        for row in extent_rows:
            previous = self.extent_regimes.get(row.extent_id)
            changed = changed or previous != row.regime
            print(format_extent_line(row, previous))
            self.extent_regimes[row.extent_id] = row.regime
            if self.color and row.regime in REGIME_COLORS:
                color_lane_vehicles(row.members, REGIME_COLORS[row.regime])
        if changed and self.pause_on_change:
            input("[regime changed] Enter to continue... ")


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    conf = INTERSECTION_CONFIGS[INTERSECTION_CONFIG]
    simulation_config = CONFIGS[args.config]
    test_name = f"watch_{args.controller}_{args.config}"
    run_meta = {"test_name": test_name, "controller": args.controller,
                "simulation_steps": args.until, "simulation_config": simulation_config,
                "intersection_config": INTERSECTION_CONFIG, "seed": args.seed,
                "blockage_scenario": None}
    ctx = setup_run(conf, test_name, simulation_config, run_meta,
                    use_gui=True, seed=args.seed, run_group="watch_extents",
                    logs_dir=args.logs_dir, movement_flag_config=FlagConfig(),
                    regime_cycle_s=args.cycle_s,
                    extra_sumo_args=["--start", "--delay", str(args.delay)])
    try:
        centre_view(args.tls, args.zoom)
    except BaseException:
        ctx.env.close()
        raise
    ctx.diagnoser.cycle_listener = CyclePrinter(args.cycle_s, args.pause_on_change, args.color)
    print(f"Run dir: {ctx.records_dir}")
    print(f"{args.controller} on {args.config}, cycle {args.cycle_s} s, flags {FlagConfig()}")

    controller = build_controller(args.controller, conf, list(ctx.handlers))

    def decide(step, intersection_id, handler):
        phase, _ = controller.choose(intersection_id, handler)
        return phase

    run_control_loop(ctx, args.until, decide)


if __name__ == "__main__":
    main()
