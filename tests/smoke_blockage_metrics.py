"""Live metric-isolation check: an obstacle with ZERO real traffic must leave
every per-vehicle metric at zero.

Runs the toy net with an empty route file so the blockage obstacle is the only
vehicle that ever exists. If any filter in MetricsRecorder or SumoEnv.get_state
misses the obstacle prefix, some metric here comes out non-zero.

The one number the obstacle legitimately CANNOT be filtered from is SUMO's own
vehicleTripStatistics (sumo_trip_count) -- asserted to be exactly the obstacle,
documenting the +1 caveat from METRICS.md.

Needs SUMO on PATH. Run: PYTHONPATH=<repo root> python tests/smoke_blockage_metrics.py
"""

import sys
import tempfile
from pathlib import Path

from configurations import INTERSECTION_CONFIGS
from sumo_env import SumoEnv
from utils.blockage_manager import BlockageManager, load_scenario
from utils.metrics_recorder import MetricsRecorder

SCENARIO = "simulations/single_intersection/scenarios/accident_single_lane.json"
SUMOCFG = "simulations/single_intersection/run.sumocfg"

EMPTY_ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15"/>
</routes>
"""

failures = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main():
    scenario = load_scenario(SCENARIO)
    with tempfile.TemporaryDirectory() as tmp:
        routes_path = Path(tmp) / "empty.rou.xml"
        routes_path.write_text(EMPTY_ROUTES)
        run_dir = Path(tmp) / "run"

        manager = BlockageManager(scenario["blockages"])
        env = SumoEnv(sumo_config=SUMOCFG,
                      intersection_config=INTERSECTION_CONFIGS["single_intersection"],
                      output_dir=run_dir,
                      blockage_manager=manager)
        env.cmd += ["--route-files", str(routes_path)]  # override: no demand
        recorder = MetricsRecorder(run_dir=run_dir, verbose=False,
                                   blockage_manager=manager)
        env.start_simulation()
        recorder.record_initial_load()

        state_during = None
        for step in range(1, 1001):
            env.step()
            recorder.record_step_summary(step)
            if step % 30 == 0:
                recorder.record_decision_wait()
            if step == 600:
                state_during = env.get_state("TLS")

        env.close()
        summary = recorder.get_final_summary()

    check(summary["total_departed_vehicles"] == 0,
          f"obstacle not counted as departed ({summary['total_departed_vehicles']})")
    check(summary["still_running_at_end"] == 0,
          f"obstacle not stuck in still_running ({summary['still_running_at_end']})")
    check(summary["cityflow_style_vehicle_count"] == 0,
          f"obstacle absent from CityFlow-style population "
          f"({summary['cityflow_style_vehicle_count']})")
    check(summary["average_per_decision_wait_s"] == 0,
          f"frozen obstacle's waiting time not sampled into AWT "
          f"({summary['average_per_decision_wait_s']})")
    check(summary["average_queue_length"] == 0,
          f"obstacle not counted as queued ({summary['average_queue_length']})")
    check(summary["total_completed_vehicles"] == 0,
          f"obstacle removal not counted as a completed trip "
          f"({summary['total_completed_vehicles']})")

    lane_state = state_during["lane_states"]["W2TLS_0"]
    check(lane_state["blocked"] is True, "lane flagged blocked mid-window")
    check(lane_state["early_queued"] == 0 and
          sum(lane_state["segments"].values()) == 0,
          f"obstacle invisible in get_state counts ({lane_state})")

    # The documented, unfilterable +1: SUMO's own statistics count the removed
    # obstacle as one finished trip.
    check(summary.get("sumo_trip_count") == 1,
          f"SUMO's own trip stats see exactly the obstacle "
          f"({summary.get('sumo_trip_count')})")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
