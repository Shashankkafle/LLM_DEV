"""Live SUMO smoke test: a blockage is described to BOTH its controllers.

The Hangzhou 4x4 scenario blocks lane road_1_4_3_0. Edge road_1_4_3 runs
intersection_1_4 -> intersection_1_3 heading South, so:

  - intersection_1_3 (downstream of the blockage) sees it via
    describe_blockages, with the approach vocabulary -- unchanged behavior,
  - intersection_1_4 (upstream of the blockage) sees it via the new
    describe_exit_blockages, with direction + feeding movements,
  - every other intersection sees nothing from either describer.

Needs SUMO on PATH. Run: PYTHONPATH=<repo root> python tests/smoke_exit_blockage_hangzhou.py
"""

import sys

from configurations import INTERSECTION_CONFIGS
from sumo_env import SumoEnv
from utils.blockage_manager import BlockageManager, load_scenario

SCENARIO = "dataset/llm_light/Hangzhou/4_4/lane_blockages.json"
SUMOCFG = "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.sumocfg"
BLOCKED_LANE = "road_1_4_3_0"
DOWNSTREAM_TL = "intersection_1_3"
UPSTREAM_TL = "intersection_1_4"

failures = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main():
    scenario = load_scenario(SCENARIO)
    manager = BlockageManager(scenario["blockages"])
    env = SumoEnv(sumo_config=SUMOCFG,
                  intersection_config=INTERSECTION_CONFIGS["three_lane"],
                  blockage_manager=manager)
    env.start_simulation()

    # The scenario activates at step 1; a few extra steps let placement settle.
    for _ in range(30):
        env.step()

    approach_down = env.describe_blockages(DOWNSTREAM_TL)
    exit_up = env.describe_exit_blockages(UPSTREAM_TL)
    approach_up = env.describe_blockages(UPSTREAM_TL)
    exit_down = env.describe_exit_blockages(DOWNSTREAM_TL)
    others_clean = all(
        env.describe_blockages(tl) == [] and env.describe_exit_blockages(tl) == []
        for tl in env.get_intersections()
        if tl not in (DOWNSTREAM_TL, UPSTREAM_TL)
    )
    env.close()

    check(len(approach_down) == 1
          and approach_down[0]["lane_id"] == BLOCKED_LANE
          and approach_down[0]["approach"] == "North"
          and approach_down[0]["segment"] == 1
          and approach_down[0]["method"] == "obstacle_vehicle",
          f"downstream TL gets the approach description ({approach_down})")

    check(len(exit_up) == 1, f"upstream TL gets exactly one exit description ({exit_up})")
    if len(exit_up) == 1:
        d = exit_up[0]
        check(d["exit_direction"] == "South",
              f"exit direction is South ({d['exit_direction']})")
        check(d["feeding_movements"] == ["ELWL", "NTST"],
              f"feeding movements are ELWL+NTST ({d['feeding_movements']})")
        check(d["lane_count"] == 3 and d["blocked_lane_index"] == 0,
              f"lane facts (count {d['lane_count']}, index {d['blocked_lane_index']})")
        check(abs(d["distance_m"] - 562.8) < 0.1,
              f"distance from upstream intersection ({d['distance_m']:.1f} m)")

    check(approach_up == [], "upstream TL has no approach description")
    check(exit_down == [], "downstream TL has no exit description")
    check(others_clean, "all other intersections see nothing")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
