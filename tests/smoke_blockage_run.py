"""Live SUMO smoke test for the blockage feature.

Drives the committed single_intersection toy net (real demand, including the
blocked lane) with the accident_single_lane scenario and asserts the physics
that the source repo never validated:

  - the obstacle is placed promptly and SURVIVES the whole window (the direct
    probe for teleport regressions -- teleport counters alone can miss the
    obstacle deleting itself),
  - a queue physically forms on the blocked lane,
  - the blocked movement's arrivals freeze during the window,
  - no teleports happen at all,
  - get_state flags the lane and describe_blockages renders the right
    approach/movement/segment.

Needs SUMO on PATH. Run: PYTHONPATH=<repo root> python tests/smoke_blockage_run.py
"""

import sys

import traci

from configurations import INTERSECTION_CONFIGS, OBSTACLE_VEHICLE_PREFIX
from sumo_env import SumoEnv
from utils.blockage_manager import BlockageManager, load_scenario

SCENARIO = "simulations/single_intersection/scenarios/accident_single_lane.json"
SUMOCFG = "simulations/single_intersection/run.sumocfg"
BLOCKED_LANE = "W2TLS_0"
OBSTACLE_ID = OBSTACLE_VEHICLE_PREFIX + "b1"
START, END = 300, 900
TOTAL_STEPS = 1100

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
                  intersection_config=INTERSECTION_CONFIGS["single_intersection"],
                  blockage_manager=manager)
    env.start_simulation()
    check("--time-to-teleport" in env.cmd, "teleporting disabled for the blockage run")

    placement_step = None
    obstacle_gaps = []          # steps in the window where the obstacle was missing
    teleport_steps = []
    max_halting_in_window = 0
    blocked_arrivals_in_window = []
    state_flag_during = None
    state_flag_after = None
    descriptions_during = None

    for step in range(1, TOTAL_STEPS + 1):
        env.step()

        vehicles = set(traci.vehicle.getIDList())
        if placement_step is None and OBSTACLE_ID in vehicles:
            placement_step = step
        if teleporting := traci.simulation.getStartingTeleportIDList():
            teleport_steps.append((step, teleporting))

        if START < step < END:
            if placement_step is not None and OBSTACLE_ID not in vehicles:
                obstacle_gaps.append(step)
            max_halting_in_window = max(
                max_halting_in_window,
                traci.lane.getLastStepHaltingNumber(BLOCKED_LANE))

        # Vehicles already DOWNSTREAM of the obstacle at activation (the
        # servable pool) legitimately keep arriving for a few green cycles.
        # Once they drain, a truly blocked lane produces zero new arrivals.
        if 600 <= step < END:
            blocked_arrivals_in_window += [
                v for v in traci.simulation.getArrivedIDList()
                if v.startswith("flow_W2E")
            ]

        if step == 600:
            state = env.get_state("TLS")
            state_flag_during = state["lane_states"][BLOCKED_LANE]["blocked"]
            descriptions_during = env.describe_blockages("TLS")
        if step == 1000:
            state_flag_after = env.get_state("TLS")["lane_states"][BLOCKED_LANE]["blocked"]
            obstacle_removed = OBSTACLE_ID not in vehicles

    env.close()

    check(placement_step is not None and placement_step <= START + 10,
          f"obstacle placed promptly (step {placement_step})")
    check(not obstacle_gaps,
          f"obstacle present for the whole window (missing at {obstacle_gaps[:5]}...)"
          if obstacle_gaps else "obstacle present for the whole window")
    check(obstacle_removed, "obstacle removed after end_step")
    check(not teleport_steps, f"no teleports (saw {teleport_steps[:3]})"
          if teleport_steps else "no teleports")
    check(max_halting_in_window >= 5,
          f"queue formed on blocked lane (max halting {max_halting_in_window})")
    check(not blocked_arrivals_in_window,
          f"blocked movement arrivals cease once the servable pool drains "
          f"({len(blocked_arrivals_in_window)} arrived after step 600)")
    check(state_flag_during is True, "get_state flags the blocked lane during the window")
    check(state_flag_after is False, "get_state clears the flag after the window")
    check(descriptions_during == [{
        "lane_id": BLOCKED_LANE, "approach": "West", "movement": "ETWT",
        "segment": 3, "method": "obstacle_vehicle", "severity": 1.0,
    }], f"describe_blockages renders approach/movement/segment ({descriptions_during})")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
