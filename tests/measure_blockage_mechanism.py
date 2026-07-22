"""T13 + T4 measurement: placement latency and lane-change escape of blockage
obstacles, under FixedTime control.

Two numbers in the paper's methods section come from here, per scenario:

  T13 placement latency -- an obstacle is only frozen onto its lane once a
      clear spot exists, so under load the PHYSICAL blockage onset can lag the
      scheduled start_step. Measured from the run's blockage event log
      (activated / placement_deferred / obstacle_placed).
  T4 escape rate -- vehicles trapped behind the obstacle may legally change
      lanes and cross via a neighboring lane; SUMO actively tries. The rate at
      which they succeed is part of what "blocked" actually means. Measured by
      polling the blocked edge every step: any vehicle first seen on the
      blocked lane behind the obstacle that crosses the stop line while the
      blockage is active must have escaped through another lane.

      Entrants are split by whether the junction connections give their next
      route edge a way out via another lane of the same edge:
        blocked-movement demand -- only the blocked lane serves their turn.
            This is the demand the manipulation corrupts. They can still
            cross by OVERTAKING: merge to a sibling lane, pass the obstacle,
            and merge back into the free stretch between the obstacle and
            the stop line. How leaky that is depends on the obstacle's
            position -- measured 64% crossing at 100 m upstream (S1) vs 0%
            at 10 m (S4), where there is no room to merge back. Their
            crossed rate is the paper's escape number.
        foreign-movement traffic -- another lane serves their turn; they were
            merely transiting the blocked lane and can merge off and cross.
            Their rate measures churn around the obstacle, not a leak.

Hard checks (fail the run): every obstacle activation has a placement event,
the obstacle is never absent mid-window, zero teleports, zero collisions.
Latency and escape are REPORTED, not thresholded: they define the
manipulation, so the numbers go to the log/paper as observed.

Speed-restriction blockages activate instantly and stay passable, so only
obstacle_vehicle blockages are tracked.

Writes blockage_mechanism_report.json into the run dir (a normal run dir with
manifest + event log, under logs/). Needs SUMO on PATH. Example:

    PYTHONPATH=. python tests/measure_blockage_mechanism.py \
        --blockage_scenario dataset/llm_light/Hangzhou/4_4/scenarios/s1_full_through_west_1_1.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import traci

from runner_baselines import FixedTimeController
from runner_common import setup_run, build_blockage_manager
from utils.run_manifest import build_manifest, finalize_manifest
from configurations import (
    BLOCKAGE_EVENTS_FILENAME,
    BLOCKAGE_METHOD_OBSTACLE,
    DEFAULT_INTERSECTION_CONFIG_NAME,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_SIMULATION_STEPS,
    INTERSECTION_CONFIGS,
    OBSTACLE_VEHICLE_PREFIX,
)

REPORT_FILENAME = "blockage_mechanism_report.json"

failures = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blockage_scenario", type=str, required=True)
    parser.add_argument("--test_name", type=str, default="blockage_mechanism")
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--intersection_config", type=str,
                        choices=list(INTERSECTION_CONFIGS.keys()),
                        default=DEFAULT_INTERSECTION_CONFIG_NAME)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_gui", action="store_true")
    return parser.parse_args()


class ObstacleTracker:
    """Escape/trapping observation for one obstacle_vehicle blockage.

    Entrants are vehicles first seen on the blocked lane BEHIND the obstacle
    (vehicles ahead of it form a pocket that legitimately discharges and are
    not tracked). Per entrant we keep its last known location relative to the
    blocked edge: on the blocked lane, on a sibling lane, or gone from the
    edge -- leaving the edge while the blockage is active means it crossed the
    stop line via another lane, i.e. it escaped the manipulation.
    """

    def __init__(self, blockage):
        self.blockage = blockage
        self.blockage_id = blockage["blockage_id"]
        self.lane_id = blockage["lane_id"]
        self.edge_id = self.lane_id.rsplit("_", 1)[0]
        self.obstacle_id = OBSTACLE_VEHICLE_PREFIX + self.blockage_id
        self.obstacle_lane_pos = (traci.lane.getLength(self.lane_id)
                                  - blockage["position"])
        self.escapable_edges = self._edges_served_by_other_lanes()
        self.entrant_location = {}   # veh_id -> "blocked_lane"|"other_lane"|"crossed"
        self.entrant_class = {}      # veh_id -> "trapped"|"foreign"|"ends_on_edge"
        self.ever_left_lane = set()
        self.crossed_in_window = set()
        self.obstacle_seen_step = None
        self.obstacle_missing_steps = []
        self.was_active = False

    def _edges_served_by_other_lanes(self):
        """Next edges reachable from the NON-blocked lanes of this edge: a
        vehicle heading to one of these can legally cross despite the
        blockage; anyone else the junction connections hold captive."""
        edges = set()
        for i in range(traci.edge.getLaneNumber(self.edge_id)):
            lane = f"{self.edge_id}_{i}"
            if lane == self.lane_id:
                continue
            for link in traci.lane.getLinks(lane):
                edges.add(traci.lane.getEdgeID(link[0]))
        return edges

    def _classify(self, veh):
        route = traci.vehicle.getRoute(veh)
        index = traci.vehicle.getRouteIndex(veh)
        if index + 1 >= len(route):
            return "ends_on_edge"
        return "foreign" if route[index + 1] in self.escapable_edges else "trapped"

    def poll(self, step, active_ids):
        if self.blockage_id not in active_ids:
            return
        self.was_active = True
        edge_vehicles = self._vehicles_by_lane()
        self._watch_obstacle(step, edge_vehicles)
        if self.obstacle_seen_step is None:
            return  # nothing to be trapped behind yet
        self._admit_entrants(edge_vehicles)
        self._update_entrants(edge_vehicles)

    def _vehicles_by_lane(self):
        return {
            f"{self.edge_id}_{i}": traci.lane.getLastStepVehicleIDs(f"{self.edge_id}_{i}")
            for i in range(traci.edge.getLaneNumber(self.edge_id))
        }

    def _watch_obstacle(self, step, edge_vehicles):
        present = self.obstacle_id in edge_vehicles.get(self.lane_id, ())
        if present and self.obstacle_seen_step is None:
            self.obstacle_seen_step = step
        elif not present and self.obstacle_seen_step is not None:
            self.obstacle_missing_steps.append(step)

    def _admit_entrants(self, edge_vehicles):
        for veh in edge_vehicles[self.lane_id]:
            if veh.startswith(OBSTACLE_VEHICLE_PREFIX) or veh in self.entrant_location:
                continue
            if traci.vehicle.getLanePosition(veh) < self.obstacle_lane_pos:
                self.entrant_location[veh] = "blocked_lane"
                self.entrant_class[veh] = self._classify(veh)

    def _update_entrants(self, edge_vehicles):
        on_edge = {veh: lane for lane, vehs in edge_vehicles.items() for veh in vehs}
        for veh, location in self.entrant_location.items():
            if location == "crossed":
                continue
            lane = on_edge.get(veh)
            if lane is None:
                self.entrant_location[veh] = "crossed"
                self.crossed_in_window.add(veh)
            elif lane != self.lane_id:
                self.entrant_location[veh] = "other_lane"
                self.ever_left_lane.add(veh)
            else:
                self.entrant_location[veh] = "blocked_lane"

    def summary(self):
        def class_stats(cls):
            members = [v for v, c in self.entrant_class.items() if c == cls]
            crossed = sum(v in self.crossed_in_window for v in members)
            return {
                "entrants": len(members),
                "crossed_in_window": crossed,
                "crossed_rate": (round(crossed / len(members), 4)
                                 if members else None),
                "ever_left_blocked_lane": sum(v in self.ever_left_lane
                                              for v in members),
                "crossed_without_leaving_lane": sum(
                    v in self.crossed_in_window and v not in self.ever_left_lane
                    for v in members),
                "on_blocked_lane_at_end": sum(
                    self.entrant_location[v] == "blocked_lane" for v in members),
                "on_other_lane_at_end": sum(
                    self.entrant_location[v] == "other_lane" for v in members),
            }

        return {
            "blockage_id": self.blockage_id,
            "lane_id": self.lane_id,
            "entrants_behind_obstacle": len(self.entrant_location),
            "blocked_movement_demand": class_stats("trapped"),
            "foreign_movement_traffic": class_stats("foreign"),
            "route_ends_on_edge": class_stats("ends_on_edge"),
            "obstacle_missing_steps": len(self.obstacle_missing_steps),
        }


def placement_latency(records_dir, blockages):
    """T13, from the event log: per obstacle blockage, when activation fired,
    how often placement was deferred, and when the obstacle was really frozen."""
    events = [json.loads(line)
              for line in (records_dir / BLOCKAGE_EVENTS_FILENAME).open()]
    results = []
    for blockage in blockages:
        if blockage["method"] != BLOCKAGE_METHOD_OBSTACLE:
            continue
        own = [e for e in events if e["blockage_id"] == blockage["blockage_id"]]
        placed = next((e for e in own if e["event"] == "obstacle_placed"), None)
        results.append({
            "blockage_id": blockage["blockage_id"],
            "lane_id": blockage["lane_id"],
            "scheduled_start": blockage["start_step"],
            "activated_at": next((e["sim_time"] for e in own
                                  if e["event"] == "activated"), None),
            "placed_at": placed["sim_time"] if placed else None,
            "placement_lag_s": (placed["sim_time"] - blockage["start_step"]
                                if placed else None),
            "frozen_at_lane_position_m": placed["lane_position_m"] if placed else None,
            "deferral_warnings": sum(e["event"] == "placement_deferred" for e in own),
        })
    return results


def main(args):
    conf = INTERSECTION_CONFIGS[args.intersection_config]
    scenario, manager = build_blockage_manager(args.blockage_scenario)

    run_meta = {
        "test_name": args.test_name,
        "controller": "fixedtime",
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "intersection_config": args.intersection_config,
        "seed": args.seed,
        "blockage_scenario": args.blockage_scenario,
    }
    manifest = build_manifest("fixedtime", args, args.intersection_config, conf)
    ctx = setup_run(conf, args.test_name, args.simulation_config, run_meta,
                    use_gui=args.use_gui, seed=args.seed,
                    blockage_manager=manager, manifest=manifest)
    controller = FixedTimeController(conf)

    trackers = [ObstacleTracker(b) for b in scenario["blockages"]
                if b["method"] == BLOCKAGE_METHOD_OBSTACLE]
    teleported = []
    collisions = []

    # run_control_loop drives normal runs; forked here because the trackers
    # need a hook after every env.step(), which the shared loop has no seam for.
    status = "crashed"
    try:
        for step in range(args.simulation_steps):
            ctx.env.step()
            ctx.recorder.record_step_summary(step)
            active_ids = {b["blockage_id"] for b in manager.get_active_blockages()}
            for tracker in trackers:
                tracker.poll(step, active_ids)
            if teleporting := traci.simulation.getStartingTeleportIDList():
                teleported.append((step, teleporting))
            if colliding := traci.simulation.getCollidingVehiclesIDList():
                collisions.append((step, colliding))
            for intersection_id, handler in ctx.handlers.items():
                handler.step()
                if handler.switch_phase:
                    phase, _ = controller.choose(intersection_id, handler)
                    ctx.recorder.record_decision_wait()
                    handler.activate_phase(phase)
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected at step {step}, stopping early.")
                break
        status = "completed"
    finally:
        try:
            ctx.env.close()
            ctx.recorder.save_final_summary()
        finally:
            finalize_manifest(ctx.records_dir, status)

    latency = placement_latency(ctx.records_dir, scenario["blockages"])
    escape = [t.summary() for t in trackers]

    print(f"\n=== Blockage mechanism report: {scenario['scenario_name']} ===")
    print(f"run dir: {ctx.records_dir}")
    for row in latency:
        print(f"\nT13 placement -- {row['blockage_id']} on {row['lane_id']} "
              f"(scheduled t={row['scheduled_start']}):")
        print(f"  activated t={row['activated_at']}, placed t={row['placed_at']}, "
              f"lag {row['placement_lag_s']} s, "
              f"deferral warnings {row['deferral_warnings']}, "
              f"frozen at {row['frozen_at_lane_position_m']} m")
    for row in escape:
        print(f"\nT4 escape -- {row['blockage_id']} on {row['lane_id']} "
              f"({row['entrants_behind_obstacle']} entrants behind obstacle):")
        for label, key in (("blocked-movement demand", "blocked_movement_demand"),
                           ("foreign-movement traffic", "foreign_movement_traffic"),
                           ("route ends on this edge", "route_ends_on_edge")):
            stats = row[key]
            print(f"  {label}: {stats['entrants']} entrants, "
                  f"{stats['crossed_in_window']} crossed in window "
                  f"(rate {stats['crossed_rate']}), "
                  f"{stats['ever_left_blocked_lane']} ever left the lane, "
                  f"at end {stats['on_blocked_lane_at_end']} on blocked lane / "
                  f"{stats['on_other_lane_at_end']} on sibling lane")
    print()

    for row in latency:
        check(row["placed_at"] is not None,
              f"{row['blockage_id']}: obstacle_placed event exists")
    for tracker, row in zip(trackers, escape):
        check(tracker.was_active, f"{row['blockage_id']}: blockage became active")
        check(row["obstacle_missing_steps"] == 0,
              f"{row['blockage_id']}: obstacle never absent mid-window "
              f"({row['obstacle_missing_steps']} missing steps)")
        # Crossing is legitimate only by overtaking via a sibling lane;
        # passing the obstacle in-lane is physically impossible, so a crosser
        # never seen off the blocked lane means the tracker or sim is wrong.
        skipped_past = sum(row[key]["crossed_without_leaving_lane"]
                           for key in ("blocked_movement_demand",
                                       "foreign_movement_traffic",
                                       "route_ends_on_edge"))
        check(skipped_past == 0,
              f"{row['blockage_id']}: every crosser left the blocked lane "
              f"first ({skipped_past} did not)")
    check(not teleported, f"zero teleports ({teleported[:3]})"
          if teleported else "zero teleports")
    check(not collisions, f"zero collisions ({collisions[:3]})"
          if collisions else "zero collisions")

    report = {
        "scenario_name": scenario["scenario_name"],
        "blockage_scenario": args.blockage_scenario,
        "simulation_config": args.simulation_config,
        "simulation_steps": args.simulation_steps,
        "seed": args.seed,
        "controller": "fixedtime",
        "placement_latency": latency,
        "escape": escape,
        "teleport_events": len(teleported),
        "collision_events": len(collisions),
        "checks_failed": list(failures),
    }
    report_path = ctx.records_dir / REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {report_path}")
    print(f"{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
