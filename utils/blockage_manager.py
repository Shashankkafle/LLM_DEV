"""Lane blockage injection for SUMO simulations.

A scenario JSON declares scheduled blockages; BlockageManager executes them
against the live TraCI session, ticked once per simulation step. Two methods:

  obstacle_vehicle  -- inserts a vehicle and freezes it at a lane position,
                       physically blocking that lane.
  speed_restriction -- reduces the lane's max speed (0 at severity 1.0);
                       original speed restored on deactivation. Affects the
                       whole lane; position is ignored.

Scenario JSON shape (see simulations/single_intersection/scenarios/):

    {
      "scenario_name": "accident_single_lane",
      "description": "...",
      "blockages": [
        {"blockage_id": "b1", "lane_id": "W2TLS_0", "position": 80.0,
         "start_step": 300, "end_step": 900,
         "method": "obstacle_vehicle", "severity": 1.0,
         "cause": "collision"}
      ]
    }

position is metres UPSTREAM of the stop line (portable across lanes of
different lengths). start_step/end_step are simulation seconds; end_step null
means "until the run ends". cause is the short factual incident description
the prompt's blockage reports show verbatim ("collision", "roadworks", ...);
it is optional and defaults per method (configurations.BLOCKAGE_DEFAULT_CAUSE).

traci is imported inside the methods that touch it (same convention as
utils/state_features.py), so the pure parts -- load_scenario and
schedule_transitions -- stay importable and testable without SUMO installed.
"""

import json

from configurations import (
    BLOCKAGE_DEFAULT_CAUSE,
    BLOCKAGE_METHOD_OBSTACLE,
    BLOCKAGE_METHOD_SPEED,
    OBSTACLE_VEHICLE_PREFIX,
    OBSTACLE_CLEARANCE_M,
)

_REQUIRED_KEYS = {"blockage_id", "lane_id", "position", "start_step",
                  "end_step", "method"}
_ALLOWED_KEYS = _REQUIRED_KEYS | {"severity", "cause"}
_METHODS = {BLOCKAGE_METHOD_OBSTACLE, BLOCKAGE_METHOD_SPEED}


def load_scenario(path):
    """Load and strictly validate a blockage scenario JSON.

    Returns the scenario dict {scenario_name, description, blockages}, with
    severity defaulted to 1.0 on each blockage. Validation is deliberately
    strict (unknown keys are errors): blockage dicts are used raw throughout
    a run, so a typo like "startstep" must fail here at load time, not
    surface as a silently-never-firing blockage mid-run.
    """
    with open(path) as f:
        scenario = json.load(f)

    for key in ("scenario_name", "description", "blockages"):
        if key not in scenario:
            raise ValueError(f"Scenario file {path} is missing '{key}'")

    seen_ids = set()
    for blockage in scenario["blockages"]:
        bid = blockage.get("blockage_id", "<missing blockage_id>")
        missing = _REQUIRED_KEYS - set(blockage)
        if missing:
            raise ValueError(f"Blockage '{bid}' is missing keys: {sorted(missing)}")
        unknown = set(blockage) - _ALLOWED_KEYS
        if unknown:
            raise ValueError(f"Blockage '{bid}' has unknown keys: {sorted(unknown)}")
        if bid in seen_ids:
            raise ValueError(f"Duplicate blockage_id '{bid}'")
        seen_ids.add(bid)

        if blockage["method"] not in _METHODS:
            raise ValueError(
                f"Blockage '{bid}': unknown method '{blockage['method']}' "
                f"(expected one of {sorted(_METHODS)})")
        blockage.setdefault("severity", 1.0)
        if not 0.0 <= blockage["severity"] <= 1.0:
            raise ValueError(f"Blockage '{bid}': severity must be in [0, 1]")
        blockage.setdefault("cause", BLOCKAGE_DEFAULT_CAUSE[blockage["method"]])
        if not isinstance(blockage["cause"], str) or not blockage["cause"].strip():
            raise ValueError(f"Blockage '{bid}': cause must be a non-empty string")
        if blockage["position"] < 0:
            raise ValueError(f"Blockage '{bid}': position must be >= 0")
        if blockage["start_step"] < 0:
            raise ValueError(f"Blockage '{bid}': start_step must be >= 0")
        end = blockage["end_step"]
        if end is not None and end <= blockage["start_step"]:
            raise ValueError(f"Blockage '{bid}': end_step must be null or > start_step")

    _reject_overlapping_speed_restrictions(scenario["blockages"])
    return scenario


def _reject_overlapping_speed_restrictions(blockages):
    """Two speed restrictions live on the same lane at once would clobber the
    saved original speed (the restore map is keyed by lane), leaving the lane
    permanently slow. Reject the scenario instead."""
    restrictions = [b for b in blockages if b["method"] == BLOCKAGE_METHOD_SPEED]
    for i, a in enumerate(restrictions):
        for b in restrictions[i + 1:]:
            if a["lane_id"] != b["lane_id"]:
                continue
            a_end = a["end_step"] if a["end_step"] is not None else float("inf")
            b_end = b["end_step"] if b["end_step"] is not None else float("inf")
            if a["start_step"] < b_end and b["start_step"] < a_end:
                raise ValueError(
                    f"Speed restrictions '{a['blockage_id']}' and "
                    f"'{b['blockage_id']}' overlap on lane {a['lane_id']}")


def schedule_transitions(schedule, current_step, active_ids, finished_ids):
    """Pure schedule logic: what to do at this tick.

    Returns (to_activate, to_deactivate, expired) as lists of blockage dicts.
    expired holds blockages whose whole [start_step, end_step) window is
    already past before they ever activated (e.g. a clock jump). They must go
    straight to finished WITHOUT activating -- activating and deactivating an
    obstacle in the same tick can leak an unmanaged vehicle out of SUMO's
    insertion queue.
    """
    to_activate, to_deactivate, expired = [], [], []
    for blockage in schedule:
        bid = blockage["blockage_id"]
        ended = (blockage["end_step"] is not None
                 and current_step >= blockage["end_step"])
        if bid in active_ids:
            if ended:
                to_deactivate.append(blockage)
        elif bid not in finished_ids and current_step >= blockage["start_step"]:
            if ended:
                expired.append(blockage)
            else:
                to_activate.append(blockage)
    return to_activate, to_deactivate, expired


class BlockageManager:
    """Executes a validated blockage schedule against the live TraCI session.

    step(t) must be called once per simulation step, after
    traci.simulationStep(), with t in SIMULATION SECONDS. Every loop that
    ticks this manager uses that same clock (env.get_current_step() /
    traci.simulation.getTime()), so a scenario means the same thing in every
    runner, including replay.
    """

    def __init__(self, blockages, scenario_name=None):
        # Validated scenario data; read-only after construction. Public so the
        # env can check lane attribution against the whole schedule, not just
        # the currently-active subset. scenario_name is provenance: it ends up
        # in final_summary.json so blockage runs are distinguishable in tables.
        self.schedule = blockages
        self.scenario_name = scenario_name
        self._active = {}             # blockage_id -> blockage dict
        self._finished = set()        # blockage_ids past their window
        self._pending_obstacles = {}  # veh_id -> blockage awaiting placement
        self._original_speeds = {}    # lane_id -> max speed before restriction
        self._deferral_warned = set()

    def reset(self):
        """Forget all mutable state, for reuse across SUMO sessions (e.g.
        CoLight builds a fresh env per training round). traci.close() already
        removed the sim-side effects, so no TraCI calls belong here."""
        self._active.clear()
        self._finished.clear()
        self._pending_obstacles.clear()
        self._original_speeds.clear()
        self._deferral_warned.clear()

    def validate_against_network(self):
        """Fail fast right after the simulation starts instead of mid-run:
        every lane must exist, and every obstacle position must fall inside
        its lane."""
        import traci
        for blockage in self.schedule:
            lane_length = traci.lane.getLength(blockage["lane_id"])
            if (blockage["method"] == BLOCKAGE_METHOD_OBSTACLE
                    and blockage["position"] > lane_length):
                raise ValueError(
                    f"Blockage '{blockage['blockage_id']}': position "
                    f"{blockage['position']} m upstream is outside lane "
                    f"{blockage['lane_id']} (length {lane_length:.1f} m)")

    def step(self, current_step):
        to_activate, to_deactivate, expired = schedule_transitions(
            self.schedule, current_step, set(self._active), self._finished)
        for blockage in expired:
            self._finished.add(blockage["blockage_id"])
        for blockage in to_activate:
            self._activate(blockage)
        self._place_pending_obstacles()
        for blockage in to_deactivate:
            self._deactivate(blockage)
            self._finished.add(blockage["blockage_id"])

    def get_active_blockages(self):
        return list(self._active.values())

    def get_blocked_lane_ids(self):
        return [b["lane_id"] for b in self._active.values()]

    def _activate(self, blockage):
        self._active[blockage["blockage_id"]] = blockage
        print(f"[blockage] activating '{blockage['blockage_id']}' on "
              f"{blockage['lane_id']} ({blockage['method']})")
        if blockage["method"] == BLOCKAGE_METHOD_OBSTACLE:
            self._activate_obstacle_vehicle(blockage)
        elif blockage["method"] == BLOCKAGE_METHOD_SPEED:
            self._activate_speed_restriction(blockage)
        else:
            # load_scenario already rejects unknown methods; backstop in case
            # a schedule was built without going through it.
            raise ValueError(f"Unknown blockage method '{blockage['method']}'")

    def _activate_obstacle_vehicle(self, blockage):
        """Insert the obstacle into SUMO's departure queue; freezing it at the
        target position happens in _place_pending_obstacles.

        ASSUMPTION (SUMO insertion model): vehicle.add() only queues the
        vehicle. On current SUMO a moveTo can force-place it immediately, but
        older versions need a simulationStep() first -- so placement lives in
        a retry loop rather than assuming either behavior.
        """
        import traci
        bid = blockage["blockage_id"]
        veh_id = OBSTACLE_VEHICLE_PREFIX + bid
        edge_id = blockage["lane_id"].rsplit("_", 1)[0]
        traci.route.add(f"{OBSTACLE_VEHICLE_PREFIX}route_{bid}", [edge_id])
        traci.vehicle.add(
            vehID=veh_id,
            routeID=f"{OBSTACLE_VEHICLE_PREFIX}route_{bid}",
            typeID="DEFAULT_VEHTYPE",
            depart="now",
            departLane="first",
            departPos="0",
            departSpeed="0",
        )
        self._pending_obstacles[veh_id] = blockage

    def _place_pending_obstacles(self):
        """Try to freeze every not-yet-placed obstacle at its target position.
        Retried every tick: placement is deferred while the vehicle is still
        in the insertion queue or while the target spot is occupied."""
        if not self._pending_obstacles:
            return
        import traci
        for veh_id, blockage in list(self._pending_obstacles.items()):
            lane_id = blockage["lane_id"]
            position = self._lane_position_from_stopline(
                blockage["position"], lane_id)
            if not self._position_is_clear(lane_id, position, veh_id):
                if blockage["blockage_id"] not in self._deferral_warned:
                    print(f"[blockage] target spot on {lane_id} is occupied; "
                          f"deferring placement of '{blockage['blockage_id']}'")
                    self._deferral_warned.add(blockage["blockage_id"])
                continue
            try:
                traci.vehicle.moveTo(veh_id, lane_id, position)
                traci.vehicle.setSpeed(veh_id, 0.0)
                # SpeedMode 0 disables all speed influencing (car-following,
                # traffic-light braking); LaneChangeMode 0 pins the lane.
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setLaneChangeMode(veh_id, 0)
                del self._pending_obstacles[veh_id]
                print(f"[blockage] obstacle '{veh_id}' frozen on {lane_id} "
                      f"at {position:.1f} m")
            except traci.TraCIException:
                pass  # still in the insertion queue; retry next tick

    def _position_is_clear(self, lane_id, position, obstacle_veh_id):
        """moveTo onto an occupied spot "succeeds" by overlapping vehicles,
        which either reports a collision every step (collision.action=warn) or
        gets the obstacle deleted one step later (action=teleport). Defer
        placement until the spot is clear instead."""
        import traci
        for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
            if veh_id == obstacle_veh_id:
                continue
            if abs(traci.vehicle.getLanePosition(veh_id) - position) < OBSTACLE_CLEARANCE_M:
                return False
        return True

    def _lane_position_from_stopline(self, distance_from_stopline, lane_id):
        """Scenario positions are metres upstream of the stop line; TraCI
        wants metres from the lane start."""
        import traci
        lane_length = traci.lane.getLength(lane_id)
        lane_pos = lane_length - distance_from_stopline
        if not 0 <= lane_pos <= lane_length:
            raise ValueError(
                f"Blockage position {distance_from_stopline} m upstream is "
                f"outside lane {lane_id} (length {lane_length:.1f} m)")
        return lane_pos

    def _activate_speed_restriction(self, blockage):
        import traci
        lane_id = blockage["lane_id"]
        # load_scenario rejects overlapping restrictions on one lane, so the
        # saved speed is always the true unrestricted speed.
        self._original_speeds[lane_id] = traci.lane.getMaxSpeed(lane_id)
        traci.lane.setMaxSpeed(
            lane_id, self._original_speeds[lane_id] * (1.0 - blockage["severity"]))

    def _deactivate(self, blockage):
        import traci
        bid = blockage["blockage_id"]
        if blockage["method"] == BLOCKAGE_METHOD_OBSTACLE:
            veh_id = OBSTACLE_VEHICLE_PREFIX + bid
            try:
                traci.vehicle.remove(veh_id)
            except traci.TraCIException:
                pass  # never successfully placed, or already gone
            self._pending_obstacles.pop(veh_id, None)
        elif blockage["lane_id"] in self._original_speeds:
            traci.lane.setMaxSpeed(
                blockage["lane_id"], self._original_speeds.pop(blockage["lane_id"]))
        self._active.pop(bid, None)
        print(f"[blockage] deactivated '{bid}'")
