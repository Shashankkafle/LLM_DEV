"""The sim<->model translator for RL controllers.

A feature REGISTRY maps a feature name to a function ``fn(env, intersection_id)``
that reads the SUMO environment and returns that feature's value. A model declares
the features it wants by name (mirroring CoLight's ``LIST_STATE_FEATURE``); the runner
emits a feature-KEYED dict per intersection, so adding features later (pressure,
time_this_phase, ...) is purely additive.

CoLight features:
  - "cur_phase"        : [logical phase index]  (agent expands to 8-bit via PHASE)
  - "lane_num_vehicle" : per-movement vehicle counts in canonical movement order
  - "adjacency_matrix" : top-k neighbor agent indices (self first)
Advanced CoLight adds (same agent, different features):
  - "traffic_movement_pressure_queue_efficient" : per-movement efficient queue pressure
  - "lane_enter_running_part"                    : near-stopline running vehicles per movement

Design: each feature has a PURE core (no TraCI; takes plain data) wrapped by a thin
env-bound registry function. The pure cores are what the unit tests exercise with
hand-built SUMO states, so feature fidelity is checked without a running simulator.
"""

import numpy as np

from configurations import (
    location_dict,
    COLIGHT_MOVEMENT_ORDER,
    COLIGHT_TOP_K_ADJACENCY,
)
from utils.general_utils import get_phase_name


# ===========================================================================
# Pure cores (no TraCI) -- unit-testable with hand-built data
# ===========================================================================

def build_phase_onehot(conf, movement_order=COLIGHT_MOVEMENT_ORDER):
    """Map each logical phase index -> 8-bit movement one-hot.

    A phase name like "ETWT" lights movements ET and WT, so those positions in
    ``movement_order`` are set to 1. Equivalent to source CoLight's hand-written
    ``PHASE`` dict, but derived from the active config so it works for any 4-phase
    protected layout (ETWT/NTST/ELWL/NLSL).
    """
    phase_map = {}
    for name, phase_cfg in conf["phases"].items():
        vec = [0] * len(movement_order)
        for token in (name[:2], name[2:]):
            vec[movement_order.index(token)] = 1
        phase_map[phase_cfg["id"]] = vec
    return phase_map


def phase_index_from_ryg(conf, ryg_string):
    """Recover the logical phase index from a SUMO green RYG string.

    Valid only when ``ryg_string`` is a known GREEN phase (the case at every
    decision point, where state is read at the end of a green window).
    """
    name = get_phase_name(conf, ryg_string)
    if name not in conf["phases"]:
        raise ValueError(
            f"cur_phase: RYG string {ryg_string!r} does not resolve to a known "
            f"green phase (got {name!r}). State must be read at a green decision point."
        )
    return [conf["phases"][name]["id"]]


def movement_vehicle_counts(state, movement_order=COLIGHT_MOVEMENT_ORDER):
    """Per-movement vehicle counts (an 8-vector) from a ``SumoEnv.get_state`` dict.

    Each movement's count = early_queued + all 3 segment counts, aggregated over
    that movement's lanes. Reuses the movement/approach breakdown ``get_state``
    already computes -- no new SUMO calls. Fixed length regardless of lanes per
    approach (2-lane vs 3-lane nets), so it ports across datasets.
    """
    counts = {m: 0 for m in movement_order}
    for phase_name, approaches in state["movement_states"].items():
        for token in (phase_name[:2], phase_name[2:]):
            approach_word = location_dict[token[0]]  # "E" -> "East"
            agg = approaches.get(approach_word)
            if agg is None:
                continue
            counts[token] = agg["early_queued"] + sum(agg["segments"].values())
    return [counts[m] for m in movement_order]


def build_adjacency_rows(positions, order, num_neighbors):
    """Top-k nearest-neighbor index rows from junction positions.

    ``positions``: {intersection_id: (x, y)}; ``order``: list of ids whose position
    in the list IS the agent index. Each row lists the num_neighbors nearest agent
    indices with self first (self distance 0). Mirrors source ``_adjacency_extraction``
    (CoLight only uses the neighbor SET; exact order is re-sorted inside the agent).
    """
    coords = np.array([positions[i] for i in order], dtype=float)
    rows = {}
    for idx, inter_id in enumerate(order):
        dist = np.sqrt(((coords - coords[idx]) ** 2).sum(axis=1))
        nearest = np.argsort(dist, kind="stable")[:num_neighbors]
        rows[inter_id] = [int(j) for j in nearest]
    return rows


def self_only_adjacency_rows(order, num_neighbors_cap):
    """Ablation rows: every intersection's neighbors are all itself.

    Through the agent's adjacency one-hot, this makes neighbor attention attend only
    to self -- removing all cross-intersection information. Used to test that the
    graph attention actually contributes on multi-intersection nets.
    """
    nn = min(num_neighbors_cap, len(order))
    return {inter_id: [idx] * nn for idx, inter_id in enumerate(order)}


def expand_state_for_memory(state, phase_map):
    """Convert a LIVE state (cur_phase = [raw index]) into the STORED form used in
    replay transitions, where cur_phase is the pre-expanded 8-bit vector.

    Mirrors source ``construct_sample.construct_state`` under BINARY_PHASE_EXPANSION:
    ``choose_action`` expands cur_phase on the fly, but ``prepare_Xs_Y`` consumes the
    stored state raw -- so transitions must carry the expanded vector or the training
    feature width won't match the network input. Other features pass through unchanged.
    """
    out = dict(state)
    raw_index = state["cur_phase"][0]
    out["cur_phase"] = list(phase_map[raw_index])
    return out


# ---------------------------------------------------------------------------
# Advanced CoLight feature cores (pure)
# ---------------------------------------------------------------------------

def movement_pressure_efficient(entering_halt, exiting_halt, exiting_lane_count,
                                movement_order=COLIGHT_MOVEMENT_ORDER):
    """Efficient queue pressure per movement: entering queue minus the per-lane-
    normalized exiting (downstream) queue.

    Source `_get_traffic_movement_pressure_efficient`: pressure = entering_queue -
    exiting_queue / 3, where 3 is CityFlow's lanes-per-approach. Here we normalize by
    the actual outgoing-edge lane count, so it generalizes to 2-lane and 3-lane nets.
    Queues are halting (stopped) counts, matching the source's waiting-vehicle queue.
    """
    out = []
    for m in movement_order:
        n_lanes = exiting_lane_count.get(m, 0) or 0
        exit_norm = (exiting_halt.get(m, 0.0) / n_lanes) if n_lanes else 0.0
        out.append(entering_halt.get(m, 0.0) - exit_norm)
    return out


def movement_segment1_counts(state, movement_order=COLIGHT_MOVEMENT_ORDER):
    """Moving vehicles in the segment nearest the stopline, per movement (an 8-vector).

    Source `lane_enter_running_part` = running (moving) vehicles in the lane part nearest
    the stopline. `get_state`'s segment_1 already counts moving vehicles in the nearest
    third of the lane (stopped vehicles are excluded into early_queued), so we reuse it.
    (Source uses a fixed near-stopline window; this uses the nearest third, net-relative.)
    """
    counts = {m: 0 for m in movement_order}
    for phase_name, approaches in state["movement_states"].items():
        for token in (phase_name[:2], phase_name[2:]):
            agg = approaches.get(location_dict[token[0]])
            if agg is not None:
                counts[token] = agg["segments"]["segment_1"]
    return [counts[m] for m in movement_order]


# ===========================================================================
# Env-bound registry functions
# ===========================================================================

def _cur_phase(env, intersection_id):
    return phase_index_from_ryg(env.intersection_config, env.get_current_phase(intersection_id))


def _lane_num_vehicle(env, intersection_id):
    return movement_vehicle_counts(env.get_state(intersection_id))


def _adjacency_matrix(env, intersection_id):
    import traci  # provided by $SUMO_HOME/tools at runtime (see sumo_env.py)
    cache = getattr(env, "_colight_adjacency", None)
    if cache is None:
        order = sorted(env.get_intersections())
        positions = {i: traci.junction.getPosition(i) for i in order}
        num_neighbors = min(COLIGHT_TOP_K_ADJACENCY, len(order))
        cache = build_adjacency_rows(positions, order, num_neighbors)
        env._colight_adjacency = cache
    return cache[intersection_id]


def _advanced_movement_topology(env, intersection_id):
    """Per-atomic-movement entering from-lanes + outgoing edge, cached on env.

    Built once from the controlled links: for each protected-green link, the from_lane
    is an entering lane (its approach comes from env.approach_mapping) and the to_lane is
    on the movement's OUTGOING edge. The 4 phases (ETWT/NTST/ELWL/NLSL) each pair two
    same-type movements, so the movement type is the phase name's 2nd char ('T'/'L').
    """
    import traci  # provided by $SUMO_HOME/tools at runtime (see sumo_env.py)
    cache = getattr(env, "_adv_topology", None)
    if cache is None:
        cache = env._adv_topology = {}
    if intersection_id in cache:
        return cache[intersection_id]

    approach_to_short = {full: short for short, full in location_dict.items()}  # "East"->"E"
    links = traci.trafficlight.getControlledLinks(intersection_id)
    approach_of_lane = env.approach_mapping[intersection_id]
    in_lanes = {m: set() for m in COLIGHT_MOVEMENT_ORDER}
    out_edge = {}
    for phase_name, phase_cfg in env.intersection_config["phases"].items():
        movement_type = phase_name[1]  # 'T' (through) or 'L' (left)
        for i, char in enumerate(phase_cfg["green"]):
            if char != "G":
                continue
            for (from_lane, to_lane, _via) in links[i]:
                approach = approach_of_lane.get(from_lane)
                if approach is None or approach == "Unknown":
                    continue
                movement = approach_to_short[approach] + movement_type
                in_lanes[movement].add(from_lane)
                out_edge.setdefault(movement, traci.lane.getEdgeID(to_lane))

    cache[intersection_id] = {"in_lanes": in_lanes, "out_edge": out_edge}
    return cache[intersection_id]


def _traffic_movement_pressure_queue_efficient(env, intersection_id):
    import traci  # provided by $SUMO_HOME/tools at runtime (see sumo_env.py)
    topo = _advanced_movement_topology(env, intersection_id)
    entering, exiting_halt, exiting_lanes = {}, {}, {}
    for movement in COLIGHT_MOVEMENT_ORDER:
        entering[movement] = sum(
            traci.lane.getLastStepHaltingNumber(lane) for lane in topo["in_lanes"][movement])
        edge = topo["out_edge"].get(movement)
        if edge is None:
            exiting_halt[movement], exiting_lanes[movement] = 0.0, 0
        else:
            exiting_halt[movement] = traci.edge.getLastStepHaltingNumber(edge)
            exiting_lanes[movement] = traci.edge.getLaneNumber(edge)
    return movement_pressure_efficient(entering, exiting_halt, exiting_lanes)


def _lane_enter_running_part(env, intersection_id):
    return movement_segment1_counts(env.get_state(intersection_id))


FEATURE_REGISTRY = {
    "cur_phase": _cur_phase,
    "lane_num_vehicle": _lane_num_vehicle,
    "adjacency_matrix": _adjacency_matrix,
    # Advanced CoLight features
    "traffic_movement_pressure_queue_efficient": _traffic_movement_pressure_queue_efficient,
    "lane_enter_running_part": _lane_enter_running_part,
}


def build_state(env, intersection_id, feature_names):
    """Emit the feature-keyed LIVE state dict for one intersection."""
    return {name: FEATURE_REGISTRY[name](env, intersection_id) for name in feature_names}


# ===========================================================================
# Reward helper (separate from state features)
# ===========================================================================

def intersection_queue_length(env, intersection_id):
    """Stopped-vehicle count on an intersection's controlled (incoming) lanes.

    Uses SUMO's halting count (speed < ~0.1 m/s), matching MetricsRecorder's
    MIN_SPEED predicate and source CoLight's queue_length reward semantics.
    """
    import traci  # provided by $SUMO_HOME/tools at runtime (see sumo_env.py)
    lanes = set(traci.trafficlight.getControlledLanes(intersection_id))
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
