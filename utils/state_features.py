"""The sim<->model translator for RL controllers.

A feature REGISTRY maps a feature name to a function ``fn(env, intersection_id)``
that reads the SUMO environment and returns that feature's value. A model declares
the features it wants by name (mirroring CoLight's ``LIST_STATE_FEATURE``); the runner
emits a feature-KEYED dict per intersection, so adding features later (pressure,
time_this_phase, ...) is purely additive.

Only the three features CoLight needs are implemented now:
  - "cur_phase"        : [logical phase index]  (agent expands to 8-bit via PHASE)
  - "lane_num_vehicle" : per-movement vehicle counts in canonical movement order
  - "adjacency_matrix" : top-k neighbor agent indices (self first)

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
    print(f"rows: {rows}")
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


FEATURE_REGISTRY = {
    "cur_phase": _cur_phase,
    "lane_num_vehicle": _lane_num_vehicle,
    "adjacency_matrix": _adjacency_matrix,
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
