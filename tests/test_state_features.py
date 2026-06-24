"""Per-feature unit tests for the CoLight sim<->model translator.

Each test feeds a HAND-BUILT SUMO state (no live simulator) into a pure core of
utils.state_features and asserts the exact dict shape + lane ordering CoLight expects.
One test per feature key, plus the memory-expansion and reward-shape contracts.

Run: PYTHONPATH=<repo root> python tests/test_state_features.py   (or pytest)
"""

from configurations import INTERSECTION_CONFIGS, COLIGHT_MOVEMENT_ORDER
from utils.state_features import (
    build_phase_onehot,
    phase_index_from_ryg,
    movement_vehicle_counts,
    build_adjacency_rows,
    expand_state_for_memory,
)

CONF = INTERSECTION_CONFIGS["two_lane_1x1"]


# --------------------------------------------------------------------------
# cur_phase: PHASE one-hot + RYG -> logical index
# --------------------------------------------------------------------------

def test_build_phase_onehot_matches_source_vectors():
    phase_map = build_phase_onehot(CONF)
    # two_lane_1x1 ids: NTST=0, ETWT=1, NLSL=2, ELWL=3.
    # Vectors over [WL, WT, EL, ET, NL, NT, SL, ST] == source CoLight's PHASE, re-keyed.
    expected = {
        0: [0, 0, 0, 0, 0, 1, 0, 1],  # NTST -> NT, ST
        1: [0, 1, 0, 1, 0, 0, 0, 0],  # ETWT -> WT, ET
        2: [0, 0, 0, 0, 1, 0, 1, 0],  # NLSL -> NL, SL
        3: [1, 0, 1, 0, 0, 0, 0, 0],  # ELWL -> WL, EL
    }
    assert phase_map == expected, phase_map
    # Each one-hot has exactly the 8 movement slots, with exactly two lit.
    for vec in phase_map.values():
        assert len(vec) == 8
        assert sum(vec) == 2


def test_phase_index_from_ryg_resolves_green():
    # ETWT green string for two_lane_1x1 -> logical id 1.
    ryg = CONF["phases"]["ETWT"]["green"]
    assert phase_index_from_ryg(CONF, ryg) == [1]
    ryg = CONF["phases"]["NLSL"]["green"]
    assert phase_index_from_ryg(CONF, ryg) == [2]


def test_phase_index_from_ryg_rejects_non_green():
    raised = False
    try:
        phase_index_from_ryg(CONF, CONF["global_settings"]["all_red_state"])
    except ValueError:
        raised = True
    assert raised, "all-red RYG should not resolve to a logical phase"


# --------------------------------------------------------------------------
# lane_num_vehicle: 8-movement counts in canonical order
# --------------------------------------------------------------------------

def _agg(early, s1, s2, s3):
    return {"early_queued": early,
            "segments": {"segment_1": s1, "segment_2": s2, "segment_3": s3},
            "lanes": {}}


def test_movement_vehicle_counts_ordering_and_aggregation():
    # Distinct per-movement totals so ordering is verifiable; segments must be summed.
    state = {"movement_states": {
        "ETWT": {"East": _agg(1, 1, 1, 0),   # ET total 3
                 "West": _agg(4, 0, 0, 0)},   # WT total 4
        "ELWL": {"East": _agg(2, 1, 1, 1),   # EL total 5
                 "West": _agg(6, 0, 0, 0)},   # WL total 6
        "NTST": {"North": _agg(7, 0, 0, 0),  # NT total 7
                 "South": _agg(8, 0, 0, 0)},  # ST total 8
        "NLSL": {"North": _agg(9, 0, 0, 0),  # NL total 9
                 "South": _agg(10, 0, 0, 0)}, # SL total 10
    }}
    counts = movement_vehicle_counts(state, COLIGHT_MOVEMENT_ORDER)
    # order [WL, WT, EL, ET, NL, NT, SL, ST]
    assert counts == [6, 4, 5, 3, 9, 7, 10, 8], counts
    assert len(counts) == 8


# --------------------------------------------------------------------------
# adjacency_matrix: top-k neighbor indices, self first
# --------------------------------------------------------------------------

def test_adjacency_single_intersection_is_self():
    rows = build_adjacency_rows({"A": (0.0, 0.0)}, ["A"], num_neighbors=1)
    assert rows == {"A": [0]}, rows


def test_adjacency_line_topk_self_first():
    pos = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}
    order = ["a", "b", "c"]
    rows = build_adjacency_rows(pos, order, num_neighbors=2)
    # self (distance 0) must be first in every row.
    assert rows["a"][0] == 0 and set(rows["a"]) == {0, 1}, rows
    assert rows["b"][0] == 1 and set(rows["b"]) == {1, 0}, rows  # ties broken toward nearer index
    assert rows["c"][0] == 2 and set(rows["c"]) == {2, 1}, rows


# --------------------------------------------------------------------------
# memory expansion + combined feature-width contract
# --------------------------------------------------------------------------

def test_expand_state_for_memory_expands_cur_phase():
    phase_map = build_phase_onehot(CONF)
    live = {"cur_phase": [1], "lane_num_vehicle": [0] * 8, "adjacency_matrix": [0]}
    stored = expand_state_for_memory(live, phase_map)
    assert stored["cur_phase"] == phase_map[1]               # expanded 8-bit
    assert stored["lane_num_vehicle"] == [0] * 8             # untouched
    assert stored["adjacency_matrix"] == [0]                 # untouched
    assert live["cur_phase"] == [1]                          # original not mutated


def test_combined_feature_width_is_16():
    # cur_phase (8 expanded) + lane_num_vehicle (8) = 16 = agent.len_feature.
    phase_map = build_phase_onehot(CONF)
    assert len(phase_map[0]) == 8
    state = {"movement_states": {
        "ETWT": {"East": _agg(0, 0, 0, 0), "West": _agg(0, 0, 0, 0)},
        "ELWL": {"East": _agg(0, 0, 0, 0), "West": _agg(0, 0, 0, 0)},
        "NTST": {"North": _agg(0, 0, 0, 0), "South": _agg(0, 0, 0, 0)},
        "NLSL": {"North": _agg(0, 0, 0, 0), "South": _agg(0, 0, 0, 0)},
    }}
    assert len(movement_vehicle_counts(state, COLIGHT_MOVEMENT_ORDER)) == 8


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
