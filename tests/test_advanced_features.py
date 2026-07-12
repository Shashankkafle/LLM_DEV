"""Per-feature unit tests for the Advanced CoLight translator cores.

Each feeds a hand-built SUMO state / queue dicts into a pure core of
utils.state_features and asserts the exact vector shape + movement ordering the
Advanced CoLight agent expects (same CoLightAgent, different features).

Run: PYTHONPATH=<repo root> python tests/test_advanced_features.py   (or pytest)
"""

from configurations import (
    INTERSECTION_CONFIGS,
    COLIGHT_MOVEMENT_ORDER,
    ADVANCED_COLIGHT_FEATURES,
)
from utils.state_features import (
    build_phase_onehot,
    movement_pressure_efficient,
    movement_segment1_counts,
)

CONF = INTERSECTION_CONFIGS["two_lane_1x1"]
# order: [WL, WT, EL, ET, NL, NT, SL, ST]


# --------------------------------------------------------------------------
# traffic_movement_pressure_queue_efficient
# --------------------------------------------------------------------------

def test_efficient_pressure_normalizes_by_outgoing_lanes():
    entering = {"WL": 6, "WT": 4, "EL": 5, "ET": 3, "NL": 9, "NT": 7, "SL": 10, "ST": 8}
    exiting_halt = {m: 6 for m in COLIGHT_MOVEMENT_ORDER}     # 6 stopped on each outgoing edge
    exiting_lanes = {m: 3 for m in COLIGHT_MOVEMENT_ORDER}    # 3 lanes -> exit_norm = 2
    out = movement_pressure_efficient(entering, exiting_halt, exiting_lanes, COLIGHT_MOVEMENT_ORDER)
    # pressure[m] = entering[m] - 6/3 = entering[m] - 2, in movement order
    assert out == [4, 2, 3, 1, 7, 5, 8, 6], out
    assert len(out) == 8


def test_efficient_pressure_guards_zero_lanes_and_missing():
    entering = {"WL": 5}
    exiting_halt = {"WL": 9, "WT": 9}
    exiting_lanes = {"WL": 0, "WT": 2}  # WL: zero-lane guard -> exit_norm 0; others missing -> 0
    out = movement_pressure_efficient(entering, exiting_halt, exiting_lanes, COLIGHT_MOVEMENT_ORDER)
    # WL (idx0): 5 - 0 = 5 ; WT (idx1): 0 - 9/2 = -4.5 ; rest: 0 - 0 = 0
    assert out[0] == 5, out
    assert out[1] == -4.5, out
    assert all(v == 0 for v in out[2:]), out


# --------------------------------------------------------------------------
# lane_enter_running_part  (near-stopline running = get_state segment_1)
# --------------------------------------------------------------------------

def _agg(early, s1, s2, s3):
    return {"early_queued": early,
            "segments": {"segment_1": s1, "segment_2": s2, "segment_3": s3},
            "lanes": {}}


def test_segment1_counts_picks_nearest_segment_only():
    # segment_1 values are the distinct ones; early_queued / seg2 / seg3 are decoys.
    state = {"movement_states": {
        "ETWT": {"East": _agg(99, 3, 1, 1),    # ET seg1 = 3
                 "West": _agg(99, 4, 2, 2)},    # WT seg1 = 4
        "ELWL": {"East": _agg(99, 5, 0, 0),    # EL seg1 = 5
                 "West": _agg(99, 6, 0, 0)},    # WL seg1 = 6
        "NTST": {"North": _agg(99, 7, 9, 9),   # NT seg1 = 7
                 "South": _agg(99, 8, 9, 9)},   # ST seg1 = 8
        "NLSL": {"North": _agg(99, 9, 0, 0),   # NL seg1 = 9
                 "South": _agg(99, 10, 0, 0)},  # SL seg1 = 10
    }}
    out = movement_segment1_counts(state, COLIGHT_MOVEMENT_ORDER)
    # order [WL, WT, EL, ET, NL, NT, SL, ST]
    assert out == [6, 4, 5, 3, 9, 7, 10, 8], out
    assert len(out) == 8


# --------------------------------------------------------------------------
# combined feature width: cur_phase(8) + pressure(8) + running(8) = 24
# --------------------------------------------------------------------------

def test_advanced_feature_list_and_width():
    assert ADVANCED_COLIGHT_FEATURES[-1] == "adjacency_matrix"  # must stay last
    assert ADVANCED_COLIGHT_FEATURES[:3] == [
        "cur_phase", "traffic_movement_pressure_queue_efficient", "lane_enter_running_part"]
    phase_map = build_phase_onehot(CONF)
    # cur_phase expands to 8; pressure 8; running 8 -> 24 (8 + 8 + 8).
    assert len(phase_map[0]) == 8


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
