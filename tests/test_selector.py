"""Tests for the A2 rule-table selector (utils/selector.py).

    uv run --with pytest python -m pytest tests/test_selector.py
    python tests/test_selector.py            # without pytest

Synthetic green records, flags and extents; no SUMO. One intersection with
hzreal lane roles: index 0 right turn (ignored), 1 through, 2 left.
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.selector import (  # noqa: E402
    ADD, AT_FLOOR, DG_ZERO, EXPIRE, HOLD, NEW_COMMAND, OUT_OF_SCOPE, ROUTE_DEFERRED,
    SOSI_SYMPTOM, RECEIVER_BOUND_DONOR_POOL, RECEIVER_BOUND_MAX_GREEN, RECEIVER_BOUND_NONE,
    ExtentView, GreenRecord, IntersectionLanes, Plan, RuleTableSelector,
    assess_donor, request_dg,
)

TLS = "tls"
PHASE_LANES = {
    "ETWT": ("E_1", "W_1"), "NTST": ("N_1", "S_1"),
    "ELWL": ("E_2", "W_2"), "NLSL": ("N_2", "S_2"),
}
ORDER = ("ETWT", "NTST", "ELWL", "NLSL")
GREEN_S = 30


def plan(**overrides):
    splits = {phase: GREEN_S for phase in ORDER}
    splits.update(overrides)
    return Plan(order=ORDER, splits=splits, offset=0)


def record(lane, green_start, tosi=0.0, held=0, entries=0, green_s=GREEN_S):
    return GreenRecord(tls=TLS, lane=lane, green_start=green_start,
                       green_end=green_start + green_s, tosi=tosi, held_over=held,
                       entered_red=entries, entered_green=0, exited_green=0)


def extent(extent_id=1, members=("N_1",), regime="L", cycle_start=0, intersections=(TLS,)):
    return ExtentView(extent_id=extent_id, members=frozenset(members),
                      intersections=frozenset(intersections), regime=regime,
                      cycle_start=cycle_start)


def selector():
    return RuleTableSelector([IntersectionLanes(TLS, PHASE_LANES)])


def feed_donors(sel, worst_lane_v, green_start=0):
    """worst_lane_v: {phase: entries per green on its worst lane}; the other
    lane of each phase sees nothing."""
    for phase, v in worst_lane_v.items():
        first, second = PHASE_LANES[phase]
        sel.on_green_closed(record(first, green_start, entries=v))
        sel.on_green_closed(record(second, green_start, entries=0))


T1_DONORS = {"ETWT": 11, "ELWL": 6, "NLSL": 2}


def issue_t1_command(sel, current_plan=None, tosi=0.5, t=100):
    current_plan = current_plan or plan()
    feed_donors(sel, T1_DONORS)
    sel.on_green_closed(record("N_1", 50, tosi=tosi, held=10))
    sel.on_green_closed(record("S_1", 50, tosi=0.0, held=0))
    return sel.evaluate(t, TLS, current_plan, [extent()], {"N_1": "T"})


def feed_receiver_window(sel, helds, first_start=200, step=100):
    for index, held in enumerate(helds):
        start = first_start + index * step
        sel.on_green_closed(record("N_1", start, tosi=0.5, held=held))
        sel.on_green_closed(record("S_1", start, tosi=0.0, held=0))


def test_t1_worked_example():
    sel = selector()
    new_plan, rows = issue_t1_command(sel)
    assert new_plan.splits == {"ETWT": 30, "NTST": 45, "ELWL": 30, "NLSL": 15}
    assert new_plan.greens_sum == 120
    assert new_plan.order == ORDER and new_plan.offset == 0
    row = rows[0]
    assert row["outcome"] == NEW_COMMAND
    assert row["phase"] == "NTST" and row["extent_id"] == 1 and row["regime"] == "L"
    assert (row["dg_requested"], row["dg_given"], row["short_by"]) == (15, 15, 0)
    assert row["donors"] == "NLSL:15;ELWL:0;ETWT:0"
    assert row["donor_floor_bound"] == "NLSL:min_green;ELWL:vc;ETWT:vc"
    assert row["receiver_bound"] == RECEIVER_BOUND_NONE
    assert row["plan_before"] == "ETWT:30;NTST:30;ELWL:30;NLSL:30"
    assert row["plan_after"] == "ETWT:30;NTST:45;ELWL:30;NLSL:15"


def test_t2_second_donor_in_ascending_vc_order():
    sel = selector()
    new_plan, rows = issue_t1_command(sel, current_plan=plan(NLSL=15))
    # NLSL: floor 5, spare 10, v/c 0.286; ELWL: spare 17, v/c 0.429; ETWT: v/c 0.786
    assert new_plan.splits == {"ETWT": 30, "NTST": 45, "ELWL": 25, "NLSL": 5}
    assert rows[0]["donors"] == "NLSL:10;ELWL:5;ETWT:0"
    assert rows[0]["donor_vc_before"] == "NLSL:0.286;ELWL:0.429;ETWT:0.786"


def test_t3_rounding():
    history = {"E_2": [record("E_2", 0, entries=6)]}
    donor = assess_donor("ELWL", ("E_2", "W_2"), history, GREEN_S)
    assert donor.floor == 13 and isinstance(donor.floor, int)
    assert donor.spare == 17 and isinstance(donor.spare, int)
    assert request_dg([record("N_1", 0, tosi=0.45)], GREEN_S) == 13


def test_t4_pool_short():
    sel = selector()
    feed_donors(sel, {"ETWT": 11})
    sel.on_green_closed(record("N_1", 50, tosi=0.5))
    new_plan, rows = sel.evaluate(100, TLS, plan(), [extent()], {"N_1": "T"})
    assert new_plan.splits == {"ETWT": 24, "NTST": 36, "ELWL": 30, "NLSL": 30}
    row = rows[0]
    assert (row["dg_requested"], row["dg_given"], row["short_by"]) == (15, 6, 9)
    assert row["receiver_bound"] == RECEIVER_BOUND_DONOR_POOL
    assert row["donors"] == "ETWT:6"


def test_t5_two_tosi_phases_out_of_scope():
    sel = selector()
    feed_donors(sel, T1_DONORS)
    new_plan, rows = sel.evaluate(100, TLS, plan(), [extent(members=("N_1", "E_1"))],
                                  {"N_1": "T", "E_1": "T"})
    assert new_plan is None
    assert rows[0]["outcome"] == OUT_OF_SCOPE
    assert rows[0]["plan_after"] == rows[0]["plan_before"]


def test_t6_sosi_only_flag_no_action():
    sel = selector()
    feed_donors(sel, T1_DONORS)
    new_plan, rows = sel.evaluate(100, TLS, plan(), [extent()], {"N_1": "S"})
    assert new_plan is None
    assert rows[0]["outcome"] == SOSI_SYMPTOM


def test_t7_right_turn_flag_ignored():
    sel = selector()
    feed_donors(sel, T1_DONORS)
    sel.on_green_closed(record("N_0", 50, tosi=0.9, held=10))
    new_plan, rows = sel.evaluate(100, TLS, plan(), [extent(members=("N_0",))],
                                  {"N_0": "T"})
    assert new_plan is None and rows == []


def test_t8_route_extent_deferred():
    sel = selector()
    feed_donors(sel, T1_DONORS)
    route = extent(members=("N_1", "far_1"), intersections=(TLS, "other"))
    new_plan, rows = sel.evaluate(100, TLS, plan(), [route], {"N_1": "T", "far_1": "T"})
    assert new_plan is None
    assert rows[0]["outcome"] == ROUTE_DEFERRED
    assert rows[0]["plan_after"] == rows[0]["plan_before"]


def test_t9_decision_point_hold_then_add():
    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    feed_receiver_window(sel, [20, 18, 16])
    new_plan, rows = sel.evaluate(450, TLS, plan_1, [extent(regime="P")], {"N_1": "T"})
    assert new_plan is None
    row = rows[0]
    assert row["outcome"] == HOLD
    assert (row["held_first"], row["held_last"]) == (20, 16)
    assert (row["window_first_t"], row["window_last_t"]) == (200, 400)

    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    feed_receiver_window(sel, [20, 20, 20])
    new_plan, rows = sel.evaluate(450, TLS, plan_1, [extent(regime="P")], {"N_1": "T"})
    # From the current plan: dg = floor(0.5 * 45) = 22, capped to 60 - 45 = 15;
    # NLSL (15 s, floor 5) gives 10, then ELWL gives 5.
    assert rows[0]["outcome"] == ADD
    assert new_plan.splits == {"ETWT": 30, "NTST": 60, "ELWL": 25, "NLSL": 5}
    assert new_plan.greens_sum == 120
    assert rows[0]["dg_requested"] == 22 and rows[0]["dg_given"] == 15
    assert rows[0]["receiver_bound"] == RECEIVER_BOUND_MAX_GREEN


def test_t10_expire_wins_over_add_and_restores_base_plan():
    sel = selector()
    base = plan()
    plan_1, _ = issue_t1_command(sel, current_plan=base)
    feed_receiver_window(sel, [20, 22, 24])
    new_plan, rows = sel.evaluate(450, TLS, plan_1, [], {"N_1": "T"})
    assert rows[0]["outcome"] == EXPIRE
    assert new_plan == base
    assert rows[0]["plan_after"] == base.text


def test_t11_recovery_blocks_add():
    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    feed_receiver_window(sel, [20, 22, 24])
    new_plan, rows = sel.evaluate(450, TLS, plan_1, [extent(regime="R")], {"N_1": "T"})
    assert new_plan is None
    assert rows[0]["outcome"] == HOLD
    assert rows[0]["add_blocked_recovery"] == 1


def test_t12_no_new_command_in_recovery_or_normal():
    for regime in ("R", "N"):
        sel = selector()
        feed_donors(sel, T1_DONORS)
        sel.on_green_closed(record("N_1", 50, tosi=0.5))
        new_plan, rows = sel.evaluate(100, TLS, plan(), [extent(regime=regime)], {"N_1": "T"})
        assert new_plan is None and rows == []


def test_t13_receiver_capped_at_max_green():
    sel = selector()
    new_plan, rows = issue_t1_command(sel, current_plan=plan(NTST=50), tosi=0.4)
    assert new_plan.splits["NTST"] == 60
    assert new_plan.splits["NLSL"] == 20
    assert rows[0]["dg_requested"] == 20 and rows[0]["dg_given"] == 10
    assert rows[0]["receiver_bound"] == RECEIVER_BOUND_MAX_GREEN


def test_t14_donor_over_capacity_greens_counted():
    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    # c = (30 + 3 - 3) / 2.143 = 14.0 vehicles: 15 entries is over, 10 is not.
    sel.on_green_closed(record("E_1", 150, entries=15))
    sel.on_green_closed(record("E_1", 250, entries=10))
    sel.on_green_closed(record("W_2", 250, entries=10))
    feed_receiver_window(sel, [20, 18, 16])
    _, rows = sel.evaluate(450, TLS, plan_1, [extent(regime="P")], {"N_1": "T"})
    assert rows[0]["donor_over_capacity_greens"] == 1


def test_t14b_donor_at_floor_when_nothing_can_be_given():
    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    feed_receiver_window(sel, [20, 20, 20])
    # Means over the last 3 greens: ETWT 15 -> 33 s floor, ELWL 14 -> 30 s,
    # NLSL 6.67 -> 15 s: every donor's spare is 0.
    feed_donors(sel, {"ETWT": 17, "ELWL": 18, "NLSL": 9}, green_start=300)
    feed_donors(sel, {"ETWT": 17, "ELWL": 18, "NLSL": 9}, green_start=400)
    new_plan, rows = sel.evaluate(450, TLS, plan_1, [extent(regime="P")], {"N_1": "T"})
    assert new_plan is None
    assert rows[0]["outcome"] == AT_FLOOR


def test_t15_reentry_after_expiry():
    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    feed_receiver_window(sel, [20, 22, 24])
    _, rows = sel.evaluate(450, TLS, plan_1, [], {"N_1": "T"})
    expiry_row = rows[0]
    assert expiry_row["reentry_after_expiry"] == 0
    feed_receiver_window(sel, [5, 9], first_start=500)
    sel.evaluate(650, TLS, plan(), [], {})
    assert expiry_row["reentry_after_expiry"] == 0
    sel.evaluate(700, TLS, plan(), [extent(extent_id=2, regime="L", cycle_start=560)],
                 {"N_1": "T"})
    assert expiry_row["reentry_after_expiry"] == 1


def test_t15b_no_reentry_after_window_passes():
    sel = selector()
    plan_1, _ = issue_t1_command(sel)
    feed_receiver_window(sel, [20, 22, 24])
    _, rows = sel.evaluate(450, TLS, plan_1, [], {"N_1": "T"})
    expiry_row = rows[0]
    feed_receiver_window(sel, [5, 5, 5, 5], first_start=500)
    sel.evaluate(900, TLS, plan(), [extent(extent_id=2, cycle_start=840)], {"N_1": "T"})
    assert expiry_row["reentry_after_expiry"] == 0


def test_dg_zero_when_tosi_is_zero():
    sel = selector()
    feed_donors(sel, T1_DONORS)
    sel.on_green_closed(record("N_1", 50, tosi=0.0))
    new_plan, rows = sel.evaluate(100, TLS, plan(), [extent()], {"N_1": "T"})
    assert new_plan is None and rows[0]["outcome"] == DG_ZERO


def test_no_action_rows_logged_once_per_extent_cycle():
    sel = selector()
    feed_donors(sel, T1_DONORS)
    _, first = sel.evaluate(100, TLS, plan(), [extent()], {"N_1": "S"})
    _, second = sel.evaluate(101, TLS, plan(), [extent()], {"N_1": "S"})
    _, third = sel.evaluate(140, TLS, plan(), [extent(cycle_start=140)], {"N_1": "S"})
    assert len(first) == 1 and second == [] and len(third) == 1


if __name__ == "__main__":
    failed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
