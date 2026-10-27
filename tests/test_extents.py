"""Tests for the extent builder (utils/extents.py).

    python -m pytest tests/test_extents.py
    python tests/test_extents.py            # without pytest

Synthetic flag sets per cycle, no SUMO.
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.extents import ExtentTracker, connected_components  # noqa: E402
from utils.regime import CycleEvents  # noqa: E402

CYCLE_S = 140

# Two routes through intersection_2_3: north-south down column 2, west-east
# along row 3. road_2_4_3_1 also has a (flagged) right turn onto the WE route.
NS_UPSTREAM = "road_2_5_3_1"
NS_MIDDLE = "road_2_4_3_1"
NS_DOWNSTREAM = "road_2_3_3_1"
WE_UPSTREAM = "road_1_3_0_1"
WE_DOWNSTREAM = "road_2_3_0_1"
WE_DOWNSTREAM_RIGHT = "road_2_3_0_0"
EXIT_LANE = "road_2_2_3_1"


def events(entries=(), exits=()):
    return CycleEvents(tosi_entry_lanes=set(entries), exit_lanes=set(exits))


def run_cycles(tracker, cycles):
    """cycles: list of (tosi_movements, CycleEvents). Returns rows per cycle."""
    return [tracker.close_cycle(index * CYCLE_S, movements, cycle_events)
            for index, (movements, cycle_events) in enumerate(cycles)]


# --- components ----------------------------------------------------------------

def test_two_link_chain_forms_one_component():
    movements = {(NS_UPSTREAM, NS_MIDDLE), (NS_MIDDLE, NS_DOWNSTREAM),
                 (NS_DOWNSTREAM, EXIT_LANE)}
    assert connected_components(movements) == [
        frozenset({NS_UPSTREAM, NS_MIDDLE, NS_DOWNSTREAM})]


def test_unflagged_target_does_not_link():
    movements = {(NS_UPSTREAM, "road_2_4_3_0"), (NS_MIDDLE, NS_DOWNSTREAM)}
    assert connected_components(movements) == [
        frozenset({NS_MIDDLE}), frozenset({NS_UPSTREAM})]


def test_two_chains_through_one_intersection_stay_two_extents():
    movements = {(NS_MIDDLE, NS_DOWNSTREAM), (NS_MIDDLE, WE_DOWNSTREAM_RIGHT),
                 (NS_DOWNSTREAM, EXIT_LANE),
                 (WE_UPSTREAM, WE_DOWNSTREAM), (WE_DOWNSTREAM, "road_3_3_0_1")}
    tracker = ExtentTracker()
    [rows] = run_cycles(tracker, [(movements, events(entries=[NS_MIDDLE, WE_UPSTREAM]))])
    assert [(row.extent_id, row.members, row.n_conns) for row in rows] == [
        (1, frozenset({WE_UPSTREAM, WE_DOWNSTREAM}), 2),
        (2, frozenset({NS_MIDDLE, NS_DOWNSTREAM}), 3)]


# --- identity -------------------------------------------------------------------

def test_extent_keeps_its_id_while_it_grows():
    tracker = ExtentTracker()
    rows = run_cycles(tracker, [
        ({(NS_MIDDLE, NS_DOWNSTREAM)}, events(entries=[NS_MIDDLE])),
        ({(NS_MIDDLE, NS_DOWNSTREAM), (NS_DOWNSTREAM, EXIT_LANE)},
         events(entries=[NS_DOWNSTREAM])),
    ])
    assert [(r.extent_id, r.newborn, r.entries) for r in rows[0]] == [(1, True, 1)]
    assert [(r.extent_id, r.newborn, r.entries, len(r.members)) for r in rows[1]] == \
        [(1, False, 1, 2)]


def test_split_keeps_id_on_oldest_member_piece_and_other_piece_is_newborn():
    chain = {(NS_UPSTREAM, NS_MIDDLE), (NS_MIDDLE, NS_DOWNSTREAM),
             (NS_DOWNSTREAM, EXIT_LANE)}
    tracker = ExtentTracker()
    rows = run_cycles(tracker, [
        ({(NS_DOWNSTREAM, EXIT_LANE)}, events(entries=[NS_DOWNSTREAM])),
        (chain, events(entries=[NS_UPSTREAM, NS_MIDDLE])),
        (chain - {(NS_MIDDLE, NS_DOWNSTREAM)}, events(exits=[NS_MIDDLE])),
    ])
    assert [(r.extent_id, r.members, r.newborn) for r in rows[2]] == [
        (1, frozenset({NS_DOWNSTREAM}), False),
        (2, frozenset({NS_UPSTREAM}), True)]
    assert rows[2][0].exits == 1


def test_oldest_member_tie_is_broken_by_lane_id():
    tracker = ExtentTracker()
    chain = {(NS_UPSTREAM, NS_MIDDLE), (NS_MIDDLE, EXIT_LANE)}
    rows = run_cycles(tracker, [
        (chain, events(entries=[NS_UPSTREAM, NS_MIDDLE])),
        ({(NS_UPSTREAM, "road_2_4_3_0"), (NS_MIDDLE, EXIT_LANE)}, events()),
    ])
    # Same first cycle; road_2_4_3_1 < road_2_5_3_1, so the NS_MIDDLE piece inherits.
    assert [(r.extent_id, r.members, r.newborn) for r in rows[1]] == [
        (1, frozenset({NS_MIDDLE}), False), (2, frozenset({NS_UPSTREAM}), True)]


def test_extent_with_no_component_vanishes():
    tracker = ExtentTracker()
    rows = run_cycles(tracker, [
        ({(NS_MIDDLE, NS_DOWNSTREAM)}, events(entries=[NS_MIDDLE])),
        (set(), events(exits=[NS_MIDDLE])),
    ])
    [vanished] = rows[1]
    assert (vanished.extent_id, vanished.members, vanished.exits, vanished.vanished) == \
        (1, frozenset(), 1, True)


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            test()
            print(f"[ok  ] {name}")
        except Exception:
            failures += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)
