"""Tests for the regime state machines (utils/regime.py) and the Diagnoser
that drives them (utils/diagnoser.py).

    python -m pytest tests/test_regime.py
    python tests/test_regime.py            # without pytest

Synthetic (entries, exits, n_active) sequences and synthetic closed greens,
no SUMO.
"""
import csv
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import EXTENTS_FILENAME, REGIME_FILENAME  # noqa: E402
from utils.diagnoser import EXTENT_COLUMNS, REGIME_COLUMNS, Diagnoser  # noqa: E402
from utils.movement_flags import FlagConfig, MovementFlags  # noqa: E402
from utils.oversaturation_indices import IntervalResult, LaneGreenTracker  # noqa: E402
from utils.regime import CycleEvents, next_extent_regime, next_regime  # noqa: E402

LANE = "in_1"
TARGET = "out_1"
CYCLE_S = 140
CAPACITY = 20


def run_machine(cycles, state="N"):
    """Regime after each cycle of an (entries, exits, n_active) sequence."""
    states = []
    for entries, exits, n_active in cycles:
        state = next_regime(state, entries, exits, n_active)
        states.append(state)
    return states


def test_full_episode_n_l_p_l_p_r_n():
    cycles = [
        (0, 0, 0),  # N
        (1, 0, 1),  # entry -> L
        (0, 0, 1),  # one cycle without entry -> P
        (2, 0, 3),  # entry during processing -> L
        (0, 0, 3),  # -> P
        (0, 0, 3),  # held -> P
        (0, 1, 2),  # exit, no entry -> R
        (0, 0, 2),  # still flagged -> R
        (0, 2, 0),  # set empties -> N
        (0, 0, 0),  # N
    ]
    assert run_machine(cycles) == list("NLPLPPRRNN")


def test_entry_during_recovery_returns_to_loading():
    assert run_machine([(0, 0, 2)], state="R") == ["R"]
    assert run_machine([(1, 0, 3), (0, 0, 3)], state="R") == ["L", "P"]


def test_exit_and_entry_in_same_cycle_gives_loading_not_recovery():
    assert run_machine([(1, 1, 2)], state="P") == ["L"]
    assert run_machine([(1, 1, 2)], state="R") == ["L"]
    assert run_machine([(1, 1, 2)], state="L") == ["L"]


def test_last_exit_empties_set_gives_normal_in_that_cycle():
    assert run_machine([(0, 1, 0)], state="P") == ["N"]
    assert run_machine([(0, 1, 0)], state="L") == ["N"]


def test_loading_moves_to_processing_even_with_set_empty():
    # No exit happened, so there is no recovery to declare; P waits for one.
    assert run_machine([(0, 0, 0)], state="L") == ["P"]


def run_extent_machine(cycles, state="N"):
    """Regime after each cycle of an (entries, exits, n_conns) sequence of an
    extent that already exists."""
    states = []
    for entries, exits, n_conns in cycles:
        state = next_extent_regime(state, entries, exits, n_conns, newborn=False)
        states.append(state)
    return states


def test_extent_newborn_is_loading_whatever_its_counts():
    assert next_extent_regime("N", 0, 0, 1, newborn=True) == "L"
    assert next_extent_regime("P", 2, 1, 3, newborn=True) == "L"


def test_extent_chained_entry_never_returns_to_loading():
    assert run_extent_machine([(1, 0, 2)], state="P") == ["P"]
    assert run_extent_machine([(1, 0, 2)], state="R") == ["P"]
    assert run_extent_machine([(1, 0, 2)], state="L") == ["P"]


def test_extent_exit_without_entry_recovers_and_empty_goes_to_normal():
    assert run_extent_machine([(0, 1, 2), (0, 0, 2), (0, 1, 0)], state="P") == list("RRN")
    assert run_extent_machine([(1, 1, 2)], state="P") == ["P"]


def test_cycle_events_split_tosi_entries_from_sosi_entries_and_record_exit_lanes():
    events = CycleEvents()
    events.add(before={("a", "x"): "T"}, after={("b", "y"): "T", ("c", "z"): "S"})
    assert (events.entries, events.exits) == (2, 1)
    assert events.tosi_entry_lanes == {"b"}
    assert events.exit_lanes == {"a"}


# --- Diagnoser over the real MovementFlags ---------------------------------------

def tracker():
    return LaneGreenTracker("tls", LANE, 150.0, [0], [TARGET], {TARGET: [0]})


def green(allred, oversaturated):
    """A closed green of LANE that either raises the flag or counts as clean."""
    tosi, ratio = (0.5, 0.9) if oversaturated else (0.0, 0.0)
    return IntervalResult(
        tls="tls", lane=LANE, green_start=allred - 33, green_end=allred - 3,
        allred=allred, halted_at_start=frozenset(), on_lane_at_allred=frozenset(),
        tosi=tosi, discharged=0, blocked_flag=False, capacity=CAPACITY,
        halted_count_start=round(ratio * CAPACITY), halted_count_end=0,
        ratio_start=ratio, ratio_end=0.0, blocked_steps={TARGET: 0},
        sosi={TARGET: 0.0}, stop_halted_steps=0, no_exit_steps=0,
        target_halted_steps={TARGET: 0},
    )


def run_diagnoser(greens_by_time, last_t):
    """Step the clock 1..last_t, closing each green at its allred second the
    way OversaturationIndices.update does before Diagnoser.update. Returns the
    regime.csv rows, the regime seen at each step and the extents.csv rows."""
    with tempfile.TemporaryDirectory() as tmp:
        flags = MovementFlags(FlagConfig(), Path(tmp))
        diagnoser = Diagnoser(flags, CYCLE_S, Path(tmp))
        lane_tracker = tracker()
        seen = {}
        for t in range(1, last_t + 1):
            if t in greens_by_time:
                diagnoser.on_interval_closed(lane_tracker, greens_by_time[t])
            diagnoser.update(t)
            seen[t] = diagnoser.regime
        diagnoser.close()
        flags.close()
        with open(Path(tmp) / REGIME_FILENAME, newline="") as f:
            rows = list(csv.DictReader(f))
        with open(Path(tmp) / EXTENTS_FILENAME, newline="") as f:
            extent_rows = list(csv.DictReader(f))
    return rows, seen, extent_rows


def test_diagnoser_reads_flag_transitions_and_logs_one_row_per_cycle():
    # Cycle 1 raises the flag; cycles 2-3 hold it; the 3rd clean green
    # (k_flag = 3) lands in cycle 4 and clears it. Cycle 5 is partial.
    greens = {
        170: green(170, oversaturated=True),
        310: green(310, oversaturated=True),
        450: green(450, oversaturated=False),
        500: green(500, oversaturated=False),
        600: green(600, oversaturated=False),
    }
    rows, seen, _ = run_diagnoser(greens, last_t=750)
    assert list(rows[0]) == REGIME_COLUMNS
    assert [r["cycle_start"] for r in rows] == ["0", "140", "280", "420", "560", "700"]
    assert [(r["entries"], r["exits"], r["n_active"]) for r in rows] == [
        ("0", "0", "0"), ("1", "0", "1"), ("0", "0", "1"),
        ("0", "0", "1"), ("0", "1", "0"), ("0", "0", "0")]
    assert [r["regime"] for r in rows] == list("NLPPNN")


def test_diagnoser_green_at_cycle_boundary_belongs_to_next_cycle():
    rows, _, _ = run_diagnoser({140: green(140, oversaturated=True)}, last_t=279)
    assert [(r["cycle_start"], r["entries"]) for r in rows] == [("0", "0"), ("140", "1")]


def test_diagnoser_logs_extents_from_carried_forward_flags():
    # The lane has no green closing in cycles 280-699; its flag still counts.
    greens = {
        170: green(170, oversaturated=True),
        710: green(710, oversaturated=False),
        760: green(760, oversaturated=False),
        810: green(810, oversaturated=False),
    }
    _, _, extent_rows = run_diagnoser(greens, last_t=900)
    assert list(extent_rows[0]) == EXTENT_COLUMNS
    assert [(r["cycle_start"], r["extent_id"], r["members"], r["n_lanes"], r["n_conns"],
             r["E"], r["X"], r["regime"], r["newborn"], r["vanished"])
            for r in extent_rows] == [
        ("140", "1", LANE, "1", "1", "1", "0", "L", "1", "0"),
        ("280", "1", LANE, "1", "1", "0", "0", "P", "0", "0"),
        ("420", "1", LANE, "1", "1", "0", "0", "P", "0", "0"),
        ("560", "1", LANE, "1", "1", "0", "0", "P", "0", "0"),
        ("700", "1", "", "0", "0", "0", "1", "N", "0", "1"),
    ]


def test_decisions_see_regime_of_last_closed_cycle():
    _, seen, _ = run_diagnoser({170: green(170, oversaturated=True)}, last_t=300)
    assert seen[278] == "N"   # cycle 140-279 not yet closed
    assert seen[279] == "L"   # closed on its last second
    assert seen[300] == "L"


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
