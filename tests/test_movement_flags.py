"""Tests for the per-movement oversaturation flags (utils/movement_flags.py).

    python -m pytest tests/test_movement_flags.py
    python tests/test_movement_flags.py            # without pytest

All synthetic IntervalResult sequences, no SUMO. The SUMO smoke check that
flags leave control untouched lives in test_oversaturation_indices.py
(test_smoke_indices_do_not_change_control).
"""
import argparse
import csv
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import MOVEMENT_FLAGS_FILENAME  # noqa: E402
from runner_common import add_movement_flag_args, movement_flag_config_from_args  # noqa: E402
from utils.movement_flags import FLAG_COLUMNS, FlagConfig, MovementFlags  # noqa: E402
from utils.oversaturation_indices import IntervalResult, LaneGreenTracker  # noqa: E402

LANE = "in_1"
TARGET = "out_1"
OTHER_TARGET = "out_2"
GREEN_S = 30
CYCLE_S = 60
CAPACITY = 20
# SOSI that loses exactly L = 5 s of a 30 s green.
SOSI_AT_L = 5 / GREEN_S


def new_tracker():
    return LaneGreenTracker("tls", LANE, 150.0, [0, 1, 2], [TARGET, OTHER_TARGET],
                            {TARGET: [0, 1], OTHER_TARGET: [2]})


def interval(index, tosi=0.0, sosi=0.0, ratio=0.0, blocked=False, other_sosi=0.0):
    """The index-th closed green of LANE; ratio_start is set directly."""
    green_start = index * CYCLE_S
    return IntervalResult(
        tls="tls", lane=LANE, green_start=green_start,
        green_end=green_start + GREEN_S, allred=green_start + GREEN_S + 3,
        halted_at_start=frozenset(), on_lane_at_allred=frozenset(),
        tosi=tosi, discharged=0, blocked_flag=blocked, capacity=CAPACITY,
        halted_count_start=None if ratio is None else round(ratio * CAPACITY),
        halted_count_end=0,
        ratio_start=ratio, ratio_end=0.0,
        blocked_steps={TARGET: 0, OTHER_TARGET: 0},
        sosi={TARGET: sosi, OTHER_TARGET: other_sosi},
        stop_halted_steps=0, no_exit_steps=0,
        target_halted_steps={TARGET: 0, OTHER_TARGET: 0},
    )


@contextmanager
def open_flags(config=None):
    with tempfile.TemporaryDirectory() as tmp:
        flags = MovementFlags(config or FlagConfig(), Path(tmp))
        try:
            yield flags
        finally:
            flags.close()


def feed(flags, greens):
    tracker = new_tracker()
    for result in greens:
        flags.on_interval_closed(tracker, result)


def state(flags, target=TARGET):
    return flags.flags[(LANE, target)]


def oversaturated(index):
    return interval(index, tosi=0.2, ratio=0.6)


def clean(index):
    return interval(index, ratio=0.1)


# --- entry ----------------------------------------------------------------------

def test_tosi_branch_enters_at_theta_in():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.1, ratio=0.30)])
        assert state(flags).active
        assert state(flags).branch == "T"
        assert state(flags).raised_at == interval(0).allred
        assert state(flags).entry_count == 1
        assert (LANE, TARGET) in flags.active_flags()


def test_tosi_below_theta_in_does_not_enter():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.5, ratio=0.29)])
        assert not state(flags).active
        assert state(flags).entry_count == 0
        assert flags.active_flags() == set()


def test_high_ratio_without_tosi_does_not_enter():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.0, ratio=0.9)])
        assert not state(flags).active


def test_sosi_branch_enters_at_l_seconds_ungated_by_ratio():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.0, sosi=SOSI_AT_L, ratio=0.0)])
        assert state(flags).active
        assert state(flags).branch == "S"


def test_sosi_below_l_seconds_does_not_enter():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.0, sosi=4.9 / GREEN_S, ratio=0.9)])
        assert not state(flags).active


def test_both_branches_in_one_green_enter_as_tosi():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.2, sosi=0.5, ratio=0.6)])
        assert state(flags).branch == "T"


def test_sosi_flag_becomes_tosi_on_a_later_tosi_green():
    with open_flags() as flags:
        feed(flags, [interval(0, sosi=0.5), interval(1, sosi=0.5)])
        assert state(flags).branch == "S"
        feed(flags, [oversaturated(2)])
        assert state(flags).branch == "T"
        feed(flags, [interval(3, sosi=0.5)])
        assert state(flags).branch == "T"


def test_n_in_needs_consecutive_entry_greens():
    with open_flags(FlagConfig(n_in=2)) as flags:
        feed(flags, [oversaturated(0), clean(1), oversaturated(2)])
        assert not state(flags).active and state(flags).entry_count == 1
        feed(flags, [oversaturated(3)])
        assert state(flags).active and state(flags).raised_at == interval(3).allred


# --- exit -------------------------------------------------------------------------

def test_exits_after_exactly_three_clean_greens_with_a_reset_between():
    with open_flags() as flags:
        feed(flags, [oversaturated(0), clean(1), clean(2)])
        assert state(flags).active and state(flags).clean_count == 2
        feed(flags, [interval(3, tosi=0.1, ratio=0.1)])
        assert state(flags).active and state(flags).clean_count == 0
        feed(flags, [clean(4), clean(5)])
        assert state(flags).active and state(flags).clean_count == 2
        feed(flags, [clean(6)])
        cleared = state(flags)
        assert not cleared.active
        assert (cleared.raised_at, cleared.entry_count, cleared.clean_count,
                cleared.branch) == (None, 0, 0, "")


def test_ratio_in_hysteresis_band_is_not_clean():
    with open_flags() as flags:
        feed(flags, [oversaturated(0)] + [interval(i, ratio=0.2) for i in range(1, 6)])
        assert state(flags).active
        assert state(flags).clean_count == 0


def test_sosi_loss_of_l_seconds_is_not_clean():
    with open_flags() as flags:
        feed(flags, [oversaturated(0)]
             + [interval(i, sosi=SOSI_AT_L, ratio=0.0) for i in range(1, 6)])
        assert state(flags).active
        assert state(flags).clean_count == 0


# --- missing inputs ----------------------------------------------------------------

def test_nan_tosi_neither_enters_nor_cleans():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=None, ratio=0.9, blocked=True)])
        assert not state(flags).active
        feed(flags, [oversaturated(1), clean(2), clean(3)])
        assert state(flags).clean_count == 2
        feed(flags, [interval(4, tosi=None, ratio=0.0, blocked=True)])
        assert state(flags).active
        assert state(flags).clean_count == 0


def test_missing_sosi_still_allows_a_tosi_entry():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.2, sosi=None, ratio=0.6)])
        assert state(flags).active and state(flags).branch == "T"


def test_missing_input_is_never_clean():
    for missing in ({"tosi": None}, {"sosi": None}, {"ratio": None}):
        with open_flags() as flags:
            feed(flags, [oversaturated(0), clean(1)])
            feed(flags, [interval(2, **{"tosi": 0.0, "sosi": 0.0, "ratio": 0.1, **missing})])
            assert state(flags).active, missing
            assert state(flags).clean_count == 0, missing


# --- movements of one lane -------------------------------------------------------------

def test_two_targets_keep_independent_sosi_state_but_share_lane_inputs():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.0, sosi=0.4, other_sosi=0.0, ratio=0.7)])
        assert state(flags, TARGET).active
        assert not state(flags, OTHER_TARGET).active
        assert flags.severity(LANE, TARGET) == flags.severity(LANE, OTHER_TARGET) == 0.7
        feed(flags, [interval(1, tosi=0.3, sosi=0.0, other_sosi=0.0, ratio=0.7)])
        assert state(flags, OTHER_TARGET).active
        assert flags.active_flags() == {(LANE, TARGET), (LANE, OTHER_TARGET)}


def test_severity_is_none_for_unseen_movement():
    with open_flags() as flags:
        assert flags.severity(LANE, TARGET) is None


# --- config ----------------------------------------------------------------------------

def test_config_rejects_too_narrow_hysteresis_gap():
    for theta_in, theta_out in ((0.5, 0.41), (0.5, 0.5), (0.3, 0.6)):
        try:
            FlagConfig(theta_in=theta_in, theta_out=theta_out)
        except ValueError:
            continue
        raise AssertionError(f"accepted theta_in={theta_in}, theta_out={theta_out}")
    FlagConfig(theta_in=0.5, theta_out=0.4)
    FlagConfig(theta_in=0.3, theta_out=0.2)


def test_cli_defaults_are_the_spec_constants_and_not_none():
    parser = argparse.ArgumentParser()
    add_movement_flag_args(parser)
    args = parser.parse_args(["--movement_flags"])
    assert (args.theta_in, args.theta_out, args.k_flag, args.n_in, args.sosi_loss_s) == \
        (0.30, 0.20, 3, 1, 5.0)
    assert movement_flag_config_from_args(args) == FlagConfig()


# --- CSV ---------------------------------------------------------------------------------

def test_csv_rows_and_columns():
    with tempfile.TemporaryDirectory() as tmp:
        flags = MovementFlags(FlagConfig(), Path(tmp))
        feed(flags, [oversaturated(0), interval(1, tosi=None, ratio=0.1),
                     interval(2, tosi=None, blocked=True)])
        flags.close()
        with open(Path(tmp) / MOVEMENT_FLAGS_FILENAME, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert reader.fieldnames == FLAG_COLUMNS
    assert FLAG_COLUMNS[-1] == "branch"
    assert len(rows) == 3 * 2
    first = rows[0]
    assert first["target_lane"] == TARGET and first["link_indices"] == "0;1"
    assert rows[1]["target_lane"] == OTHER_TARGET and rows[1]["link_indices"] == "2"
    assert (first["tosi"], first["ratio_start"], first["active"], first["raised_at"],
            first["branch"]) == ("0.2000", "0.6000", "1", str(interval(0).allred), "T")
    skipped, blocked = rows[2], rows[4]
    assert (skipped["skipped"], skipped["tosi"], skipped["active"]) == ("1", "", "1")
    assert (blocked["blocked"], blocked["skipped"], blocked["active"], blocked["branch"]) == \
        ("1", "1", "1", "T")


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
