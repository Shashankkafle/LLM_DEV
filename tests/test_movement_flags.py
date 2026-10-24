"""Tests for the per-movement oversaturation flags (utils/movement_flags.py).

    python -m pytest tests/test_movement_flags.py
    python tests/test_movement_flags.py            # without pytest

All synthetic IntervalResult sequences, no SUMO. The SUMO smoke check that
flags leave control untouched lives in test_oversaturation_indices.py
(test_smoke_indices_do_not_change_control).
"""
import csv
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import MOVEMENT_FLAGS_FILENAME  # noqa: E402
from utils.movement_flags import FLAG_COLUMNS, FlagConfig, MovementFlags  # noqa: E402
from utils.oversaturation_indices import IntervalResult, LaneGreenTracker  # noqa: E402

LANE = "in_1"
TARGET = "out_1"
OTHER_TARGET = "out_2"
GREEN_S = 30
CYCLE_S = 60
CAPACITY = 20


def new_tracker():
    return LaneGreenTracker("tls", LANE, 150.0, [0, 1, 2], [TARGET, OTHER_TARGET],
                            {TARGET: [0, 1], OTHER_TARGET: [2]})


def interval(index, tosi=0.0, sosi=0.0, ratio=0.0, blocked=False, other_sosi=0.0):
    """The index-th closed green of LANE; ratio_end is set directly."""
    green_start = index * CYCLE_S
    return IntervalResult(
        tls="tls", lane=LANE, green_start=green_start,
        green_end=green_start + GREEN_S, allred=green_start + GREEN_S + 3,
        halted_at_start=frozenset(), on_lane_at_allred=frozenset(),
        tosi=tosi, discharged=0, blocked_flag=blocked, capacity=CAPACITY,
        halted_count_start=0,
        halted_count_end=None if ratio is None else round(ratio * CAPACITY),
        ratio_start=0.0, ratio_end=ratio,
        blocked_steps={TARGET: 0, OTHER_TARGET: 0},
        sosi={TARGET: sosi, OTHER_TARGET: other_sosi},
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

def test_enters_after_one_green_with_tosi_and_high_ratio():
    with open_flags() as flags:
        feed(flags, [oversaturated(0)])
        assert state(flags).active
        assert state(flags).raised_at == interval(0).allred
        assert state(flags).entry_count == 1
        assert (LANE, TARGET) in flags.active_flags()


def test_tosi_without_high_ratio_does_not_enter():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.5, ratio=0.49)])
        assert not state(flags).active
        assert state(flags).entry_count == 0
        assert flags.active_flags() == set()


def test_sosi_alone_enters():
    with open_flags() as flags:
        feed(flags, [interval(0, tosi=0.0, sosi=0.3, ratio=0.5)])
        assert state(flags).active


def test_n_in_needs_consecutive_entry_greens():
    with open_flags(FlagConfig(n_in=2)) as flags:
        feed(flags, [oversaturated(0), clean(1), oversaturated(2)])
        assert not state(flags).active and state(flags).entry_count == 1
        feed(flags, [oversaturated(3)])
        assert state(flags).active and state(flags).raised_at == interval(3).allred


# --- exit -------------------------------------------------------------------------

def test_dirty_green_resets_clean_count_and_clears_after_k_flag():
    with open_flags() as flags:
        feed(flags, [oversaturated(0), clean(1), clean(2)])
        assert state(flags).active and state(flags).clean_count == 2
        feed(flags, [interval(3, tosi=0.1, ratio=0.1)])
        assert state(flags).active and state(flags).clean_count == 0
        feed(flags, [clean(4), clean(5)])
        assert state(flags).active
        feed(flags, [clean(6)])
        cleared = state(flags)
        assert not cleared.active
        assert (cleared.raised_at, cleared.entry_count, cleared.clean_count) == (None, 0, 0)


def test_ratio_in_hysteresis_band_is_not_clean():
    with open_flags() as flags:
        feed(flags, [oversaturated(0)] + [interval(i, ratio=0.4) for i in range(1, 6)])
        assert state(flags).active
        assert state(flags).clean_count == 0


# --- blocked and missing inputs ------------------------------------------------------

def test_blocked_green_forces_inactive_and_resets():
    with open_flags(FlagConfig(n_in=2)) as flags:
        feed(flags, [oversaturated(0), oversaturated(1)])
        assert state(flags).active
        feed(flags, [interval(2, tosi=None, sosi=0.5, ratio=0.9, blocked=True)])
        blocked = state(flags)
        assert not blocked.active
        assert (blocked.raised_at, blocked.entry_count, blocked.clean_count) == (None, 0, 0)
        feed(flags, [oversaturated(3)])
        assert not state(flags).active and state(flags).entry_count == 1


def test_missing_input_skips_without_changing_state():
    for missing in ({"tosi": None}, {"sosi": None}, {"ratio": None}):
        with open_flags() as flags:
            feed(flags, [oversaturated(0), clean(1)])
            before = (state(flags).active, state(flags).raised_at, state(flags).entry_count,
                      state(flags).clean_count, state(flags).last_ratio)
            feed(flags, [interval(2, **{"tosi": 0.0, "sosi": 0.0, "ratio": 0.1, **missing})])
            after = (state(flags).active, state(flags).raised_at, state(flags).entry_count,
                     state(flags).clean_count, state(flags).last_ratio)
            assert after == before, missing


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
    assert len(rows) == 3 * 2
    first = rows[0]
    assert first["target_lane"] == TARGET and first["link_indices"] == "0;1"
    assert rows[1]["target_lane"] == OTHER_TARGET and rows[1]["link_indices"] == "2"
    assert (first["tosi"], first["ratio_end"], first["active"], first["raised_at"]) == \
        ("0.2000", "0.6000", "1", str(interval(0).allred))
    skipped, blocked = rows[2], rows[4]
    assert (skipped["skipped"], skipped["tosi"], skipped["active"]) == ("1", "", "1")
    assert (blocked["blocked"], blocked["skipped"], blocked["active"], blocked["raised_at"]) == \
        ("1", "0", "0", "")


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
