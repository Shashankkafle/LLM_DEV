"""Tests for the evaluation logs (utils/evaluation_logs.py) as the Diagnoser
drives them, and for the executor log headers (utils/executor_log.py).

    python -m pytest tests/test_evaluation_logs.py
    python tests/test_evaluation_logs.py            # without pytest

Synthetic closed greens, no SUMO.
"""
import csv
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import (  # noqa: E402
    COMMANDS_FILENAME,
    GROUND_TRUTH_FILENAME,
    MOVEMENT_FLAGS_FILENAME,
    NETWORK_METRICS_FILENAME,
    PLAN_CHANGES_FILENAME,
)
from utils.diagnoser import Diagnoser  # noqa: E402
from utils.evaluation_logs import (  # noqa: E402
    GROUND_TRUTH_COLUMNS,
    NETWORK_METRICS_COLUMNS,
    EvaluationLogs,
    NetworkState,
    side_street_halted,
)
from utils.executor_log import COMMAND_COLUMNS, PLAN_CHANGE_COLUMNS, ExecutorLog  # noqa: E402
from utils.movement_flags import FlagConfig, MovementFlags  # noqa: E402
from utils.oversaturation_indices import IntervalResult, LaneGreenTracker  # noqa: E402

CYCLE_S = 140
CAPACITY = 20
LANE = "in_1"
OTHER_LANE = "in_2"
TARGET = "out_1"


def tracker(lane):
    return LaneGreenTracker("tls", lane, 150.0, [0], [TARGET], {TARGET: [0]})


def green(lane, allred, tosi=0.0, ratio=0.0, held_over=0, discharged=0):
    """A closed green of lane; held_over vehicles are both halted at its
    start and still on the lane at all-red."""
    held = frozenset(f"{lane}_veh{i}" for i in range(held_over))
    return IntervalResult(
        tls="tls", lane=lane, green_start=allred - 33, green_end=allred - 3,
        allred=allred, halted_at_start=held, on_lane_at_allred=held,
        tosi=tosi, discharged=discharged, blocked_flag=False, capacity=CAPACITY,
        halted_count_start=round(ratio * CAPACITY), halted_count_end=0,
        ratio_start=ratio, ratio_end=0.0, blocked_steps={TARGET: 0},
        sosi={TARGET: 0.0}, stop_halted_steps=0, no_exit_steps=0,
        target_halted_steps={TARGET: 0},
    )


def run(greens, last_t, config=None, network_state=None):
    """Feed greens (allred -> [IntervalResult]) through MovementFlags,
    Diagnoser and EvaluationLogs. Returns {filename: rows}."""
    trackers = {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        flags = MovementFlags(config or FlagConfig(), out)
        diagnoser = Diagnoser(flags, CYCLE_S, out, EvaluationLogs(out, network_state))
        for t in range(1, last_t + 1):
            for result in greens.get(t, []):
                lane_tracker = trackers.setdefault(result.lane, tracker(result.lane))
                diagnoser.on_interval_closed(lane_tracker, result)
            diagnoser.update(t)
        diagnoser.close()
        flags.close()
        files = {}
        for name in (MOVEMENT_FLAGS_FILENAME, GROUND_TRUTH_FILENAME, NETWORK_METRICS_FILENAME):
            with open(out / name, newline="") as f:
                files[name] = list(csv.DictReader(f))
    return files


# --- ground truth ----------------------------------------------------------------------

# Four greens of LANE at TOSI > 0 and ratio_start 0.4: a flag under
# theta_in = 0.3, none under theta_in = 0.5.
LOADED_GREENS = {t: [green(LANE, t, tosi=0.2, ratio=0.4)] for t in (100, 200, 300, 400)}


def test_ground_truth_does_not_depend_on_theta_in():
    low = run(LOADED_GREENS, last_t=559, config=FlagConfig(theta_in=0.3, theta_out=0.2))
    high = run(LOADED_GREENS, last_t=559, config=FlagConfig(theta_in=0.5, theta_out=0.4))
    assert {row["active"] for row in low[MOVEMENT_FLAGS_FILENAME]} == {"1"}
    assert {row["active"] for row in high[MOVEMENT_FLAGS_FILENAME]} == {"0"}
    assert low[GROUND_TRUTH_FILENAME] == high[GROUND_TRUTH_FILENAME]


def test_ground_truth_needs_three_consecutive_tosi_greens_and_carries_forward():
    rows = run(LOADED_GREENS, last_t=559)[GROUND_TRUTH_FILENAME]
    assert list(rows[0]) == GROUND_TRUTH_COLUMNS
    # Greens close at 100 | 200 | 300, 400 | -, so cycles end on counts 1, 2, 4, 4.
    assert [(row["cycle_start"], row["consec_tosi_greens"], row["gt_oversat"])
            for row in rows] == [("0", "1", "0"), ("140", "2", "0"),
                                 ("280", "4", "1"), ("420", "4", "1")]


def test_a_green_without_tosi_resets_the_run():
    greens = {100: [green(LANE, 100, tosi=0.2)], 200: [green(LANE, 200, tosi=0.2)],
              300: [green(LANE, 300, tosi=0.0)], 450: [green(LANE, 450, tosi=0.2)]}
    rows = run(greens, last_t=559)[GROUND_TRUTH_FILENAME]
    assert [row["consec_tosi_greens"] for row in rows] == ["1", "2", "0", "1"]


# --- network metrics ---------------------------------------------------------------------

def test_total_overflow_carries_each_lanes_last_green_forward():
    # LANE's only green is in cycle 0; OTHER_LANE's are in cycles 0 and 1.
    greens = {50: [green(LANE, 50, held_over=4, discharged=3),
                   green(OTHER_LANE, 50, held_over=2, discharged=1)],
              200: [green(OTHER_LANE, 200, held_over=1, discharged=5)]}
    rows = run(greens, last_t=419)[NETWORK_METRICS_FILENAME]
    assert list(rows[0]) == NETWORK_METRICS_COLUMNS
    assert [(row["total_overflow"], row["discharged"]) for row in rows] == [
        ("6", "4"), ("5", "5"), ("5", "0")]


def test_side_street_halted_excludes_flagged_lanes():
    def network_state(cycle_start, t):
        return NetworkState(vehicles_in_system=50, total_halted=10,
                            lane_halting={LANE: 6, OTHER_LANE: 3})

    # LANE is flagged by its green at 200 (cycle 1) and stays flagged.
    greens = {50: [green(LANE, 50), green(OTHER_LANE, 50)],
              200: [green(LANE, 200, tosi=0.2, ratio=0.6)]}
    rows = run(greens, last_t=419, network_state=network_state)[NETWORK_METRICS_FILENAME]
    assert [(row["vehicles_in_system"], row["total_halted"], row["side_street_halted"])
            for row in rows] == [("50", "10", "10"), ("50", "10", "4"), ("50", "10", "4")]


def test_network_columns_are_empty_without_logged_state():
    rows = run({50: [green(LANE, 50)]}, last_t=139)[NETWORK_METRICS_FILENAME]
    assert [(row["vehicles_in_system"], row["total_halted"], row["side_street_halted"])
            for row in rows] == [("", "", "")]
    assert side_street_halted(10, None, {LANE}) is None


# --- executor log -------------------------------------------------------------------------

def test_executor_log_writes_headers_and_appends_rows():
    with tempfile.TemporaryDirectory() as tmp:
        log = ExecutorLog(Path(tmp))
        log.log_plan_change(300, "tls", 140, 120, {"ETWT": 40}, {"ETWT": 30}, "cycle_change")
        log.close()
        with open(Path(tmp) / COMMANDS_FILENAME, newline="") as f:
            commands = list(csv.reader(f))
        with open(Path(tmp) / PLAN_CHANGES_FILENAME, newline="") as f:
            plans = list(csv.reader(f))
    assert commands == [COMMAND_COLUMNS]
    assert plans == [PLAN_CHANGE_COLUMNS,
                     ["300", "tls", "140", "120", '{"ETWT":40}', '{"ETWT":30}', "cycle_change"]]


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
