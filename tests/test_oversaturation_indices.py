"""Tests for the TOSI/SOSI oversaturation indices (utils/oversaturation_indices.py).

    python -m pytest tests/test_oversaturation_indices.py                 # everything
    python -m pytest tests/test_oversaturation_indices.py -k "not smoke"  # no SUMO
    python tests/test_oversaturation_indices.py [--skip-smoke]            # without pytest

The rule and state-machine tests need no SUMO. The smoke tests run FixedTime
on hzreal for 600-700 s (SUMO on PATH, ~1-2 min each) and write their run dirs to
tests/artifacts/oversat/<test_name>/ (wiped on each run).
"""
import csv
import shutil
import subprocess
import sys
import traceback
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import OBSTACLE_VEHICLE_PREFIX  # noqa: E402
from experiments import BLOCKAGES, CONFIGS  # noqa: E402
from utils.oversaturation_indices import (  # noqa: E402
    EMPTY_SNAPSHOT,
    LaneGreenTracker,
    build_lane_snapshot,
    compute_sosi,
    compute_tosi,
    exited_ids,
    is_spillback_blocked,
    lane_signal,
    queue_storage_ratio,
    storage_capacity,
)

ARTIFACTS = REPO / "tests" / "artifacts" / "oversat"
SMOKE_STEPS = 600
# c3 activates at 500 s; road_2_4_3_1's first blocked green reaches all-red at 628.
C3_STEPS = 700
OBSTACLE = OBSTACLE_VEHICLE_PREFIX + "c3"
LANE_LENGTH = 100.0
LANE = "in_1"
TARGET = "out_1"


def require_sumo():
    if shutil.which("sumo") is None:
        raise unittest.SkipTest("sumo is not on PATH")


def artifact_dir(test_name):
    path = ARTIFACTS / test_name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


# --- rules --------------------------------------------------------------------

def test_tosi_from_synthetic_id_sets():
    tosi, discharged = compute_tosi({1, 2, 3}, {2, 3, 4}, green_s=30)
    assert tosi == 2 * 2.0 / 30
    assert discharged == 1


def test_tosi_clamps_to_one():
    tosi, _ = compute_tosi({1, 2, 3}, {2, 3, 4}, green_s=3)
    assert tosi == 1.0


def test_count_flat_exit_counts_as_exit():
    assert exited_ids({1, 2}, {2, 3}) == {1}
    assert not is_spillback_blocked(True, len(exited_ids({1, 2}, {2, 3})), True)


def test_sosi_needs_all_three_rules():
    assert is_spillback_blocked(True, 0, True)
    assert not is_spillback_blocked(False, 0, True)
    assert not is_spillback_blocked(True, 1, True)
    assert not is_spillback_blocked(True, 0, False)


def test_sosi_short_green_is_none():
    assert compute_sosi(0, 3) is None
    assert compute_sosi(27, 30) == 1.0
    assert compute_sosi(40, 30) == 1.0


def test_storage_ratio():
    assert storage_capacity(572.8) == 76
    assert queue_storage_ratio(11, 76) == 11 / 76
    assert queue_storage_ratio(3, 0) is None


def test_lane_signal_collapses_links():
    state = "gggGGGrrryyy"
    assert lane_signal(state, [0, 1, 2]) == "g"
    assert lane_signal(state, [3, 4, 5]) == "G"
    assert lane_signal(state, [6, 7, 8]) == "r"
    assert lane_signal(state, [9, 10, 11]) == "y"


def test_obstacle_excluded_from_snapshot_rules():
    vehicles = [(OBSTACLE, 0.0, LANE_LENGTH - 10), ("v1", 5.0, 50.0)]
    snapshot = build_lane_snapshot(vehicles, LANE_LENGTH)
    assert snapshot.ids == {"v1"}
    assert snapshot.halted_ids == frozenset()
    assert not snapshot.halted_near_stop          # rule (a)
    target = build_lane_snapshot([(OBSTACLE, 0.0, 5.0)], LANE_LENGTH)
    assert not target.halted_near_entry           # rule (c)


# --- LaneGreenTracker on synthetic snapshots -------------------------------------

def stalled_queue_snapshot():
    """Five halted vehicles, the head at the stop line."""
    return build_lane_snapshot(
        [(f"q{i}", 0.0, LANE_LENGTH - 2 - 7.5 * i) for i in range(5)], LANE_LENGTH)


def drive(tracker, signals, lane_snapshots, target_snapshot, blocked=False):
    """Feed one signal per second. lane_snapshots[i] is the lane at time i;
    returns the closed intervals and the green-start count."""
    closed, starts = [], 0
    for tau, signal in enumerate(signals):
        observation = tracker.observe(tau, signal, lane_snapshots[tau],
                                      lane_snapshots[tau + 1], {TARGET: target_snapshot},
                                      blocked)
        starts += observation.green_started
        if observation.closed is not None:
            closed.append(observation.closed)
    return closed, starts


def cycle_signals(red_before=5, green=30, yellow=3, red_after=2):
    return ["r"] * red_before + ["G"] * green + ["y"] * yellow + ["r"] * red_after


def new_tracker():
    return LaneGreenTracker("tls", LANE, LANE_LENGTH, [0], [TARGET])


def test_tracker_full_cycle_with_spillback():
    signals = cycle_signals()
    queue = stalled_queue_snapshot()
    snapshots = [queue] * (len(signals) + 1)
    full_target = build_lane_snapshot([("t1", 0.0, 3.0)], LANE_LENGTH)
    (result,), starts = drive(new_tracker(), signals, snapshots, full_target)
    assert starts == 1
    assert (result.green_start, result.green_end, result.allred) == (5, 35, 38)
    assert result.green_s == 30
    assert result.halted_at_start == {f"q{i}" for i in range(5)}
    assert result.held_over == 5 and result.discharged == 0
    assert result.tosi == 5 * 2.0 / 30
    assert result.blocked_steps == {TARGET: 27}
    assert result.sosi == {TARGET: 1.0}
    assert result.ratio_start == 5 / 13


def test_stalled_queue_with_empty_target_is_not_spillback():
    signals = cycle_signals()
    snapshots = [stalled_queue_snapshot()] * (len(signals) + 1)
    (result,), _ = drive(new_tracker(), signals, snapshots, EMPTY_SNAPSHOT)
    assert result.blocked_steps == {TARGET: 0}
    assert result.sosi == {TARGET: 0.0}


def test_h_comes_from_the_snapshot_when_green_came_on():
    """At tau=5 the green starts; the snapshot at time 5 holds the queue, the
    one at time 6 already shows the head moving. H must be the time-5 set."""
    signals = cycle_signals()
    queue = stalled_queue_snapshot()
    moving = build_lane_snapshot([("q0", 2.6, LANE_LENGTH - 1)]
                                 + [(f"q{i}", 0.0, LANE_LENGTH - 2 - 7.5 * i) for i in range(1, 5)],
                                 LANE_LENGTH)
    snapshots = [queue] * 6 + [moving] + [EMPTY_SNAPSHOT] * (len(signals) - 5)
    (result,), _ = drive(new_tracker(), signals, snapshots, EMPTY_SNAPSHOT)
    assert "q0" in result.halted_at_start
    assert result.discharged == 5 and result.tosi == 0.0


def test_obstacle_excluded_from_h_and_r():
    signals = cycle_signals()
    lane = build_lane_snapshot([(OBSTACLE, 0.0, LANE_LENGTH - 10), ("q1", 0.0, LANE_LENGTH - 20)],
                               LANE_LENGTH)
    snapshots = [lane] * (len(signals) + 1)
    (result,), _ = drive(new_tracker(), signals, snapshots, EMPTY_SNAPSHOT)
    assert OBSTACLE not in result.halted_at_start
    assert OBSTACLE not in result.on_lane_at_allred
    assert result.held_over == 1


def test_blocked_lane_suppresses_tosi_only():
    signals = cycle_signals()
    snapshots = [stalled_queue_snapshot()] * (len(signals) + 1)
    (result,), _ = drive(new_tracker(), signals, snapshots, EMPTY_SNAPSHOT, blocked=True)
    assert result.blocked_flag
    assert result.tosi is None
    assert result.sosi == {TARGET: 0.0}


def test_green_to_green_without_red_closes_at_restart():
    signals = ["G"] * 10 + ["y"] * 3 + ["G"] * 10 + ["y"] * 3 + ["r"]
    snapshots = [stalled_queue_snapshot()] * (len(signals) + 1)
    closed, starts = drive(new_tracker(), signals, snapshots, EMPTY_SNAPSHOT)
    assert starts == 2
    assert [(r.green_start, r.green_end, r.allred) for r in closed] == [(0, 10, 13), (13, 23, 26)]


def test_always_green_lane_never_closes():
    signals = ["g"] * 100
    snapshots = [EMPTY_SNAPSHOT] * (len(signals) + 1)
    closed, starts = drive(new_tracker(), signals, snapshots, EMPTY_SNAPSHOT)
    assert starts == 1 and closed == []


# --- smoke: FixedTime on hzreal ------------------------------------------------------

def run_fixedtime(logs_dir, blockage="none"):
    command = [sys.executable, "runner_baselines.py", "--controller", "fixedtime",
               "--simulation_config", CONFIGS["hzreal"], "--simulation_steps",
               str(SMOKE_STEPS), "--seed", "1", "--test_name", "ft", "--logs_dir",
               str(logs_dir)]
    if BLOCKAGES[blockage]:
        command += ["--blockage_scenario", BLOCKAGES[blockage]]
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "runner_stdout.txt", "w") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return sorted(logs_dir.glob("ft_*"))[-1]


def read_rows(run_dir):
    with open(run_dir / "oversaturation_indices.csv", newline="") as f:
        return list(csv.DictReader(f))


def assert_unit_or_empty(value, row):
    assert value == "" or 0.0 <= float(value) <= 1.0, row


def test_smoke_fixedtime_hzreal():
    require_sumo()
    run_dir = run_fixedtime(artifact_dir("test_smoke_fixedtime_hzreal"))
    rows = read_rows(run_dir)
    assert {row["tls"] for row in rows} == {f"intersection_{x}_{y}"
                                            for x in range(1, 5) for y in range(1, 5)}
    for row in rows:
        assert_unit_or_empty(row["tosi"], row)
        assert_unit_or_empty(row["sosi"], row)
        assert not row["lane"].endswith("_0"), row   # always-green right turns
        if int(row["t_green_start"]) >= 60:
            assert row["green_s"] == "30", row
    print(f"  {len(rows)} rows -> {run_dir}")


def sumo_output_body(path):
    """SUMO output minus its header comment, which embeds the run dir path."""
    text = path.read_text(encoding="utf-8")
    return text[text.index("-->"):]


def run_in_process(logs_dir, stub_indices, movement_flags=False):
    """runner_baselines.main on c3, optionally with the indices replaced by a
    no-op (no subscriptions, no reads) to get the untouched reference, or with
    the movement flags listening on top of the indices."""
    import runner_baselines
    import runner_common

    class NoIndices:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, t):
            pass

        def close(self):
            pass

    real = runner_common.OversaturationIndices
    if stub_indices:
        runner_common.OversaturationIndices = NoIndices
    try:
        runner_baselines.main(runner_baselines.argparse.Namespace(
            controller="fixedtime", test_name="ft", run_group=None,
            simulation_steps=C3_STEPS, simulation_config=CONFIGS["hzreal"],
            use_gui=False, seed=1, blockage_scenario=BLOCKAGES["c3"],
            intersection_config="three_lane", logs_dir=str(logs_dir),
            oversat_shadow=False, movement_flags=movement_flags,
            theta_in=None, theta_out=None, n_in=None, k_flag=None))
    finally:
        runner_common.OversaturationIndices = real
    return sorted(Path(logs_dir).glob("ft_*"))[-1]


def assert_same_control(run, reference):
    phase_files = sorted(p.name for p in (run / "phase_sequences").iterdir())
    assert phase_files
    for name in phase_files:
        assert ((run / "phase_sequences" / name).read_bytes()
                == (reference / "phase_sequences" / name).read_bytes()), name
    assert (run / "network_timeseries.csv").read_bytes() == \
        (reference / "network_timeseries.csv").read_bytes()
    assert sumo_output_body(run / "tripinfo.xml") == sumo_output_body(reference / "tripinfo.xml")


def read_flag_rows(run_dir):
    with open(run_dir / "movement_flags.csv", newline="") as f:
        return list(csv.DictReader(f))


def movements(rows):
    return {(row["lane"], row["target_lane"]) for row in rows}


def test_smoke_indices_do_not_change_control():
    require_sumo()
    out_dir = artifact_dir("test_smoke_indices_do_not_change_control")
    live = run_in_process(out_dir / "live", stub_indices=False)
    reference = run_in_process(out_dir / "stub", stub_indices=True)
    with_flags = run_in_process(out_dir / "flags", stub_indices=False, movement_flags=True)
    assert_same_control(live, reference)
    assert_same_control(with_flags, reference)
    rows = read_rows(live)
    blocked = [row for row in rows if row["blocked_flag"] == "1"]
    assert blocked and all(row["lane"] == "road_2_4_3_1" and row["tosi"] == ""
                           for row in blocked)
    assert not (live / "movement_flags.csv").exists()
    flag_rows = read_flag_rows(with_flags)
    assert movements(flag_rows) == movements(read_rows(with_flags))
    assert len(flag_rows) == len(read_rows(with_flags))


if __name__ == "__main__":
    skip_smoke = "--skip-smoke" in sys.argv
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or (skip_smoke and "smoke" in name):
            continue
        try:
            test()
            print(f"[ok  ] {name}")
        except unittest.SkipTest as skip:
            print(f"[skip] {name}: {skip}")
        except Exception:
            failures += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    sys.exit(1 if failures else 0)
