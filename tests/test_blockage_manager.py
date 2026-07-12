"""Unit tests for the pure parts of utils/blockage_manager.py.

load_scenario and schedule_transitions never touch traci, so everything here
runs with no SUMO installed and no live simulator -- same pattern as
tests/test_state_features.py. TraCI side effects (insertion, freezing,
speed changes) are covered by the live smoke test instead.

Run: PYTHONPATH=<repo root> python tests/test_blockage_manager.py   (or pytest)
"""

import json
import tempfile
from pathlib import Path

from utils.blockage_manager import (
    BlockageManager,
    load_scenario,
    schedule_transitions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "simulations" / "single_intersection" / "scenarios"


def _blockage(bid="b1", lane="W2TLS_0", position=80.0, start=300, end=900,
              method="obstacle_vehicle", **extra):
    b = {"blockage_id": bid, "lane_id": lane, "position": position,
         "start_step": start, "end_step": end, "method": method}
    b.update(extra)
    return b


def _load(blockages):
    """Round-trip a scenario through a temp file and load_scenario."""
    scenario = {"scenario_name": "test", "description": "test",
                "blockages": blockages}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scenario.json"
        path.write_text(json.dumps(scenario))
        return load_scenario(path)


def _load_raises(blockages, expected_fragment):
    try:
        _load(blockages)
    except ValueError as e:
        assert expected_fragment in str(e), f"wrong error: {e}"
        return
    raise AssertionError(f"expected ValueError containing '{expected_fragment}'")


# --------------------------------------------------------------------------
# load_scenario: the shipped scenarios must parse
# --------------------------------------------------------------------------

def test_shipped_accident_scenario_loads():
    scenario = load_scenario(SCENARIOS_DIR / "accident_single_lane.json")
    assert scenario["scenario_name"] == "accident_single_lane"
    (b,) = scenario["blockages"]
    assert b["lane_id"] == "W2TLS_0"
    assert b["method"] == "obstacle_vehicle"
    assert (b["start_step"], b["end_step"]) == (300, 900)


def test_shipped_construction_scenario_loads():
    scenario = load_scenario(SCENARIOS_DIR / "construction_zone.json")
    assert len(scenario["blockages"]) == 2
    # Overlapping windows are fine on DIFFERENT lanes.
    assert {b["lane_id"] for b in scenario["blockages"]} == {"W2TLS_0", "W2TLS_1"}
    assert all(b["end_step"] is None for b in scenario["blockages"])


# --------------------------------------------------------------------------
# load_scenario: strict validation
# --------------------------------------------------------------------------

def test_severity_defaults_to_full():
    scenario = _load([_blockage()])  # _blockage() carries no severity key
    assert scenario["blockages"][0]["severity"] == 1.0


def test_missing_key_rejected():
    b = _blockage()
    del b["start_step"]
    _load_raises([b], "missing keys")


def test_unknown_key_rejected():
    _load_raises([_blockage(startstep=300)], "unknown keys")


def test_unknown_method_rejected():
    _load_raises([_blockage(method="lane_closure")], "unknown method")


def test_bad_severity_rejected():
    _load_raises([_blockage(severity=1.5)], "severity")


def test_end_not_after_start_rejected():
    _load_raises([_blockage(start=300, end=300)], "end_step")


def test_duplicate_ids_rejected():
    _load_raises([_blockage(bid="dup"), _blockage(bid="dup", start=1000, end=1100)],
                 "Duplicate")


def test_overlapping_speed_restrictions_same_lane_rejected():
    _load_raises(
        [_blockage(bid="a", method="speed_restriction", start=100, end=500),
         _blockage(bid="b", method="speed_restriction", start=400, end=800)],
        "overlap")


def test_sequential_speed_restrictions_same_lane_allowed():
    scenario = _load(
        [_blockage(bid="a", method="speed_restriction", start=100, end=400),
         _blockage(bid="b", method="speed_restriction", start=400, end=800)])
    assert len(scenario["blockages"]) == 2


# --------------------------------------------------------------------------
# schedule_transitions: pure scheduling logic
# --------------------------------------------------------------------------

def test_activation_fires_at_and_after_start():
    schedule = [_blockage(start=300, end=900)]
    for step, expect in ((299, 0), (300, 1), (301, 1)):
        to_activate, _, _ = schedule_transitions(schedule, step, set(), set())
        assert len(to_activate) == expect, f"step {step}"


def test_mid_window_start_activates():
    to_activate, to_deactivate, expired = schedule_transitions(
        [_blockage(start=300, end=900)], 500, set(), set())
    assert [b["blockage_id"] for b in to_activate] == ["b1"]
    assert to_deactivate == [] and expired == []


def test_whole_window_already_past_expires_without_activating():
    to_activate, to_deactivate, expired = schedule_transitions(
        [_blockage(start=300, end=900)], 1000, set(), set())
    assert to_activate == [] and to_deactivate == []
    assert [b["blockage_id"] for b in expired] == ["b1"]
    # Once finished, it never comes back.
    result = schedule_transitions([_blockage(start=300, end=900)], 1001,
                                  set(), {"b1"})
    assert result == ([], [], [])


def test_deactivation_at_end_step():
    schedule = [_blockage(start=300, end=900)]
    _, to_deactivate, _ = schedule_transitions(schedule, 899, {"b1"}, set())
    assert to_deactivate == []
    _, to_deactivate, _ = schedule_transitions(schedule, 900, {"b1"}, set())
    assert [b["blockage_id"] for b in to_deactivate] == ["b1"]


def test_open_ended_blockage_never_deactivates():
    schedule = [_blockage(start=100, end=None)]
    result = schedule_transitions(schedule, 100000, {"b1"}, set())
    assert result == ([], [], [])


def test_finished_blockage_does_not_reactivate():
    # After a normal deactivation the id is in finished; >= start must not
    # re-fire it.
    result = schedule_transitions([_blockage(start=300, end=900)], 950,
                                  set(), {"b1"})
    assert result == ([], [], [])


def test_independent_blockages_transition_independently():
    schedule = [_blockage(bid="early", start=100, end=200),
                _blockage(bid="late", start=150, end=None)]
    to_activate, to_deactivate, _ = schedule_transitions(
        schedule, 200, {"early"}, set())
    assert [b["blockage_id"] for b in to_activate] == ["late"]
    assert [b["blockage_id"] for b in to_deactivate] == ["early"]


# --------------------------------------------------------------------------
# BlockageManager: traci-free behavior
# --------------------------------------------------------------------------

def test_manager_idle_steps_need_no_traci():
    # Before any blockage is due, step() must not touch traci at all --
    # this is what keeps the manager usable in unit tests and cheap in runs.
    manager = BlockageManager([_blockage(start=100, end=200)])
    for step in range(5):
        manager.step(step)
    assert manager.get_active_blockages() == []
    assert manager.get_blocked_lane_ids() == []


def test_manager_expires_missed_window_without_traci():
    manager = BlockageManager([_blockage(start=10, end=20)])
    manager.step(50)  # whole window already past: straight to finished
    assert manager.get_active_blockages() == []
    manager.step(51)  # stays finished, still no traci needed
    assert manager.get_active_blockages() == []


def test_manager_reset_clears_state():
    manager = BlockageManager([_blockage(start=10, end=20)])
    manager.step(50)
    manager.reset()
    # After reset the missed window is treated as missed again (fresh run).
    manager.step(50)
    assert manager.get_active_blockages() == []


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
