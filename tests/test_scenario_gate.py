"""Tests for the scenario gate (utils/scenario_gate.py).

    python -m pytest tests/test_scenario_gate.py
    python tests/test_scenario_gate.py            # without pytest

Synthetic lane greens and a hand-built network, no SUMO.
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import BLOCKAGE_SCENARIO_COPY_FILENAME  # noqa: E402
from utils.network_topology import Network  # noqa: E402
from utils.run_logs import LaneGreen  # noqa: E402
from utils.scenario_gate import first_persistent_green, target_feeder_lanes  # noqa: E402

THETA_IN = 0.30
FEEDER = "road_2_5_3_1"
BLOCKED = "road_2_4_3_1"
TURN_FEEDER = "road_1_4_0_0"
ELSEWHERE = "road_0_1_0_1"


def lane_green(t_close, ratio_start, lane=FEEDER):
    return LaneGreen(t_close=t_close, tls="tls", lane=lane, tosi=0.0,
                     ratio_start=ratio_start, ratio_end=0.0, max_sosi=0.0,
                     held_over=0, discharged=0)


def sequence(ratios, lane=FEEDER):
    return [lane_green(100 * (i + 1), ratio, lane) for i, ratio in enumerate(ratios)]


def test_gate_passes_on_three_consecutive_loaded_greens():
    greens = sequence([0.1, 0.35, 0.30, 0.42, 0.1])
    assert first_persistent_green(greens, {FEEDER}, THETA_IN) == 400


def test_gate_fails_on_two_consecutive_loaded_greens():
    greens = sequence([0.35, 0.40, 0.1, 0.35, 0.40])
    assert first_persistent_green(greens, {FEEDER}, THETA_IN) is None


def test_a_missing_ratio_breaks_the_run():
    greens = sequence([0.35, 0.40, None, 0.35])
    assert first_persistent_green(greens, {FEEDER}, THETA_IN) is None


def test_only_feeder_lanes_count():
    greens = sequence([0.5, 0.5, 0.5], lane=ELSEWHERE)
    assert first_persistent_green(greens, {FEEDER}, THETA_IN) is None


def test_feeder_lanes_come_from_net_connections_and_surge_entry_edges():
    network = Network(
        edge_lanes={"road_0_3_0": ["road_0_3_0_0", "road_0_3_0_1"]},
        connections={(FEEDER, BLOCKED), (TURN_FEEDER, BLOCKED), (ELSEWHERE, "road_1_1_0_1")})
    manifest = {"surge": {"surges": [{"route": ["road_0_3_0", "road_1_3_0"]}]}}
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / BLOCKAGE_SCENARIO_COPY_FILENAME).write_text(
            json.dumps({"blockages": [{"lane_id": BLOCKED, "end_step": 1700}]}))
        lanes = target_feeder_lanes(manifest, run_dir, network)
    assert lanes == {FEEDER, TURN_FEEDER, "road_0_3_0_0", "road_0_3_0_1"}


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
