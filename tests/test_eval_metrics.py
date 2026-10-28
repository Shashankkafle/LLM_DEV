"""tools/eval_metrics.py end to end on a logged run, after
tools/verify_flags.py --replay.

    python -m pytest tests/test_eval_metrics.py
    python tests/test_eval_metrics.py            # without pytest

Needs the shadow batch v3 run mp_cross_r1200_s1 under logs/ (not in git);
skipped when it is absent. The replay goes to a temporary dir, so the logged
run is never touched. No SUMO.
"""
import json
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import eval_metrics  # noqa: E402
from configurations import FINAL_SUMMARY_FILENAME  # noqa: E402
import verify_flags  # noqa: E402

RUN_GLOB = "mp_cross_r1200_s1_*"
SHADOW_BATCH = REPO / "logs" / "shadow_tuning_v3"


def logged_run():
    runs = sorted(SHADOW_BATCH.glob(RUN_GLOB))
    if not runs:
        raise unittest.SkipTest(f"no {RUN_GLOB} under {SHADOW_BATCH}")
    return runs[0]


def test_eval_metrics_runs_on_a_replayed_shadow_run():
    run_dir = logged_run()
    with tempfile.TemporaryDirectory() as tmp:
        replay_dir = Path(tmp) / run_dir.name
        verify_flags.replay(run_dir, replay_dir)
        assert verify_flags.verify(replay_dir)
        metrics = eval_metrics.evaluate(replay_dir)
        summary = json.loads((replay_dir / FINAL_SUMMARY_FILENAME).read_text())
    assert metrics["scenario_gate"]["passed"] is True
    assert metrics["detector"]["gt_positive_lane_cycles"] > 0
    assert metrics["peak_vehicles_in_system"] > 0
    # Both crossing surges enter on lane 1 of their entry edge.
    rates = metrics["arrival_rates"]["lanes"]
    assert {lane for lane, rate in rates.items() if "surge_entry" in rate["roles"]} == {
        "road_0_3_0_0", "road_0_3_0_1", "road_0_3_0_2",
        "road_2_5_3_0", "road_2_5_3_1", "road_2_5_3_2"}
    # The cycles' ATT contributions add up to the run's CityFlow-style ATT.
    assert abs(metrics["att_s"] - summary["cityflow_style_att_s"]) < 0.5
    assert set(metrics["regime_slices"]["per_regime"]) == {"N", "L", "P", "R"}


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_"):
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
