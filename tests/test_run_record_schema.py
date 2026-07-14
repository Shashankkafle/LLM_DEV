"""Run-record schema guard: the manifest and decision/summary records must
keep the keys the analysis tooling depends on.

Pure Python -- no SUMO needed (get_final_summary and the decision recorders
never touch TraCI). Run: PYTHONPATH=<repo root> python tests/test_run_record_schema.py
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from configurations import INTERSECTION_CONFIGS, RUN_RECORD_SCHEMA_VERSION
from utils.metrics_recorder import MetricsRecorder
from utils.run_manifest import build_manifest, save_manifest, finalize_manifest

SUMOCFG = "simulations/single_intersection/run.sumocfg"
SCENARIO = "simulations/single_intersection/scenarios/accident_single_lane.json"

failures = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def make_args():
    return argparse.Namespace(
        test_name="schema_test", simulation_steps=100,
        simulation_config=SUMOCFG, intersection_config="single_intersection",
        seed=7, blockage_scenario=SCENARIO, hide_blockage_info=True,
        blockage_info_scope="both", use_gui=False,
    )


def check_manifest(tmp):
    conf = INTERSECTION_CONFIGS["single_intersection"]
    manifest = build_manifest("fixedtime", make_args(), "single_intersection", conf)

    check(manifest["schema_version"] == RUN_RECORD_SCHEMA_VERSION,
          "manifest carries the current schema_version")
    check(manifest["controller"] == "fixedtime", "manifest names the controller")
    check(manifest["status"] == "running", "manifest starts as status=running")
    check(manifest["args"].get("seed") == 7, "raw CLI args are dumped")
    check("use_gui" in manifest["args"], "args include use_gui")
    check("git_commit" in manifest["provenance"]
          and "git_dirty" in manifest["provenance"],
          "provenance has git commit + dirty flag")
    check(manifest["provenance"]["packages"].get("python"),
          "provenance records the python version")
    env = manifest["environment"]
    check(isinstance(env["input_files"], dict) and env["input_files"],
          "environment fingerprints the sumocfg + net/route files")
    check(env["signal_timing"]["yellow_duration"] == 3
          and env["signal_timing"]["default_green_duration"] == 30,
          "environment records the signal timing constants")
    check(env["phases"]["ETWT"]["green"], "environment records the phase RYG strings")
    check(env["seed"] == 7 and env["simulation_steps"] == 100,
          "environment records seed and step limit")
    blockage = manifest["blockage"]
    check(blockage["scenario_name"] == "accident_single_lane"
          and blockage["hide_blockage_info"] is True,
          "blockage block records scenario name + ablation flags")

    save_manifest(tmp, manifest)
    finalize_manifest(tmp, "completed")
    reloaded = json.loads((Path(tmp) / "run_manifest.json").read_text())
    check(reloaded["status"] == "completed", "finalize stamps the final status")
    check(reloaded["ended_at"] is not None
          and reloaded["wall_clock_duration_s"] is not None,
          "finalize stamps end time + duration")


def check_decision_records(tmp):
    run_dir = Path(tmp) / "run"
    run_info = {"test_name": "schema_test", "controller": "maxpressure", "seed": 7}
    recorder = MetricsRecorder(run_dir=run_dir, verbose=False, run_info=run_info)

    recorder.record_simple_decision(
        step=30, intersection_id="TLS", previous_phase="ETWT",
        activated_phase="NTST", decision_type="rule_decision",
        controller_state={"pressures": {"ETWT": 1, "NTST": 4}},
        traffic_state={"ETWT": {}})
    recorder.record_decision(
        step=60, state_dict={"movement_states": {}}, prompt="user prompt",
        llm_output="...<signal>ETWT</signal>", previous_phase="NTST",
        final_phase="ETWT", decision_type="llm_decision", latency_ms=12.5,
        extracted_signal="ETWT", intersection_id="TLS",
        blockage_facts=[{"lane_id": "W2TLS_0"}], exit_blockage_facts=[],
        blockage_info_in_prompt=False,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50})
    recorder.record_decision(
        step=90, state_dict={"movement_states": {}}, prompt="user prompt",
        llm_output=None, previous_phase="ETWT", final_phase="ETWT",
        decision_type="inference_error", latency_ms=3.0,
        extracted_signal=None, intersection_id="TLS",
        error="RuntimeError('CUDA out of memory')")

    lines = [json.loads(line) for line in
             (run_dir / "TLS" / "decisions.jsonl").read_text().splitlines()]
    check(len(lines) == 3, "all three decisions were appended")

    simple, llm, err = lines
    check(simple["intersection_id"] == "TLS" and simple["controller"] == "maxpressure",
          "simple decision carries intersection_id + controller")
    check(simple["controller_state"]["pressures"]["NTST"] == 4,
          "simple decision keeps the controller's working state")
    check(simple["phase_action"]["phase_changed"] is True,
          "simple decision records the phase transition")
    check(llm["blockage_facts"] and llm["blockage_info_in_prompt"] is False,
          "LLM decision records blockage facts even when hidden from the prompt")
    check(llm["metrics"]["prompt_tokens"] == 100
          and llm["metrics"]["completion_tokens"] == 50,
          "LLM decision records token usage")
    check(err["llm_output"]["error"] and
          err["phase_action"]["decision_type"] == "inference_error",
          "inference errors are recorded per decision")

    summary = recorder.get_final_summary()
    check(summary["schema_version"] == RUN_RECORD_SCHEMA_VERSION,
          "final summary carries schema_version")
    check(summary["controller"] == "maxpressure" and summary["seed"] == 7,
          "final summary carries the run identity")
    check(summary["decisions_inference_error"] == 1,
          "inference errors are counted in the summary")
    check(summary["total_prompt_tokens"] == 100
          and summary["total_completion_tokens"] == 50,
          "token totals reach the summary")
    check(summary["inference_latency_ms_mean"] is not None,
          "latency is aggregated into the summary")
    check("decision_wait_samples" in summary and "run_wall_clock_s" in summary,
          "summary keeps raw AWT samples + wall clock")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        check_manifest(tmp)
        check_decision_records(tmp)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
