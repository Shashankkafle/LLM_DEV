"""End-to-end wiring check for runner.py --blockage_scenario with a stub LLM.

Runs the real main() (real SumoEnv, PhaseHandler, MetricsRecorder, blockage
manager) with LLM_Inference replaced by a canned-answer stub, then inspects the
run directory for the blockage traces: section in in-window prompts only,
blocked flags in decisions.jsonl, blocked_lanes in step summaries, scenario in
replay meta. Writes a normal run dir under logs/ (delete it afterwards).

Needs SUMO on PATH. Run from the repo root:
    PYTHONPATH=. python tests/smoke_blockage_llm_wiring.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import runner


class StubLLM:
    def __init__(self, llm_path):
        pass

    def initialize_llm(self):
        pass

    def inference(self, prompt):
        return "analysis...<signal>ETWT</signal>"


runner.LLM_Inference = StubLLM

args = argparse.Namespace(
    test_name="blockage_wiring_smoke",
    simulation_steps=1000,
    simulation_config="simulations/single_intersection/run.sumocfg",
    llm_path="stub",
    use_gui=False,
    seed=None,
    blockage_scenario="simulations/single_intersection/scenarios/accident_single_lane.json",
    intersection_config="single_intersection",
)
runner.main(args)

run_dir = sorted(Path("logs").glob("blockage_wiring_smoke_*"))[-1]
print(f"\nInspecting {run_dir}")
failures = []


def check(cond, msg):
    print(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


decisions = [json.loads(l) for l in (run_dir / "TLS" / "decisions.jsonl").open()]
in_window = [d for d in decisions if 320 <= d["step"] < 880
             and d["llm_input"]["user_prompt"]]
# Loop step N runs sim second N+1, so the decision at loop step 299 already
# sees the t=300 activation -- the pre-window ends at loop step 298.
pre_window = [d for d in decisions if d["step"] < 299
              and d["llm_input"]["user_prompt"]]
check(in_window and all("LANE BLOCKAGE CONTEXT" in d["llm_input"]["user_prompt"]
                        for d in in_window),
      f"all {len(in_window)} in-window prompts carry the blockage section")
check(in_window and all(
    "- West approach through lane (signal ETWT), segment 3: stopped vehicle — full blockage."
    in d["llm_input"]["user_prompt"] for d in in_window),
      "in-window prompts render the exact blockage bullet")
check(pre_window and all("LANE BLOCKAGE" not in d["llm_input"]["user_prompt"]
                         for d in pre_window),
      f"all {len(pre_window)} pre-window prompts are blockage-free")
check(in_window and all(
    d["traffic_state"]["ETWT"]["West"]["lanes"]["W2TLS_0"]["blocked"] is True
    for d in in_window),
      "in-window decisions log the blocked flag in traffic_state")

steps = [json.loads(l) for l in (run_dir / "step_summaries.jsonl").open()]
check(all("blocked_lanes" in s for s in steps),
      "every step summary line has blocked_lanes")
check(steps[500]["blocked_lanes"] == ["W2TLS_0"] and steps[100]["blocked_lanes"] == [],
      "blocked_lanes time series matches the window")

meta = json.loads((run_dir / "replay_meta.json").read_text())
check(meta.get("blockage_scenario", "").endswith("accident_single_lane.json"),
      "replay meta records the scenario path")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECKS FAILED'}")
sys.exit(1 if failures else 0)
