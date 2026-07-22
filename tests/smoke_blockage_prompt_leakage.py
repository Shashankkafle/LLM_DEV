"""T8a smoke: the information firewall between the informed and uninformed
LLM blockage arms.

The L2-vs-L1 ablation attributes any performance gap to the event text alone.
That is only valid if the uninformed (--hide_blockage_info) arm's prompts carry
no trace of the blockage: not as text, not as a phantom vehicle in the counts,
and not as a changed decision cadence (the runner skips the LLM call on empty
intersections, so an unfiltered obstacle could change WHEN the model is asked,
not just what it reads).

Three same-seed stub-LLM runs on the toy net, with demand ONLY on the East
approach so no real vehicle ever drives on or brakes behind the blocked lane
W2TLS_0. With physics pinned equal by construction, any cross-run prompt
difference is leakage:

  A: no blockage scenario                    (clean world)
  B: blockage active + --hide_blockage_info  (uninformed L1 arm)
  C: blockage active, info shown             (informed L2 arm)

Checked:
  - A vs B: identical decision cadence, byte-identical user prompts for the
    whole run, and identical lane-level counts once the blocked flags (which
    belong in the record, not the prompt) are removed.
  - B vs C: identical cadence; every C prompt equals its B prompt with exactly
    the blockage section inserted, present exactly while facts are active.
  - No B user prompt ever mentions the blockage, while B's run record proves
    the blockage physically ran (obstacle_placed event, blocked_lanes step
    series, per-decision blockage facts) -- identity is meaningful, not a
    dead manager.
  - The commonsense explainer lives in the shared system prompt, so both arms
    get identical generic guidance.

Needs SUMO on PATH. Run from the repo root:
    PYTHONPATH=. python tests/smoke_blockage_prompt_leakage.py
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import runner
from configurations import (
    BLOCKAGE_EVENTS_FILENAME,
    LLM_COMMONSENSE_BLOCK,
    LLM_SYSTEM_PROMPT,
)
from utils.prompt_builder import APPROACH_BLOCKAGE_HEADER, EXIT_BLOCKAGE_HEADER

SCENARIO = "simulations/single_intersection/scenarios/accident_single_lane.json"
NET_FILE = Path("simulations/single_intersection/net.xml").resolve()
BLOCKED_LANE = "W2TLS_0"
SIM_STEPS = 1000
SEED = 7

# Demand only on the East approach: nothing ever enters the blocked West lane,
# so the obstacle cannot cause any real traffic change. Sparse enough that the
# intersection is regularly empty, exercising the no_action_empty cadence path.
EAST_ONLY_ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15"/>
    <route id="r_E2W" edges="E2TLS TLS2W"/>
    <flow id="flow_E2W" type="car" route="r_E2W" begin="0" end="3600" probability="0.08"/>
</routes>
"""

SUMOCFG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net}"/>
        <route-files value="{routes}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="false"/>
        <duration-log.statistics value="true"/>
    </report>
</configuration>
"""

failures = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


class StubLLM:
    """Fixed answer regardless of prompt, so all three runs make identical
    phase decisions and their physics stay comparable."""

    def __init__(self, llm_path):
        pass

    def initialize_llm(self):
        pass

    def inference(self, prompt):
        return "analysis...<signal>ETWT</signal>"


runner.LLM_Inference = StubLLM


def run_once(test_name, sumocfg, scenario, hide_info):
    args = argparse.Namespace(
        test_name=test_name,
        simulation_steps=SIM_STEPS,
        simulation_config=str(sumocfg),
        llm_path="stub",
        use_gui=False,
        seed=SEED,
        blockage_scenario=scenario,
        hide_blockage_info=hide_info,
        blockage_info_scope="both",
        intersection_config="single_intersection",
    )
    runner.main(args)
    return sorted(Path("logs").glob(f"{test_name}_*"))[-1]


def load_decisions(run_dir):
    path = run_dir / "TLS" / "decisions.jsonl"
    return [json.loads(line) for line in path.open()]


def cadence(decisions):
    return [(d["step"], d["intersection_id"], d["phase_action"]["decision_type"])
            for d in decisions]


def user_prompt(decision):
    return decision["llm_input"]["user_prompt"]


def first_prompt_diff(decisions_x, decisions_y):
    """Index and step of the first pair whose user prompts differ, or None."""
    for i, (dx, dy) in enumerate(zip(decisions_x, decisions_y)):
        if user_prompt(dx) != user_prompt(dy):
            return i, dx["step"]
    return None


def without_blocked_flags(node):
    """traffic_state minus the per-lane blocked flags, which legitimately
    differ between a blockage run's record and a clean run's."""
    if isinstance(node, dict):
        return {k: without_blocked_flags(v)
                for k, v in node.items() if k != "blocked"}
    return node


def strip_blockage_sections(prompt):
    """Remove the inserted blockage section(s); a sectionless prompt is
    returned unchanged. Mirrors get_prompt's assembly: sections are a pure
    insertion between the observation and 'Please answer:'."""
    starts = [prompt.find("\n" + header)
              for header in (APPROACH_BLOCKAGE_HEADER, EXIT_BLOCKAGE_HEADER)]
    starts = [i for i in starts if i != -1]
    if not starts:
        return prompt
    return prompt[:min(starts)] + prompt[prompt.index("Please answer:"):]


def signal_served_facts(decision):
    return [b for b in (decision["blockage_facts"] or [])
            if b["movement"] is not None]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        routes = Path(tmp) / "east_only.rou.xml"
        routes.write_text(EAST_ONLY_ROUTES)
        sumocfg = Path(tmp) / "leakage.sumocfg"
        sumocfg.write_text(SUMOCFG_TEMPLATE.format(net=NET_FILE,
                                                   routes=routes.resolve()))

        dir_a = run_once("t8a_leak_a_clean", sumocfg, None, hide_info=False)
        dir_b = run_once("t8a_leak_b_hidden", sumocfg, SCENARIO, hide_info=True)
        dir_c = run_once("t8a_leak_c_informed", sumocfg, SCENARIO, hide_info=False)

    print(f"\nComparing runs:\n  A {dir_a}\n  B {dir_b}\n  C {dir_c}\n")
    dec_a, dec_b, dec_c = (load_decisions(d) for d in (dir_a, dir_b, dir_c))

    # --- B's blockage really ran (otherwise every identity below is vacuous)
    events = [json.loads(line)
              for line in (dir_b / BLOCKAGE_EVENTS_FILENAME).open()]
    check(any(e["event"] == "obstacle_placed" for e in events),
          "B: obstacle physically placed (event log)")
    steps_b = [json.loads(line) for line in (dir_b / "step_summaries.jsonl").open()]
    check(any(s["blocked_lanes"] == [BLOCKED_LANE] for s in steps_b),
          "B: blocked_lanes appears in the step series")
    check(any(signal_served_facts(d) for d in dec_b),
          "B: decisions carry blockage facts in the record (hidden, not absent)")

    # --- A vs B: the physical apparatus is invisible to the uninformed channel
    check(cadence(dec_a) == cadence(dec_b),
          "A vs B: identical decision cadence (steps, intersections, types)")
    diff = first_prompt_diff(dec_a, dec_b)
    check(diff is None,
          "A vs B: user prompts byte-identical for the whole run"
          + (f" (first diff at decision {diff[0]}, step {diff[1]})" if diff else ""))
    counts_equal = all(
        without_blocked_flags(da["traffic_state"])
        == without_blocked_flags(db["traffic_state"])
        for da, db in zip(dec_a, dec_b))
    check(counts_equal,
          "A vs B: lane-level counts identical (blocked flags aside)")

    # --- B never says it, in any wording
    prompted_b = [d for d in dec_b if user_prompt(d)]
    check(prompted_b and all("block" not in user_prompt(d).lower()
                             and "collision" not in user_prompt(d).lower()
                             for d in prompted_b),
          f"B: no blockage vocabulary in any of {len(prompted_b)} user prompts")
    check(all(d["blockage_info_in_prompt"] is not True for d in dec_b),
          "B: blockage_info_in_prompt never true")

    # --- B vs C: the arms differ by exactly the blockage section
    check(cadence(dec_b) == cadence(dec_c),
          "B vs C: identical decision cadence")
    pairs = [(db, dc) for db, dc in zip(dec_b, dec_c) if user_prompt(db)]
    check(all(strip_blockage_sections(user_prompt(dc)) == user_prompt(db)
              for db, dc in pairs),
          "B vs C: every C prompt is its B prompt plus only the blockage section")
    section_matches_facts = all(
        (APPROACH_BLOCKAGE_HEADER in user_prompt(dc))
        == bool(signal_served_facts(dc))
        for _, dc in pairs)
    check(section_matches_facts,
          "C: section present exactly when blockage facts are active")
    with_section = sum(APPROACH_BLOCKAGE_HEADER in user_prompt(dc)
                       for _, dc in pairs)
    check(with_section >= 3,
          f"C: enough in-window prompted decisions to be meaningful "
          f"({with_section} carry the section)")

    # --- Both arms share the generic guidance; only event facts may differ
    check(LLM_COMMONSENSE_BLOCK in LLM_SYSTEM_PROMPT,
          "commonsense block lives in the shared system prompt (both arms)")

    if failures:
        print(f"\n{len(failures)} CHECKS FAILED -- run dirs kept for inspection")
        return 1
    for run_dir in (dir_a, dir_b, dir_c):
        shutil.rmtree(run_dir)
    print("\nALL CHECKS PASSED -- run dirs removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
