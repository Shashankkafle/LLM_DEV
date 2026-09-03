"""Run-level audit of the C3 blockage-text prompts.

tests/test_prompt_builder.py pins the prompt TEMPLATES; this script audits the
prompts that were actually logged during runs. It reads
<run_dir>/<intersection_id>/decisions.jsonl (field llm_input.user_prompt) and
checks the C3 arms against the frozen spec:

  1  the two blockage blocks appear at exactly intersection_2_3 (approach side)
     and intersection_2_4 (downstream side), golden-exact, nowhere else
  2  they appear only while the blockage is active
  3  the approach-only arm informs 2_3 and leaves 2_4 identical to the -text arm
  4  the -text arm carries no blockage text and matches the clean run
  5  the system prompt is the frozen one and is identical across arms
  6  the "563 m past ... on a 573 m link" claim matches the network geometry

Read-only: nothing under logs/ and no template, golden, or scenario is written.

    python audit_prompts.py [--logs-dir logs] [--seed 1] [--out prompt_audit_report.md]

Run dirs are found by scanning --logs-dir for LLM run manifests and classifying
each by its blockage block; pass --text-run/--partial-run/--notext-run/
--normal-run to name them explicitly instead.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configurations import LLM_COMMONSENSE_BLOCK, LLM_SYSTEM_PROMPT

APPROACH_HEADER = "LANE BLOCKAGE REPORT (approaches to this intersection)"
EXIT_HEADER = "DOWNSTREAM BLOCKAGE REPORT (roads leaving this intersection)"

APPROACH_INTERSECTION = "intersection_2_3"
EXIT_INTERSECTION = "intersection_2_4"

DEFAULT_SCENARIO = ("dataset/llm_light/Hangzhou/4_4/scenarios/"
                    "c3_through_north_2_3_1200s.json")
DEFAULT_NET = "dataset/llm_light/Hangzhou/4_4/roadnet.net.xml"

# The goldens are written out in full rather than produced by calling
# utils/prompt_builder, so that a template edit fails this audit instead of
# silently redefining what it checks. Slot values are hand-derived from the C3
# scenario plus the net topology: road_2_4_3_1 runs intersection_2_4 ->
# intersection_2_3 heading South, 572.8 m, lane 1 of 3 (the through lane), and
# the obstacle sits 10 m upstream of 2_3's stop line (segment 1: 10 <= 57.28).
GOLDEN_APPROACH_SECTION = (
    APPROACH_HEADER + "\n"
    "The following blockage is currently active on an approach to this "
    "intersection:\n"
    "- North approach, through lane (served by signal NTST), segment 1: "
    "collision — the lane is fully blocked. Vehicles behind the blockage in "
    "this lane cannot reach the intersection until it clears. Within signal "
    "NTST, the North queued and approaching counts reported above include "
    "these vehicles; the South counts are unaffected by this blockage."
)

GOLDEN_EXIT_SECTION = (
    EXIT_HEADER + "\n"
    "The following road leaving this intersection is blocked further "
    "downstream:\n"
    "- Road exiting toward the South: collision — 1 of 3 lanes fully blocked, "
    "located 563 m past this intersection on a 573 m link. Movements "
    "releasing vehicles onto this road: North→South through (part of signal "
    "NTST) and East→South left turn (part of signal ELWL)."
)

GOLDEN_DISTANCE_CLAIM = "563 m past this intersection on a 573 m link"

ARM_LABELS = {
    "text": "+text (both reports)",
    "partial": "partial (approach-side report only)",
    "notext": "-text (--hide_blockage_info)",
    "normal": "normal conditions (no blockage)",
}

# Hand-compiled on 2026-07-27 by reading tests/test_prompt_builder.py,
# tests/smoke_event_text_matrix.py and tests/smoke_blockage_prompt_leakage.py.
# "Template" = the test pins the builder's output; "run" = the test inspects
# prompts logged by a real runner.main() run.
EXISTING_TEST_COVERAGE = [
    ("CHECK 1 - golden blockage blocks at 2_3 / 2_4, absent elsewhere",
     "PARTIAL (template + run, different scenario)",
     "test_prompt_builder.test_blockage_section_golden / "
     "test_exit_blockage_section_golden pin both block texts at template "
     "level, on hand-built fixtures. smoke_event_text_matrix T1 does the "
     "run-level equivalent (exact section diff at the two named "
     "intersections, section-free everywhere else) but for S1/S2/S3/S4 on a "
     "400-step stub-LLM run, never for C3 and never at 2_3 / 2_4. The C3 "
     "goldens themselves are pinned nowhere."),
    ("CHECK 1 - the '563 m ... 573 m link' slot values",
     "PARTIAL (template, same numbers, different lane)",
     "test_exit_blockage_section_golden pins '563 m past this intersection on "
     "a 573 m link' for EXIT_BLOCKAGE_SOUTH (road_1_4_3_0, distance_m 562.8, "
     "lane_length_m 572.8 fed in as fixture constants). smoke_event_text_"
     "matrix derives the same numbers live for road_1_4_3. Neither touches "
     "road_2_4_3_1, and no test derives the numbers from the net file."),
    ("CHECK 2 - blockage sections only inside the active window",
     "PARTIAL (run, different scenario)",
     "smoke_event_text_matrix.check_scenario_sections asserts pre-window "
     "prompts are section-free (window start 50, S1-S4). No test checks the "
     "window END, and none uses C3's [500, 1700)."),
    ("CHECK 3 - approach-only arm (--blockage_info_scope approach)",
     "NONE",
     "No test sets blockage_info_scope to anything but 'both' (verified by "
     "grep across tests/): smoke_blockage_llm_wiring, "
     "smoke_blockage_prompt_leakage, smoke_event_text_matrix, "
     "test_run_record_schema and verify_batch_loop_equivalence all hard-code "
     "'both'. There is no golden and no run-level test for the partial arm."),
    ("CHECK 4 - -text arm carries no blockage text; equals the clean run",
     "COVERED at template + run level, on the toy net",
     "smoke_blockage_prompt_leakage (T8a) is exactly this check: A-vs-B "
     "byte-identical user prompts for a whole run, no blockage vocabulary in "
     "B, cadence equality. But it runs on simulations/single_intersection "
     "with demand engineered so the obstacle cannot perturb traffic. "
     "smoke_event_text_matrix T2 adds 'L1 arm never carries a section' on "
     "hangzhou 4x4. Neither is C3, and neither faces the real case where the "
     "blockage genuinely changes traffic."),
    ("CHECK 5 - system prompt frozen and identical across arms",
     "COVERED at template level; PARTIAL at run level",
     "test_prompt_builder.test_system_prompt_is_pinned and "
     "test_commonsense_block_is_pinned pin both strings verbatim. "
     "smoke_event_text_matrix T4 checks the http_llm._format_prompt seam and "
     "that the block never leaks into a logged user prompt. Neither reads "
     "back what a real run recorded in run_manifest.json (llm.system_prompt / "
     "llm.example_formatted_prompt)."),
    ("CHECK 6 - position semantics vs. the SUMO network",
     "PARTIAL (run, different lane)",
     "smoke_event_text_matrix T3 checks link_length - distance == 10 within "
     "rounding for road_1_4_3, from the prompt text alone. No test opens the "
     "net file, and none covers road_2_4_3_1."),
]


# --- loading -----------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifest(run_dir):
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def arm_of(manifest):
    """Which prompt arm a manifest describes, or None if it is not one of ours."""
    if manifest.get("controller") != "llm":
        return None
    blockage = manifest.get("blockage")
    if blockage is None:
        return "normal"
    if not str(blockage.get("scenario_name") or "").startswith("c3"):
        return None
    if blockage.get("hide_blockage_info"):
        return "notext"
    if blockage.get("blockage_info_scope") == "approach":
        return "partial"
    return "text"


def discover_runs(logs_dir, seed):
    """Newest matching run dir per arm, keyed by arm name."""
    found = {}
    for manifest_path in sorted(Path(logs_dir).rglob("run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arm = arm_of(manifest)
        if arm is None:
            continue
        if seed is not None and manifest.get("environment", {}).get("seed") != seed:
            continue
        found[arm] = manifest_path.parent
    return found


def load_prompts(run_dir):
    """{intersection_id: [(step, prompt_or_None, facts_active), ...]}."""
    per_intersection = {}
    for decisions_path in sorted(Path(run_dir).glob("*/decisions.jsonl")):
        rows = []
        for decision in load_jsonl(decisions_path):
            facts = (decision.get("blockage_facts") or []) + \
                    (decision.get("exit_blockage_facts") or [])
            rows.append((decision["step"],
                         (decision.get("llm_input") or {}).get("user_prompt"),
                         bool(facts)))
        per_intersection[decisions_path.parent.name] = rows
    return per_intersection


def prompted(rows):
    return [(step, prompt) for step, prompt, _ in rows if prompt]


# --- prompt dissection -------------------------------------------------------

def section_of(prompt):
    """The blockage-report block of a prompt, '' when there is none."""
    starts = [i for i in (prompt.find(APPROACH_HEADER), prompt.find(EXIT_HEADER))
              if i != -1]
    if not starts:
        return ""
    return prompt[min(starts):prompt.index("Please answer:")].rstrip("\n")


def structure_of(prompt):
    """The prompt with every number masked: same sections and wording, traffic
    counts ignored."""
    return re.sub(r"\d+", "#", prompt)


def insertion_point_is_clean(prompt):
    """No blockage section, and 'Please answer:' still directly follows the
    observation block across the base prompt's single newline -- which is what
    get_prompt emits when there is nothing to insert."""
    index = prompt.find("Please answer:")
    return (index != -1 and section_of(prompt) == ""
            and prompt[index - 1] == "\n" and prompt[index - 2] != "\n")


def first_divergence(got, expected, context=60):
    """(index, got_window, expected_window) at the first differing character."""
    limit = min(len(got), len(expected))
    index = next((i for i in range(limit) if got[i] != expected[i]), limit)
    start = max(0, index - context)
    return index, got[start:index + context], expected[start:index + context]


def divergence_block(got, expected, label):
    index, got_window, expected_window = first_divergence(got, expected)
    return [
        f"First divergence in {label} at character {index}:",
        "",
        "```",
        f"got      ...{got_window}...",
        f"expected ...{expected_window}...",
        "```",
    ]


# --- window ------------------------------------------------------------------

def load_window(scenario_path):
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    blockage = scenario["blockages"][0]
    return blockage["start_step"], blockage["end_step"], blockage


def in_window(step, start, end):
    """A decision logged at loop index `step` was taken after the simulation
    advanced to second step+1, which is the clock BlockageManager.step() ticks
    on -- so the blockage is active for that decision iff start <= step+1 < end.
    """
    return start <= step + 1 < end


# --- checks ------------------------------------------------------------------

class Result:
    def __init__(self, name):
        self.name = name
        self.status = "PASS"
        self.lines = []

    def fail(self, message):
        self.status = "FAIL"
        self.lines.append(f"- FAIL: {message}")

    def no_data(self, message):
        self.status = "NO DATA"
        self.lines.append(f"- NO DATA: {message}")

    def note(self, message):
        self.lines.append(f"- {message}")


def check_1_text_arm(runs, prompts, window):
    result = Result("CHECK 1 - golden blockage blocks in the +text run")
    if "text" not in runs:
        result.no_data("no +text run dir available")
        return result
    start, end, _ = window
    per_intersection = prompts["text"]

    for intersection, golden in ((APPROACH_INTERSECTION, GOLDEN_APPROACH_SECTION),
                                 (EXIT_INTERSECTION, GOLDEN_EXIT_SECTION)):
        rows = per_intersection.get(intersection)
        if rows is None:
            result.fail(f"{intersection} has no decisions.jsonl")
            continue
        active = [(step, prompt) for step, prompt in prompted(rows)
                  if in_window(step, start, end)]
        if not active:
            result.fail(f"{intersection}: no prompted decisions inside "
                        f"[{start}, {end})")
            continue
        bad = [(step, section_of(prompt)) for step, prompt in active
               if section_of(prompt) != golden]
        if bad:
            step, got = bad[0]
            result.fail(f"{intersection}: {len(bad)} of {len(active)} in-window "
                        f"prompts do not match the golden block")
            result.lines += divergence_block(
                got, golden, f"{intersection} step {step}")
        else:
            result.note(f"{intersection}: all {len(active)} in-window prompts "
                        f"match the golden block exactly")

    exit_rows = per_intersection.get(EXIT_INTERSECTION, [])
    exit_active = [prompt for step, prompt in prompted(exit_rows)
                   if in_window(step, start, end)]
    missing_claim = [p for p in exit_active if GOLDEN_DISTANCE_CLAIM not in p]
    if exit_active and not missing_claim:
        result.note(f'all {len(exit_active)} downstream prompts contain '
                    f'"{GOLDEN_DISTANCE_CLAIM}"')
    elif exit_active:
        result.fail(f'{len(missing_claim)} downstream prompts lack '
                    f'"{GOLDEN_DISTANCE_CLAIM}"')

    others = [i for i in sorted(per_intersection)
              if i not in (APPROACH_INTERSECTION, EXIT_INTERSECTION)]
    dirty = [(i, step) for i in others
             for step, prompt in prompted(per_intersection[i])
             if section_of(prompt)]
    if dirty:
        result.fail(f"blockage text at {len(dirty)} prompts outside the two "
                    f"named intersections, first at {dirty[0][0]} step {dirty[0][1]}")
    else:
        total = sum(len(prompted(per_intersection[i])) for i in others)
        result.note(f"the other {len(others)} intersections carry no blockage "
                    f"text in any of their {total} prompts")
    return result


def check_2_window(runs, prompts, window):
    result = Result("CHECK 2 - window discipline in the +text run")
    if "text" not in runs:
        result.no_data("no +text run dir available")
        return result
    start, end, _ = window
    result.note(f"Mapping used: a decision logged with step index s was taken "
                f"after the simulation advanced to second s+1 (runner_common's "
                f"loop calls env.step() first, and env.step() ticks "
                f"BlockageManager with traci's clock), so the blockage is "
                f"active for that decision iff {start} <= s+1 < {end}, i.e. "
                f"step index {start - 1}..{end - 2}.")

    for intersection in (APPROACH_INTERSECTION, EXIT_INTERSECTION):
        rows = prompts["text"].get(intersection, [])
        early = [s for s, p in prompted(rows)
                 if not in_window(s, start, end) and section_of(p)]
        missing = [s for s, p in prompted(rows)
                   if in_window(s, start, end) and not section_of(p)]
        if early:
            result.fail(f"{intersection}: blockage text outside the window at "
                        f"{len(early)} steps, first at step {early[0]}")
        if missing:
            result.fail(f"{intersection}: blockage text missing inside the "
                        f"window at {len(missing)} steps, first at step "
                        f"{missing[0]}")
        if not early and not missing:
            inside = [s for s, p in prompted(rows) if in_window(s, start, end)]
            outside = [s for s, p in prompted(rows)
                       if not in_window(s, start, end)]
            result.note(f"{intersection}: {len(inside)} in-window prompts all "
                        f"carry the block (steps {min(inside)}..{max(inside)}), "
                        f"{len(outside)} out-of-window prompts all clean")

    mismatched = [(i, s) for i, rows in prompts["text"].items()
                  for s, p, facts in rows
                  if p and bool(section_of(p)) != facts]
    if mismatched:
        result.fail(f"section presence disagrees with the run's own recorded "
                    f"blockage facts at {len(mismatched)} prompts, first "
                    f"{mismatched[0]}")
    else:
        result.note("section presence agrees with the per-decision "
                    "blockage_facts the run recorded independently")
    return result


def check_3_partial_arm(runs, prompts, window):
    result = Result("CHECK 3 - approach-only run")
    if "partial" not in runs:
        result.no_data("no partial (approach-only) run dir available")
        return result
    start, end, _ = window
    per_intersection = prompts["partial"]

    rows = per_intersection.get(APPROACH_INTERSECTION, [])
    active = [(s, p) for s, p in prompted(rows) if in_window(s, start, end)]
    bad = [(s, section_of(p)) for s, p in active
           if section_of(p) != GOLDEN_APPROACH_SECTION]
    if not active:
        result.fail(f"{APPROACH_INTERSECTION}: no in-window prompted decisions")
    elif bad:
        step, got = bad[0]
        result.fail(f"{APPROACH_INTERSECTION}: {len(bad)} of {len(active)} "
                    f"in-window prompts do not match the golden approach block")
        result.lines += divergence_block(got, GOLDEN_APPROACH_SECTION,
                                         f"{APPROACH_INTERSECTION} step {step}")
    else:
        result.note(f"{APPROACH_INTERSECTION}: all {len(active)} in-window "
                    f"prompts match the golden approach block exactly")

    exit_rows = per_intersection.get(EXIT_INTERSECTION, [])
    leaked = [s for s, p in prompted(exit_rows) if section_of(p)]
    if leaked:
        result.fail(f"{EXIT_INTERSECTION}: downstream block present at "
                    f"{len(leaked)} prompts, first at step {leaked[0]}")
    else:
        result.note(f"{EXIT_INTERSECTION}: no blockage text in any of "
                    f"{len(prompted(exit_rows))} prompts")

    if "notext" not in runs:
        result.note("cannot compare 2_4 against the -text run: no -text run dir")
        return result
    compare_at_intersection(result, prompts["partial"], prompts["notext"],
                            EXIT_INTERSECTION, "partial", "-text")
    return result


def check_4_notext_arm(runs, prompts):
    result = Result("CHECK 4 - -text run")
    if "notext" not in runs:
        result.no_data("no -text run dir available")
        return result
    dirty = [(i, s) for i, rows in prompts["notext"].items()
             for s, p in prompted(rows) if section_of(p)]
    total = sum(len(prompted(rows)) for rows in prompts["notext"].values())
    if dirty:
        result.fail(f"blockage text in {len(dirty)} prompts, first at "
                    f"{dirty[0][0]} step {dirty[0][1]}")
    else:
        result.note(f"no blockage text in any of {total} prompts across "
                    f"{len(prompts['notext'])} intersections")

    vocabulary = [(i, s) for i, rows in prompts["notext"].items()
                  for s, p in prompted(rows)
                  if "blockage" in p.lower() or "collision" in p.lower()]
    if vocabulary:
        result.fail(f"blockage vocabulary leaked into {len(vocabulary)} "
                    f"prompts, first at {vocabulary[0][0]} step {vocabulary[0][1]}")
    else:
        result.note("no blockage vocabulary ('blockage', 'collision') anywhere")

    if "normal" not in runs:
        result.no_data("no normal-conditions run dir: cannot compare prompts")
        return result
    for intersection in sorted(prompts["notext"]):
        compare_at_intersection(result, prompts["notext"], prompts["normal"],
                                intersection, "-text", "normal",
                                quiet_on_success=True)
    summarize_comparison(result, prompts["notext"], prompts["normal"],
                         "-text", "normal")
    return result


def matched_steps(rows_a, rows_b):
    a = {s: p for s, p in prompted(rows_a)}
    b = {s: p for s, p in prompted(rows_b)}
    return [(s, a[s], b[s]) for s in sorted(set(a) & set(b))]


def compare_at_intersection(result, prompts_a, prompts_b, intersection,
                            label_a, label_b, quiet_on_success=False):
    pairs = matched_steps(prompts_a.get(intersection, []),
                          prompts_b.get(intersection, []))
    if not pairs:
        result.fail(f"{intersection}: no decision steps shared by the {label_a} "
                    f"and {label_b} runs")
        return
    differing = [(s, a, b) for s, a, b in pairs if a != b]
    if not differing:
        if not quiet_on_success:
            result.note(f"{intersection}: all {len(pairs)} matched-step prompts "
                        f"are byte-identical between {label_a} and {label_b}")
        return
    structural = [(s, a, b) for s, a, b in differing
                  if structure_of(a) != structure_of(b)]
    if structural:
        step, got, expected = structural[0]
        result.fail(f"{intersection}: {len(structural)} of {len(pairs)} "
                    f"matched-step prompts differ structurally between "
                    f"{label_a} and {label_b} (not just in traffic counts)")
        result.lines += divergence_block(
            structure_of(got), structure_of(expected),
            f"{intersection} step {step} (digits masked)")
    elif not quiet_on_success:
        result.note(f"{intersection}: {len(differing)} of {len(pairs)} "
                    f"matched-step prompts differ, all only in traffic counts")


def summarize_comparison(result, prompts_a, prompts_b, label_a, label_b):
    pairs = [pair for intersection in sorted(prompts_a)
             for pair in matched_steps(prompts_a.get(intersection, []),
                                       prompts_b.get(intersection, []))]
    if not pairs:
        return
    identical = sum(1 for _, a, b in pairs if a == b)
    structural = sum(1 for _, a, b in pairs
                     if structure_of(a) != structure_of(b))
    unclean = sum(1 for _, a, _ in pairs if not insertion_point_is_clean(a))
    if identical == len(pairs):
        result.note(f"comparison possible: BYTE-IDENTICAL. All {len(pairs)} "
                    f"matched-step prompts across all intersections are equal "
                    f"between {label_a} and {label_b}")
    else:
        result.note(f"comparison possible: STRUCTURAL fallback. "
                    f"{identical}/{len(pairs)} matched-step prompts are "
                    f"byte-identical; the rest differ because the blockage "
                    f"changed traffic. {len(pairs) - structural}/{len(pairs)} "
                    f"are identical once digits are masked "
                    f"({structural} differ structurally)")
    if unclean:
        result.fail(f"{unclean} {label_a} prompts have a disturbed insertion "
                    f"point ('Please answer:' does not directly follow the "
                    f"observation block)")
    else:
        result.note(f"insertion point 'Please answer:' untouched in all "
                    f"{len(pairs)} {label_a} prompts")


def check_5_system_prompt(runs, manifests, prompts):
    result = Result("CHECK 5 - system prompt")
    result.note("decisions.jsonl stores only the user prompt; the system "
                "prompt a run used is recorded in run_manifest.json under "
                "llm.system_prompt, with llm.example_formatted_prompt holding "
                "the fully templated first prompt.")
    recorded = {}
    for arm in ("text", "partial", "notext"):
        manifest = manifests.get(arm)
        if manifest is None:
            continue
        llm = manifest.get("llm")
        if not llm or "system_prompt" not in llm:
            result.fail(f"{arm}: run manifest records no llm.system_prompt")
            continue
        recorded[arm] = llm["system_prompt"]

    if not recorded:
        result.no_data("no run manifest records a system prompt")
        return result

    for arm, system_prompt in recorded.items():
        if system_prompt != LLM_SYSTEM_PROMPT:
            result.fail(f"{arm}: recorded system prompt differs from "
                        f"configurations.LLM_SYSTEM_PROMPT")
            result.lines += divergence_block(system_prompt, LLM_SYSTEM_PROMPT,
                                             f"{arm} system prompt")
        elif system_prompt.count(LLM_COMMONSENSE_BLOCK) != 1:
            result.fail(f"{arm}: LLM_COMMONSENSE_BLOCK appears "
                        f"{system_prompt.count(LLM_COMMONSENSE_BLOCK)} times "
                        f"in the system prompt (expected exactly once)")
        else:
            result.note(f"{arm}: system prompt matches "
                        f"configurations.LLM_SYSTEM_PROMPT verbatim and "
                        f"carries LLM_COMMONSENSE_BLOCK exactly once")

    if len(set(recorded.values())) > 1:
        result.fail(f"the {len(recorded)} arms did not use the same system "
                    f"prompt ({len(set(recorded.values()))} distinct values)")
    elif len(recorded) > 1:
        result.note(f"all {len(recorded)} arms used one identical system prompt")
    else:
        result.note("only one arm's manifest was available, so cross-arm "
                    "identity could not be checked")

    for arm, manifest in manifests.items():
        formatted = (manifest.get("llm") or {}).get("example_formatted_prompt")
        if formatted is None:
            result.note(f"{arm}: manifest has no example_formatted_prompt "
                        f"(captured only after a successful inference)")
        elif formatted.count(LLM_COMMONSENSE_BLOCK) != 1:
            result.fail(f"{arm}: the templated prompt carries "
                        f"LLM_COMMONSENSE_BLOCK "
                        f"{formatted.count(LLM_COMMONSENSE_BLOCK)} times")
        else:
            result.note(f"{arm}: the templated prompt actually sent carries "
                        f"the commonsense block exactly once")

    leaked = [(arm, i, s) for arm, per_intersection in prompts.items()
              for i, rows in per_intersection.items()
              for s, p in prompted(rows) if LLM_COMMONSENSE_BLOCK[:60] in p]
    if leaked:
        result.fail(f"the commonsense block leaked into {len(leaked)} user "
                    f"prompts, first {leaked[0]}")
    elif prompts:
        result.note("the commonsense block never appears in a logged user "
                    "prompt (no duplication between system and user turns)")
    return result


def check_6_position_semantics(net_path, scenario_path, event_logs):
    result = Result("CHECK 6 - position semantics")
    start, end, blockage = load_window(scenario_path)
    lane_id = blockage["lane_id"]
    edge_id = lane_id.rsplit("_", 1)[0]

    root = ET.parse(net_path).getroot()
    edge = root.find(f"./edge[@id='{edge_id}']")
    if edge is None:
        result.fail(f"edge {edge_id} not found in {net_path}")
        return result
    lane = edge.find(f"./lane[@id='{lane_id}']")
    if lane is None:
        result.fail(f"lane {lane_id} not found in {net_path}")
        return result

    length = float(lane.get("length"))
    lane_count = len(edge.findall("./lane"))
    from_node, to_node = edge.get("from"), edge.get("to")
    position = blockage["position"]
    distance_from_upstream = length - position

    result.note(f"Net: {net_path}")
    result.note(f"Edge {edge_id} runs {from_node} -> {to_node}; "
                f"{lane_count} lanes; lane {lane_id} is index "
                f"{lane.get('index')}, length {length:.2f} m")
    result.note(
        "Convention found: a scenario's `position` is metres UPSTREAM of the "
        "STOP LINE, not metres from the lane start. "
        "utils/blockage_manager._lane_position_from_stopline converts it for "
        "TraCI as lane_pos = lane_length - position, and "
        "SumoEnv.describe_exit_blockages reports distance_m = lane_length - "
        "position as the distance from the UPSTREAM intersection. So position "
        f"{position} m puts the obstacle at SUMO lane position "
        f"{distance_from_upstream:.1f} m (measured from the lane start at "
        f"{from_node}), i.e. {position:.0f} m before the stop line at "
        f"{to_node} and {distance_from_upstream:.1f} m past {from_node}.")

    if to_node != APPROACH_INTERSECTION or from_node != EXIT_INTERSECTION:
        result.fail(f"expected the blocked lane to run {EXIT_INTERSECTION} -> "
                    f"{APPROACH_INTERSECTION}, found {from_node} -> {to_node}")

    claim = (f"{distance_from_upstream:.0f} m past this intersection on a "
             f"{length:.0f} m link")
    if claim == GOLDEN_DISTANCE_CLAIM:
        result.note(f'net geometry reproduces the golden claim exactly: '
                    f'"{claim}"')
    else:
        result.fail(f'net geometry yields "{claim}", golden says '
                    f'"{GOLDEN_DISTANCE_CLAIM}"')

    if lane_count != 3:
        result.fail(f"golden says '1 of 3 lanes'; the edge has {lane_count} lanes")
    else:
        result.note("golden's '1 of 3 lanes' matches the edge's lane count")

    segment_1_limit = length / 10
    if position <= segment_1_limit:
        result.note(f"segment 1 boundary is length/10 = "
                    f"{segment_1_limit:.2f} m, so position {position} m is in "
                    f"segment 1 -- matching the golden approach block")
    else:
        result.fail(f"position {position} m is past the segment 1 boundary "
                    f"({segment_1_limit:.2f} m), but the golden says segment 1")

    placements = []
    for label, path in event_logs:
        for event in load_jsonl(path):
            if event.get("event") == "obstacle_placed" and event["lane_id"] == lane_id:
                placements.append((label, event["sim_time"],
                                   event["lane_position_m"]))
    if not placements:
        result.note("no run output records the obstacle's actual placement; "
                    "the geometry check above stands on the net file alone")
    for label, sim_time, lane_position in placements:
        if abs(lane_position - distance_from_upstream) <= 0.05:
            result.note(f"cross-check: {label} froze the obstacle at SUMO lane "
                        f"position {lane_position} m at t={sim_time}, matching "
                        f"lane_length - position = "
                        f"{distance_from_upstream:.1f} m")
        else:
            result.fail(f"cross-check: {label} froze the obstacle at "
                        f"{lane_position} m, expected "
                        f"{distance_from_upstream:.1f} m")
        if sim_time > start:
            result.note(f"note: in {label} placement was deferred to t="
                        f"{sim_time} (spot occupied), while the blockage "
                        f"counts as active -- and is reported in prompts -- "
                        f"from t={start}")
    return result


# --- report ------------------------------------------------------------------

def model_of(manifest):
    llm = manifest.get("llm") or {}
    path = llm.get("llm_path_arg") or manifest.get("args", {}).get("llm_path")
    family = llm.get("model_family")
    dtype = llm.get("torch_dtype")
    details = ", ".join(p for p in (family, dtype) if p)
    return f"`{path}`" + (f" ({details})" if details else "")


def render_report(runs, manifests, results, window, args):
    start, end, blockage = window
    lines = [
        "# C3 prompt audit",
        "",
        f"Generated by `audit_prompts.py`, "
        f"{datetime.now().isoformat(timespec='seconds')}.",
        f"Scenario `{args.scenario}`: obstacle on `{blockage['lane_id']}` at "
        f"position {blockage['position']}, active steps [{start}, {end}).",
        "",
        "## Runs audited",
        "",
    ]
    if runs:
        lines += ["| Arm | Run dir | Seed | Model | Steps with prompts |",
                  "|---|---|---|---|---|"]
        for arm in ("text", "partial", "notext", "normal"):
            if arm not in runs:
                continue
            manifest = manifests[arm]
            seed = manifest.get("environment", {}).get("seed")
            lines.append(
                f"| {ARM_LABELS[arm]} | `{runs[arm]}` | {seed} | "
                f"{model_of(manifest)} | "
                f"{manifest.get('environment', {}).get('simulation_steps')} |")
    missing = [ARM_LABELS[arm] for arm in ("text", "partial", "notext", "normal")
               if arm not in runs]
    if missing:
        lines += ["",
                  f"**Missing: {', '.join(missing)}.** Searched "
                  f"`{args.logs_dir}` for run manifests with "
                  f"`controller == \"llm\"` and a `c3*` blockage scenario"
                  + (f" at seed {args.seed}" if args.seed is not None else "")
                  + ". Every check over a missing run reports NO DATA -- that "
                    "is an absence of evidence, not a pass. Point the audit at "
                    "the run dirs with `--logs-dir`, `--seed`, or the explicit "
                    "`--text-run` / `--partial-run` / `--notext-run` / "
                    "`--normal-run` flags."]
    lines += ["", "## Existing test coverage", "",
              "How much of each check the committed tests already pin. "
              "Template-level coverage constrains the builder, not the prompts "
              "a run actually emitted, so it reduces but does not remove the "
              "need for the run-level check.", "",
              "| Check | Existing coverage | Detail |", "|---|---|---|"]
    for check, verdict, detail in EXISTING_TEST_COVERAGE:
        lines.append(f"| {check} | {verdict} | {detail} |")

    lines += ["", "## Results", "",
              "| Check | Verdict |", "|---|---|"]
    for result in results:
        lines.append(f"| {result.name} | **{result.status}** |")
    lines.append("")
    for result in results:
        lines += [f"### {result.name} - {result.status}", ""] + result.lines + [""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--net-file", default=DEFAULT_NET)
    parser.add_argument("--out", default="prompt_audit_report.md")
    for arm in ("text", "partial", "notext", "normal"):
        parser.add_argument(f"--{arm}-run", default=None,
                            help=f"run dir for the {ARM_LABELS[arm]} arm")
    args = parser.parse_args()

    runs = discover_runs(args.logs_dir, args.seed)
    for arm in ("text", "partial", "notext", "normal"):
        override = getattr(args, f"{arm}_run")
        if override:
            runs[arm] = Path(override)

    manifests = {arm: read_manifest(run_dir) for arm, run_dir in runs.items()}
    prompts = {arm: load_prompts(run_dir) for arm, run_dir in runs.items()}
    window = load_window(args.scenario)

    event_logs = [(f"{ARM_LABELS[arm]} run", Path(run_dir) / "blockage_events.jsonl")
                  for arm, run_dir in runs.items()
                  if (Path(run_dir) / "blockage_events.jsonl").exists()]
    if not event_logs:
        event_logs = [(str(path.parent), path) for path
                      in sorted(Path(args.logs_dir).rglob("blockage_events.jsonl"))
                      if any(e.get("lane_id") == "road_2_4_3_1"
                             for e in load_jsonl(path))]

    results = [
        check_1_text_arm(runs, prompts, window),
        check_2_window(runs, prompts, window),
        check_3_partial_arm(runs, prompts, window),
        check_4_notext_arm(runs, prompts),
        check_5_system_prompt(runs, manifests, prompts),
        check_6_position_semantics(args.net_file, args.scenario, event_logs),
    ]

    Path(args.out).write_text(
        render_report(runs, manifests, results, window, args), encoding="utf-8")
    for result in results:
        print(f"[{result.status:7s}] {result.name}")
    print(f"\nReport: {args.out}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
