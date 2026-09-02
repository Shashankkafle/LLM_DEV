"""Live scenario-matrix verification of the event-text templates v2.1.

Runs runner.main() (real env, real wiring, stub LLM) over the S0-S4 blockage
scenarios on the hangzhou 4x4 network, in both prompt arms (L2 = informed,
L1 = --hide_blockage_info), plus a two-seed repeat of S4, then checks the
prompts logged in decisions.jsonl:

  T1  every template path renders the exact expected section, with slot
      values hand-derived from the net topology (approach, signal for
      left vs through, segment, opposite approach, K-of-M lane counts,
      distance / link length) -- and nothing renders anywhere else
  T2  a right-turn-lane blockage is suppressed at its own intersection, and
      the L1 arm never carries a section in any scenario
  T3  S4's single event record produces consistent upstream (downstream
      report) and downstream (approach report) messages: same cause, same
      full/partial severity, same physical spot
  T4  the section-0 commonsense block rides in the system prompt of BOTH
      arms (verified at the open_llm._format_prompt seam -- decisions.jsonl
      stores only the user prompt -- and verified NOT to leak into any
      logged user prompt)
  T5  the event-section text is byte-identical across two seeds of the
      same scenario

Full prompt dumps (2-3 control steps per scenario x template path) and a
REPORT.md go to logs/prompt_matrix_report_<timestamp>/.

Needs SUMO on PATH. Run from the repo root:
    PYTHONPATH=. python tests/smoke_event_text_matrix.py
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import runner
from configurations import LLM_COMMONSENSE_BLOCK, LLM_SYSTEM_PROMPT, LLM_DEFAULT_PATH


class StubLLM:
    """Replaces the real model: the run exercises every piece of wiring except
    inference itself, so prompts land in decisions.jsonl at full speed."""

    def __init__(self, llm_path, **kwargs):
        # **kwargs: build_llm passes the real backend's settings (max_new_tokens,
        # reasoning, quantization...); none of them mean anything to a stub.
        pass

    def initialize_llm(self):
        pass

    def inference(self, prompt):
        return "analysis...<signal>ETWT</signal>"

    def inference_batch(self, prompts):
        return [self.inference(p) for p in prompts]


runner.LLM_Inference = StubLLM

SUMOCFG = "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.sumocfg"
SCEN = "dataset/llm_light/Hangzhou/4_4/scenarios"
STEPS = 400
WINDOW_START = 50          # start_step in all four scenario JSONs
IN_WINDOW = WINDOW_START + 10   # decisions at/after this step must show the event
PRE_WINDOW = WINDOW_START - 3   # decisions before this step must be event-free

# (run_id, scenario_path, hide_blockage_info, seed)
RUNS = [
    ("s0_l2", None, False, None),
    ("s0_l1", None, True, None),
    ("s1_l2", f"{SCEN}/s1_full_through_west_1_1.json", False, None),
    ("s1_l1", f"{SCEN}/s1_full_through_west_1_1.json", True, None),
    ("s2_l2", f"{SCEN}/s2_partial_left_west_1_1.json", False, None),
    ("s2_l1", f"{SCEN}/s2_partial_left_west_1_1.json", True, None),
    ("s3_l2", f"{SCEN}/s3_rightturn_north_1_3.json", False, None),
    ("s3_l1", f"{SCEN}/s3_rightturn_north_1_3.json", True, None),
    ("s4_l2_seed42", f"{SCEN}/s4_through_north_1_3.json", False, 42),
    ("s4_l2_seed43", f"{SCEN}/s4_through_north_1_3.json", False, 43),
    ("s4_l1", f"{SCEN}/s4_through_north_1_3.json", True, 42),
]

APPROACH_HEADER = "LANE BLOCKAGE REPORT (approaches to this intersection)"
EXIT_HEADER = "DOWNSTREAM BLOCKAGE REPORT (roads leaving this intersection)"

# ---------------------------------------------------------------------------
# Expected sections, hand-derived from the v2.1 templates plus net topology
# (probe: lane 0 = right-turn/unsignalized, lane 1 = through, lane 2 = left;
# road_0_1_0 is 786.4 m and fringe-fed; road_1_4_3 is 572.8 m, runs
# intersection_1_4 -> intersection_1_3 heading South). Deliberately written
# out in full, NOT built by calling prompt_builder -- so template drift and
# slot bugs both fail the diff.
# ---------------------------------------------------------------------------

# S1: obstacle, road_0_1_0_1 (through, ETWT), position 100 m on 786.4 m
# lane -> segment 2 (78.64 < 100 <= 262.1).
EXPECTED_S1 = {"intersection_1_1": (
    APPROACH_HEADER + "\n"
    "The following blockage is currently active on an approach to this "
    "intersection:\n"
    "- West approach, through lane (served by signal ETWT), segment 2: "
    "collision — the lane is fully blocked. Vehicles behind the blockage "
    "in this lane cannot reach the intersection until it clears. Within "
    "signal ETWT, the West queued and approaching counts reported above "
    "include these vehicles; the East counts are unaffected by this blockage."
)}

# S2: speed restriction 0.6, road_0_1_0_2 (left-turn, ELWL), position 30 m
# -> segment 1 (30 <= 78.64).
EXPECTED_S2 = {"intersection_1_1": (
    APPROACH_HEADER + "\n"
    "The following blockage is currently active on an approach to this "
    "intersection:\n"
    "- West approach, left-turn lane (served by signal ELWL), segment 1: "
    "roadworks — the lane is partially blocked. Vehicles can pass the "
    "obstruction slowly, so this lane discharges at a reduced rate. Within "
    "signal ELWL, the West queued and approaching counts reported above "
    "include vehicles delayed behind the obstruction; the East counts are "
    "unaffected by this blockage."
)}

# S3/S4 exit side at intersection_1_4: road_1_4_3 exits South, 3 lanes, one
# blocked, 572.8 - 10 = 562.8 -> "563 m past" on a "573 m link"; through
# movement listed before the left turn.
_EXIT_1_4 = (
    EXIT_HEADER + "\n"
    "The following road leaving this intersection is blocked further "
    "downstream:\n"
    "- Road exiting toward the South: collision — 1 of 3 lanes fully "
    "blocked, located 563 m past this intersection on a 573 m link. "
    "Movements releasing vehicles onto this road: North→South through "
    "(part of signal NTST) and East→South left turn (part of signal ELWL)."
)

# S3: right-turn lane road_1_4_3_0 -> intersection_1_3 suppressed entirely;
# intersection_1_4 still reports the road as blocked downstream.
EXPECTED_S3 = {"intersection_1_3": "", "intersection_1_4": _EXIT_1_4}

# S4: through lane road_1_4_3_1 (NTST), position 10 m on 572.8 m lane ->
# segment 1 (10 <= 57.28) at intersection_1_3, the exit bullet at 1_4.
EXPECTED_S4 = {
    "intersection_1_3": (
        APPROACH_HEADER + "\n"
        "The following blockage is currently active on an approach to this "
        "intersection:\n"
        "- North approach, through lane (served by signal NTST), segment 1: "
        "collision — the lane is fully blocked. Vehicles behind the "
        "blockage in this lane cannot reach the intersection until it clears. "
        "Within signal NTST, the North queued and approaching counts reported "
        "above include these vehicles; the South counts are unaffected by "
        "this blockage."
    ),
    "intersection_1_4": _EXIT_1_4,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

report_dir = Path("logs") / f"prompt_matrix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
dump_dir = report_dir / "dumps"
dump_dir.mkdir(parents=True, exist_ok=True)

failures = []
notes = []
report_lines = []


def emit(line):
    report_lines.append(line)
    print(line.encode("ascii", "backslashreplace").decode())


def check(cond, msg):
    emit(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def note(msg):
    notes.append(msg)
    emit(f"[note] {msg}")


def run_matrix():
    """Execute all runs; returns {run_id: {tl_id: [decision dicts]}}."""
    results = {}
    for run_id, scenario, hide, seed in RUNS:
        print(f"\n=== running {run_id} (scenario={scenario}, hide={hide}, seed={seed}) ===")
        t0 = time.time()
        runner.main(argparse.Namespace(
            test_name=f"pmx_{run_id}",
            run_group=None,
            simulation_steps=STEPS,
            simulation_config=SUMOCFG,
            llm_path="stub",
            use_gui=False,
            seed=seed,
            blockage_scenario=scenario,
            hide_blockage_info=hide,
            blockage_info_scope="both",
            intersection_config="three_lane",
            # The batching/LLM-backend flags runner.main reads. Defaults only --
            # the stub never loads a model -- but they must exist or main() raises.
            sequential=False,
            max_batch_size=0,
            max_new_tokens=None,
            request_timeout=None,
            reasoning_max_tokens=None,
            reasoning="auto",
            quantization="none",
            logs_dir=None,
        ))
        run_dir = sorted(Path("logs").glob(f"pmx_{run_id}_*"))[-1]
        per_tl = {}
        for tl_dir in run_dir.iterdir():
            f = tl_dir / "decisions.jsonl"
            if f.exists():
                with f.open(encoding="utf-8") as fh:
                    per_tl[tl_dir.name] = [json.loads(line) for line in fh]
        results[run_id] = per_tl
        emit(f"run {run_id}: {run_dir} ({time.time() - t0:.0f}s, "
             f"{sum(len(v) for v in per_tl.values())} decisions)")
    return results


def prompts_of(per_tl, tl):
    return [(d["step"], d["llm_input"]["user_prompt"])
            for d in per_tl.get(tl, [])
            if d["llm_input"]["user_prompt"]]


def section_of(prompt):
    """The blockage-section block of a prompt ('' when absent): everything
    from the first section header up to 'Please answer:'."""
    starts = [i for i in (prompt.find(APPROACH_HEADER), prompt.find(EXIT_HEADER))
              if i != -1]
    if not starts:
        return ""
    return prompt[min(starts):prompt.index("Please answer:")].rstrip("\n")


def in_window(pairs):
    return [(s, p) for s, p in pairs if s >= IN_WINDOW]


def pre_window(pairs):
    return [(s, p) for s, p in pairs if s < PRE_WINDOW]


def dump(run_id, tl, pairs, tag):
    """Write up to 3 full prompts (first/middle/last control step) to files."""
    picks = pairs[:1] + ([pairs[len(pairs) // 2]] if len(pairs) > 2 else []) + \
            (pairs[-1:] if len(pairs) > 1 else [])
    for step, prompt in picks:
        path = dump_dir / f"{run_id}_{tl}_step{step}_{tag}.txt"
        path.write_text(prompt, encoding="utf-8")
    return [f"{run_id}_{tl}_step{s}_{tag}.txt" for s, _ in picks]


def check_scenario_sections(results, run_id, expected_by_tl, label):
    """Every in-window prompt at each listed TL must carry exactly the
    expected section; every other TL (and every pre-window prompt) must be
    section-free."""
    per_tl = results[run_id]
    for tl, expected in expected_by_tl.items():
        pairs = in_window(prompts_of(per_tl, tl))
        check(len(pairs) >= 2,
              f"{label}: {tl} has >=2 in-window control steps with prompts "
              f"({len(pairs)} found)")
        bad = [(s, section_of(p)) for s, p in pairs if section_of(p) != expected]
        if expected:
            check(not bad,
                  f"{label}: all {len(pairs)} in-window prompts at {tl} carry "
                  f"the exact expected section")
        else:
            check(not bad,
                  f"{label}: all {len(pairs)} in-window prompts at {tl} are "
                  f"section-free (suppressed)")
        if bad:
            s, got = bad[0]
            emit(f"       first mismatch at step {s}:")
            emit("       --- got ---")
            emit(got if got else "(no section)")
            emit("       --- expected ---")
            emit(expected if expected else "(no section)")
        if pairs:
            files = dump(run_id, tl, pairs, "in_window")
            emit(f"       dumps: {', '.join(files)}")

    for tl in per_tl:
        pairs = prompts_of(per_tl, tl)
        if tl not in expected_by_tl:
            dirty = [s for s, p in pairs if section_of(p)]
            check(not dirty,
                  f"{label}: {tl} never carries any section "
                  f"(dirty steps: {dirty[:5] if dirty else 'none'})")
        else:
            dirty = [s for s, p in pre_window(pairs) if section_of(p)]
            check(not dirty,
                  f"{label}: {tl} pre-window prompts (step < {PRE_WINDOW}) are "
                  f"section-free")


def main():
    results = run_matrix()

    # ------------------------------------------------------------------ T1
    emit("\n### T1 - one dump per scenario x template path; exact section diff")
    check_scenario_sections(results, "s1_l2", EXPECTED_S1,
                            "T1/S1 (approach full, through/ETWT, segment 2)")
    check_scenario_sections(results, "s2_l2", EXPECTED_S2,
                            "T1/S2 (approach partial, left/ELWL, segment 1)")
    check_scenario_sections(results, "s4_l2_seed42", EXPECTED_S4,
                            "T1/S4 (dual: approach at 1_3 + downstream at 1_4)")
    # S0 baseline: no section anywhere, ever.
    s0_dirty = [
        (tl, s) for tl, ds in results["s0_l2"].items()
        for s, p in prompts_of(results["s0_l2"], tl) if section_of(p)
    ]
    check(not s0_dirty, "T1/S0: no section in any prompt of the no-blockage run")
    baseline = prompts_of(results["s0_l2"], "intersection_1_1")
    if baseline:
        files = dump("s0_l2", "intersection_1_1", baseline, "baseline")
        emit(f"       baseline dumps: {', '.join(files)}")

    # ------------------------------------------------------------------ T2
    emit("\n### T2 - suppression: right-turn-lane blockage + clean L1 arm")
    check_scenario_sections(results, "s3_l2", EXPECTED_S3,
                            "T2/S3 (right-turn lane road_1_4_3_0)")
    note("S3 by design still yields a DOWNSTREAM report at intersection_1_4: "
         "the exit template counts blocked lanes on the road (storage), not "
         "signal-served lanes, and it reads identically to S4's exit bullet.")
    for run_id in ("s0_l1", "s1_l1", "s2_l1", "s3_l1", "s4_l1"):
        dirty = [
            (tl, s) for tl in results[run_id]
            for s, p in prompts_of(results[run_id], tl) if section_of(p)
        ]
        check(not dirty, f"T2/L1: {run_id} has no event section in any prompt "
                         f"at any intersection ({sum(len(v) for v in results[run_id].values())} decisions)")

    # ------------------------------------------------------------------ T3
    emit("\n### T3 - S4 dual-message consistency (one record, two controllers)")
    s4 = results["s4_l2_seed42"]
    down = [(s, section_of(p)) for s, p in in_window(prompts_of(s4, "intersection_1_3"))
            if section_of(p)]
    up = [(s, section_of(p)) for s, p in in_window(prompts_of(s4, "intersection_1_4"))
          if section_of(p)]
    check(bool(down) and bool(up),
          f"T3: both controllers report the event ({len(down)} downstream, "
          f"{len(up)} upstream messages)")
    if down and up:
        d_txt, u_txt = down[0][1], up[0][1]
        check("collision" in d_txt and "collision" in u_txt,
              "T3: cause 'collision' appears in both messages")
        check("fully blocked" in d_txt and "fully blocked" in u_txt
              and "partially" not in d_txt and "partially" not in u_txt,
              "T3: both messages agree on full (not partial) severity")
        import re
        m = re.search(r"located (\d+) m past this intersection on a (\d+) m link",
                      u_txt)
        check(bool(m), "T3: upstream message quotes distance and link length")
        if m:
            dist, link = int(m.group(1)), int(m.group(2))
            # 10 m upstream of the stop line (segment 1 <= 57.28 m) must match
            # link_length - distance within the two roundings.
            check(abs((link - dist) - 10) <= 1,
                  f"T3: upstream spot (link {link} - located {dist} = "
                  f"{link - dist} m before the stop line) matches the "
                  f"downstream 'segment 1' (blockage at 10 m)")
        check("segment 1" in d_txt,
              "T3: downstream message places the blockage in segment 1")
        d_steps = {s for s, _ in down}
        u_steps = {s for s, _ in up}
        check(bool(d_steps) and bool(u_steps)
              and min(d_steps) < max(u_steps) and min(u_steps) < max(d_steps),
              "T3: the two messages are active over overlapping control steps")

    # ------------------------------------------------------------------ T4
    emit("\n### T4 - section-0 commonsense block in BOTH arms' system prompt")
    check(LLM_SYSTEM_PROMPT.count(LLM_COMMONSENSE_BLOCK) == 1,
          "T4: LLM_SYSTEM_PROMPT contains the commonsense block exactly once")
    from models_inference.LLM.open_llm import LLM_Inference
    seam = LLM_Inference(LLM_DEFAULT_PATH)
    seam._logged_first_prompt = True
    seam.model_family = "alpaca"
    formatted = seam._format_prompt("USER_CONTENT_SENTINEL")
    check(formatted.count(LLM_COMMONSENSE_BLOCK) == 1
          and "USER_CONTENT_SENTINEL" in formatted,
          "T4: alpaca-format model input carries the block exactly once")
    try:
        from transformers import AutoTokenizer
        seam.tokenizer = AutoTokenizer.from_pretrained(seam.llm_path,
                                                       trust_remote_code=True)
        seam.model_family = "chatml"
        formatted = seam._format_prompt("USER_CONTENT_SENTINEL")
        check(formatted.count(LLM_COMMONSENSE_BLOCK) == 1
              and "USER_CONTENT_SENTINEL" in formatted,
              "T4: chatml-format model input (default Qwen tokenizer) carries "
              "the block exactly once")
    except Exception as e:
        note(f"chatml seam check skipped (tokenizer unavailable: {e})")
    note("open_llm._format_prompt is the single entry point for every "
         "inference call and takes no arm-dependent input, so the block is "
         "identical in L1 and L2, in every scenario including S0.")
    leaked = [
        (run_id, tl, s) for run_id in results for tl in results[run_id]
        for s, p in prompts_of(results[run_id], tl)
        if LLM_COMMONSENSE_BLOCK[:60] in p
    ]
    check(not leaked,
          "T4: the block never leaks into a logged USER prompt (no duplication)")

    # ------------------------------------------------------------------ T5
    emit("\n### T5 - event-section byte-identity across seeds (S4, 42 vs 43)")
    for tl in ("intersection_1_3", "intersection_1_4"):
        secs = {}
        for run_id in ("s4_l2_seed42", "s4_l2_seed43"):
            secs[run_id] = sorted({
                section_of(p)
                for _, p in in_window(prompts_of(results[run_id], tl))
                if section_of(p)
            })
        check(secs["s4_l2_seed42"] == secs["s4_l2_seed43"]
              and len(secs["s4_l2_seed42"]) == 1,
              f"T5: {tl} renders one distinct section, byte-identical across "
              f"seeds ({len(secs['s4_l2_seed42'])} vs {len(secs['s4_l2_seed43'])} variants)")

    # ------------------------------------------------------------------ report
    verdict = "ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECKS FAILED"
    emit(f"\n{verdict}")
    header = [
        "# Event-text templates v2.1 - live scenario-matrix verification",
        "",
        f"Generated by tests/smoke_event_text_matrix.py, {datetime.now().isoformat(timespec='seconds')}",
        f"Network: {SUMOCFG}; {STEPS} steps; blockage window starts at t={WINDOW_START}.",
        "Scenario JSONs: dataset/llm_light/Hangzhou/4_4/scenarios/.",
        "Full prompt dumps are in dumps/ next to this file.",
        "",
        f"## Verdict: {verdict}",
        "",
    ]
    (report_dir / "REPORT.md").write_text(
        "\n".join(header + report_lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report_dir / 'REPORT.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
