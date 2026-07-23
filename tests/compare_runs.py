"""Compare two completed runs decision-by-decision and in aggregate.

Built to quantify how far a batched LLM run drifts from a --sequential run of the
same config/seed. The two process identical traffic until the first decision
flips (a pure fp16 batched-vs-single divergence); after that the applied phases
differ, so traffic -- and every downstream decision -- diverges by butterfly
effect. A naive whole-run diff therefore mixes the one fp16 flip with all the
trajectory divergence it causes. So this reports each separately:

  * decisions identical BEFORE the first divergence -- the clean fp16 signal
    (both runs saw identical traffic up to there); large => flips are rare.
  * overall divergence among comparable (same step+intersection) decisions --
    conflates fp16 with downstream trajectory divergence; read as "how far apart
    did they end up", not "the per-decision flip rate".
  * cadence divergence -- once traffic differs, the runs query different
    intersections at different steps.
  * aggregate final_summary metrics side by side -- the decision-relevant
    headline: if ATT / throughput / valid_rate match within the seed-to-seed
    spread you'd see anyway, the drift does not change the campaign's
    conclusions even when individual decisions differ.

    python tests/compare_runs.py <run_dir_sequential> <run_dir_batched>
"""
import json
import sys
from pathlib import Path


def load_decisions(run_dir):
    """(intersection_id, step) -> decision event. One decision per intersection
    per step, so the key is unique."""
    decisions = {}
    for path in sorted(Path(run_dir).glob("*/decisions.jsonl")):
        intersection_id = path.parent.name
        for line in path.open():
            event = json.loads(line)
            decisions[(intersection_id, event["step"])] = event
    return decisions


def action(event):
    """The control-relevant outcome: (decision_type, activated_phase)."""
    phase_action = event["phase_action"]
    return (phase_action["decision_type"], phase_action.get("activated_phase"))


def extracted_signal(event):
    return (event.get("llm_output") or {}).get("extracted_signal")


def is_llm_query(event):
    return event["phase_action"]["decision_type"] != "no_action_empty"


def load_summary(run_dir):
    path = Path(run_dir) / "final_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


AGG_FIELDS = [
    "total_decisions", "decisions_llm_queried", "decisions_no_action_empty",
    "llm_phase_decisions", "decisions_inference_error", "valid_response_rate",
    "total_completed_vehicles", "cityflow_style_att_s", "cityflow_clock_att_s",
    "cityflow_style_awt_s", "sumo_effective_att_s", "sumo_mean_trip_duration_s",
    "average_queue_length", "total_completion_tokens",
]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python tests/compare_runs.py <run_dir_A> <run_dir_B>")
    dir_a, dir_b = sys.argv[1], sys.argv[2]
    dec_a, dec_b = load_decisions(dir_a), load_decisions(dir_b)

    keys_a, keys_b = set(dec_a), set(dec_b)
    common = keys_a & keys_b
    ordered = sorted(common, key=lambda k: (k[1], k[0]))  # (step, intersection)

    differ = [k for k in ordered if action(dec_a[k]) != action(dec_b[k])]
    first_div = differ[0] if differ else None
    identical_before = (
        len([k for k in ordered if (k[1], k[0]) < (first_div[1], first_div[0])])
        if first_div else len(ordered))

    llm_both = [k for k in ordered
                if is_llm_query(dec_a[k]) and is_llm_query(dec_b[k])]
    sig_flips = [k for k in llm_both
                 if extracted_signal(dec_a[k]) != extracted_signal(dec_b[k])]

    cadence_only = (keys_a ^ keys_b)
    first_cadence = min((k[1] for k in cadence_only), default=None)

    print(f"A: {dir_a}")
    print(f"B: {dir_b}\n")
    print(f"decision points:   A={len(keys_a)}  B={len(keys_b)}  common={len(common)}")
    print(f"cadence-only keys: {len(cadence_only)} "
          f"(runs desynced once traffic diverged"
          + (f", first at step {first_cadence}" if first_cadence is not None else "")
          + ")")

    print(f"\ncomparable decisions (same step+intersection): {len(common)}")
    print(f"  agree:  {len(common) - len(differ)}")
    pct = 100 * len(differ) / max(1, len(common))
    print(f"  differ: {len(differ)}  ({pct:.1f}%)  "
          f"[fp16 flips + downstream trajectory divergence]")
    print(f"  signal flips among LLM-queried-in-both: "
          f"{len(sig_flips)}/{len(llm_both)}")

    if first_div:
        k = first_div
        print(f"\nFIRST divergence: step {k[1]}, {k[0]}")
        print(f"  A: {action(dec_a[k])}  signal={extracted_signal(dec_a[k])!r}")
        print(f"  B: {action(dec_b[k])}  signal={extracted_signal(dec_b[k])!r}")
        print(f"  decisions identical before it: {identical_before}  "
              f"<-- clean fp16 signal (identical traffic up to here)")
    else:
        print("\nno divergence: the two runs are decision-identical")

    sum_a, sum_b = load_summary(dir_a), load_summary(dir_b)
    if sum_a or sum_b:
        print("\naggregate metrics (A vs B) -- the decision-relevant comparison:")
        for field in AGG_FIELDS:
            if field not in sum_a and field not in sum_b:
                continue
            va, vb = sum_a.get(field), sum_b.get(field)
            delta = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = f"   d={vb - va:+.3g}"
            print(f"  {field:30s} A={va}  B={vb}{delta}")

    print("\ninterpretation: 'identical before first flip' is the clean fp16 "
          "sensitivity;\nthe differ% conflates fp16 with trajectory divergence. "
          "If the aggregate\nmetrics match within seed-to-seed spread, batching "
          "doesn't change conclusions.")


if __name__ == "__main__":
    main()
