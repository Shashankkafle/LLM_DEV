"""Size --max_new_tokens for a chain-of-thought model from a real short run.

A thinking model spends reasoning tokens out of the same budget as its answer,
so too small a cap yields an empty completion, which the runner can only record
as a parse error and hold the current phase -- a run that looks complete but
whose controller never actually decided anything. This reads a run's decision
logs and reports the output-token distribution, so the production cap is a
measurement rather than a guess.

Usage -- first a short run at a deliberately generous cap:

    python run_matrix.py --experiment llm_real_normal --seeds 1 --steps 600 \\
        --llm_paths openrouter:<provider>/<model> \\
        --max_new_tokens 16384 --request_timeout 600

then point this at the run directory it printed:

    python tests/probe_reasoning_budget.py logs/<run_group>/<run_dir>

Read the output like this:
  * truncated (finish_reason=length) > 0        -> the cap is too small, full stop
  * empty-answer calls > 0                      -> same, the budget went to thinking
  * suggested cap                               -> p99 of total output, +50% headroom
"""
import argparse
import json
import sys
from pathlib import Path

DECISIONS_FILENAME = "decisions.jsonl"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path,
                    help="A run directory (holds one subdir per intersection).")
    ap.add_argument("--headroom", type=float, default=1.5,
                    help="Multiplier on the p99 output length for the suggested "
                         "cap. The default leaves 50%% margin for the tail this "
                         "short probe did not see.")
    return ap.parse_args()


def load_llm_decisions(run_dir):
    """Every decision that actually called the model, across all intersections.

    Empty-intersection holds never reach the LLM, so they carry no token counts
    and would drag every percentile down if counted."""
    decisions = []
    for log_file in sorted(run_dir.glob(f"*/{DECISIONS_FILENAME}")):
        for line in log_file.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("phase_action", {}).get("decision_type") == "no_action_empty":
                continue
            if event.get("metrics", {}).get("completion_tokens") is not None:
                decisions.append(event)
    return decisions


def percentile(sorted_values, fraction):
    """Nearest-rank percentile. No numpy dependency for a diagnostic script."""
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1,
                max(0, round(fraction * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize(decisions):
    completions = sorted(d["metrics"]["completion_tokens"] for d in decisions)
    reasonings = sorted(d["metrics"].get("reasoning_tokens") or 0 for d in decisions)
    truncated = [d for d in decisions
                 if d["metrics"].get("finish_reason") == "length"]
    # The failure this whole script exists to catch: budget spent thinking, so
    # nothing was left to answer with.
    empty_answers = [d for d in decisions
                     if not (d["llm_output"].get("raw_text") or "").strip()]
    unparsed = [d for d in decisions
                if d["llm_output"].get("parsing_valid") is False]
    return {
        "calls": len(decisions),
        "completions": completions,
        "reasonings": reasonings,
        "truncated": len(truncated),
        "empty_answers": len(empty_answers),
        "unparsed": len(unparsed),
    }


def print_report(stats, headroom):
    completions = stats["completions"]
    calls = stats["calls"]
    p50 = percentile(completions, 0.50)
    p99 = percentile(completions, 0.99)
    reasoning_mean = sum(stats["reasonings"]) / calls

    print(f"\nLLM calls analyzed:       {calls}")
    print(f"output tokens  mean/p50:  {sum(completions) / calls:.0f} / {p50}")
    print(f"output tokens  p99/max:   {p99} / {completions[-1]}")
    print(f"reasoning tokens mean:    {reasoning_mean:.0f} "
          f"({100 * reasoning_mean * calls / sum(completions):.0f}% of output)")
    print(f"truncated (length):       {stats['truncated']}")
    print(f"empty answers:            {stats['empty_answers']}")
    print(f"unparseable answers:      {stats['unparsed']}")

    suggested = int(p99 * headroom)
    print(f"\nsuggested --max_new_tokens: {suggested}  (p99 x {headroom})")

    if stats["truncated"] or stats["empty_answers"]:
        print("\n[FAIL] The cap used for this probe was itself too small, so the "
              "numbers above are censored -- the true tail is longer than "
              "anything measured here. Re-probe with a bigger cap before "
              "trusting the suggestion.")
        return 1
    print("\n[OK] No call hit the cap, so the distribution above is complete.")
    return 0


def main():
    args = parse_args()
    if not args.run_dir.is_dir():
        sys.exit(f"Not a directory: {args.run_dir}")
    decisions = load_llm_decisions(args.run_dir)
    if not decisions:
        sys.exit(f"No LLM decisions with token counts under {args.run_dir}. "
                 "Is this an LLM run directory (one subdir per intersection)?")
    return print_report(summarize(decisions), args.headroom)


if __name__ == "__main__":
    sys.exit(main())
