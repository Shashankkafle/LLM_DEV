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


def is_truncated(decision):
    return decision["metrics"].get("finish_reason") == "length"


def is_empty(decision):
    return not (decision["llm_output"].get("raw_text") or "").strip()


def summarize(decisions):
    """Split the failures by their cause, because they have different fixes.

    A bigger cap only helps the truncated ones. A call that stopped on its own
    with no answer, or answered without a usable <signal>, fails at any cap."""
    completions = sorted(d["metrics"]["completion_tokens"] for d in decisions)
    reasonings = sorted(d["metrics"].get("reasoning_tokens") or 0 for d in decisions)
    truncated = [d for d in decisions if is_truncated(d)]
    unparsed = [d for d in decisions
                if d["llm_output"].get("parsing_valid") is False]
    return {
        "calls": len(decisions),
        "completions": completions,
        "reasonings": reasonings,
        "truncated": len(truncated),
        "empty_answers": sum(1 for d in decisions if is_empty(d)),
        "unparsed": len(unparsed),
        # Budget exhausted mid-answer -- a bigger cap fixes exactly these.
        "truncated_empty": sum(1 for d in truncated if is_empty(d)),
        # Stopped voluntarily with nothing to say: not a budget problem.
        "empty_not_truncated": sum(1 for d in decisions
                                   if is_empty(d) and not is_truncated(d)),
        # Produced text, but no valid <signal>: a prompt/model-fit problem.
        "text_but_unparsed": sum(1 for d in unparsed if not is_empty(d)),
    }


def print_histogram(completions, buckets=12):
    """A runaway tail and a fat body look identical in a mean. They do not look
    identical here, and they call for opposite fixes."""
    width = max(1, -(-completions[-1] // buckets))
    print("\noutput-token histogram:")
    for bucket in range(buckets):
        low, high = bucket * width, (bucket + 1) * width
        count = sum(1 for c in completions if low <= c < high)
        if count:
            bar = "#" * max(1, round(40 * count / len(completions)))
            print(f"  {low:>6}-{high:<6} {count:>4}  {bar}")


def print_report(stats, headroom):
    completions = stats["completions"]
    calls = stats["calls"]
    total_output = sum(completions)
    reasoning_mean = sum(stats["reasonings"]) / calls

    print(f"\nLLM calls analyzed:       {calls}")
    print(f"output tokens  mean:      {total_output / calls:.0f}")
    for fraction in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"output tokens  p{int(fraction * 100):<2}:      "
              f"{percentile(completions, fraction)}")
    print(f"output tokens  max:       {completions[-1]}")
    print(f"reasoning tokens mean:    {reasoning_mean:.0f} "
          f"({100 * reasoning_mean * calls / total_output:.0f}% of output)")

    print(f"\ntruncated (length):       {stats['truncated']}"
          f"  -> a bigger cap fixes these")
    print(f"  of which empty:         {stats['truncated_empty']}")
    print(f"stopped, but no answer:   {stats['empty_not_truncated']}"
          f"  -> NOT a budget problem")
    print(f"answered, no <signal>:    {stats['text_but_unparsed']}"
          f"  -> NOT a budget problem")
    print(f"total unparseable:        {stats['unparsed']} "
          f"({100 * stats['unparsed'] / calls:.1f}% of calls)")

    print_histogram(completions)

    p95, p99 = percentile(completions, 0.95), percentile(completions, 0.99)
    # A body-vs-tail gap this wide means the tail is runaway reasoning, not the
    # real cost of a decision. Sizing the cap off p99 then pays for the runaway
    # on every call that wants it, instead of cutting it off.
    bimodal = stats["truncated"] and p99 >= 4 * max(1, percentile(completions, 0.50))
    if bimodal:
        print(f"\n[!] Bimodal: p50={percentile(completions, 0.50)} but "
              f"p99={p99}. The tail is runaway reasoning, not the true cost of "
              f"a decision -- raising the cap mostly buys the runaways more "
              f"room. Bound the thinking instead (--reasoning_max_tokens) and "
              f"set the cap from p95={p95}.")
        print(f"suggested: --reasoning_max_tokens {int(p95)} "
              f"--max_new_tokens {int(p95 * headroom)}")
    else:
        print(f"\nsuggested --max_new_tokens: {int(p99 * headroom)}  "
              f"(p99 x {headroom})")

    if stats["truncated"]:
        print("\n[FAIL] Calls hit the cap, so every percentile above is "
              "censored at it -- the true tail is longer than anything measured "
              "here.")
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
