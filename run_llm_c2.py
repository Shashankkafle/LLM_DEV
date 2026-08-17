"""One trigger for the C2 LLM grid: the +text and -text arms at seeds 1-3,
preceded by the clean baseline the C2 delta is measured against.

Three run_matrix experiments, run one after the other (never concurrently --
one 14B on one GPU):

  1. llm_real_normal      clean network, seeds 1-3   (the delta's baseline)
  2. llm_real_c2_text     C2, incident text in the prompt
  3. llm_real_c2_notext   C2, incident text hidden (--hide_blockage_info)

Each arm lands in its own logs/<experiment>/ group. Reuse and skipping come
from run_matrix: a completed identity is never recomputed, so the clean phase
costs nothing if llm_real_normal seeds 1-3 already ran on this box, and
re-running this script after a crash resumes instead of restarting.

    python run_llm_c2.py                      # all three phases, seeds 1-3
    python run_llm_c2.py --dry_run            # print the plan, run nothing
    python run_llm_c2.py --arms text notext   # skip the clean baseline
    python run_llm_c2.py --seeds 1            # seed 1 first, verify, then rerun

Runs are long -- wrap the call in tmux/nohup and tee a log.

Aggregate with (one scope covering all three groups, so the C2 arms find their
clean baseline and get a delta):
    python build_results.py
"""
import argparse
import subprocess
import sys

# Phase order matters: the clean baseline runs first so a broken pipeline shows
# up on the cheapest arm.
ARM_EXPERIMENTS = {
    "clean": "llm_real_normal",
    "text": "llm_real_c2_text",
    "notext": "llm_real_c2_notext",
}
DEFAULT_ARMS = ["clean", "text", "notext"]
SEEDS = [1, 2, 3]


def matrix_cmd(experiment, seeds, dry_run, force):
    cmd = [sys.executable, "-u", "run_matrix.py", "--experiment", experiment,
           "--seeds", *[str(s) for s in seeds]]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry_run")
    return cmd


def run_arm(arm, seeds, dry_run, force):
    experiment = ARM_EXPERIMENTS[arm]
    cmd = matrix_cmd(experiment, seeds, dry_run, force)
    print(f"\n########## {arm} ({experiment}, seeds {seeds}) ##########")
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                        choices=list(ARM_EXPERIMENTS),
                        help=f"Phases to run, in order (default: {DEFAULT_ARMS}).")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Seeds for every arm (default: {SEEDS}).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute every run, ignoring completed ones. "
                             "Drop this when resuming an interrupted grid.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print each arm's plan without running anything.")
    return parser.parse_args()


def main(args):
    # An arm that fails does not stop the rest: the remaining arms are
    # independent runs, and the exit code below still reports the failure.
    exit_codes = {}
    for arm in args.arms:
        exit_codes[arm] = run_arm(arm, args.seeds, args.dry_run, args.force)

    print("\nArm exit codes:", exit_codes)
    if not args.dry_run:
        print("Aggregate with:  python build_results.py")
    return 1 if any(exit_codes.values()) else 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
