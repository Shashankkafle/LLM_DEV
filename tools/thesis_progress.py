"""One line per thesis-arm run: simulated time vs wall time, and an ETA.

    python tools/thesis_progress.py                       # logs/thesis_llm_arm
    python tools/thesis_progress.py logs/thesis_llm_arm_smoke
    watch -n 60 python tools/thesis_progress.py

The ETA extrapolates the run's average pace so far to its horizon. For
hzreal_none and cross_1h it is an upper bound: their demand ends at 3599 s, so
the second hour has fewer vehicles, empty intersections skip the LLM call, and
the run stops early if the network fully drains (residual gridlock can prevent
that, since teleport is off).
"""
import argparse
import glob
import json
import os
from datetime import datetime, timedelta

DEFAULT_GROUP = "logs/thesis_llm_arm"


def last_step(step_log):
    """Step of the last complete line, reading only the file's tail."""
    if not os.path.exists(step_log) or os.path.getsize(step_log) == 0:
        return None
    with open(step_log, "rb") as handle:
        handle.seek(max(0, os.path.getsize(step_log) - 4096))
        lines = [line for line in handle.read().splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)["step"]
        except (ValueError, KeyError):
            continue
    return None


def hours(seconds):
    return f"{seconds / 3600:5.2f}h"


def describe_run(run_dir, now):
    manifest = json.load(open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8"))
    horizon = manifest["args"]["simulation_steps"]
    started = datetime.fromisoformat(manifest["started_at"])
    ended = manifest.get("ended_at")
    wall_s = ((datetime.fromisoformat(ended) if ended else now) - started).total_seconds()
    step = last_step(os.path.join(run_dir, "step_summaries.jsonl"))
    sim_s = 0 if step is None else step + 1

    line = (f"{os.path.basename(run_dir):<44} {manifest.get('status', '?'):<9} "
            f"sim {sim_s:5d}/{horizon}s  wall {hours(wall_s)}")
    if sim_s and wall_s:
        pace = sim_s / wall_s
        line += f"  {pace * 3600:6.0f} sim-s/wall-h"
        if not ended:
            remaining_s = (horizon - sim_s) / pace
            finish = now + timedelta(seconds=remaining_s)
            line += f"  ETA {hours(remaining_s)} ({finish:%m-%d %H:%M})"
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("group_dir", nargs="?", default=DEFAULT_GROUP)
    args = parser.parse_args()

    now = datetime.now()
    manifests = sorted(glob.glob(os.path.join(args.group_dir, "*", "run_manifest.json")))
    if not manifests:
        print(f"no runs under {args.group_dir}")
    for manifest in manifests:
        print(describe_run(os.path.dirname(manifest), now))


if __name__ == "__main__":
    main()
