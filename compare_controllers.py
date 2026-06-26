"""
compare_controllers.py

Collects several runs' final_summary.json into one comparison table so
controllers (LLM, FixedTime, MaxPressure, CoLight) can be ranked on the same
scenario and the same population-faithful metrics.

The headline metric is cityflow_style_att_s (mean travel time counting in-flight
vehicles at the horizon, as CityFlow does). Throughput, never-inserted, and
teleports are shown alongside because under an oversaturated scenario a low ATT
can hide a controller that simply admits fewer vehicles.

Usage:
    # bare paths (label taken from the run-dir name) or label=path overrides:
    python compare_controllers.py \
        LLM=/path/to/reruns/llm_run/final_summary.json \
        FixedTime=logs/fixedtime_hz4x4_.../final_summary.json \
        MaxPressure=logs/maxpressure_hz4x4_.../

A path may point at a final_summary.json or at the run directory containing one.
"""
import argparse
import json
from pathlib import Path

# (json key, column header, width). None value renders as "-".
COLUMNS = [
    ("cityflow_style_att_s", "CityFlow-ATT", 12),
    ("cityflow_style_awt_s", "CityFlow-AWT", 12),
    ("sumo_vehicles_finished", "finished", 9),
    ("sumo_vehicles_not_inserted", "not-ins", 8),
    ("sumo_teleports_total", "teleport", 9),
    ("average_queue_length", "mean-queue", 11),
    ("sumo_effective_att_s", "eff-ATT", 9),
]


def resolve_summary_path(raw_path):
    """Accept either a final_summary.json or a directory containing one."""
    path = Path(raw_path)
    if path.is_dir():
        path = path / "final_summary.json"
    return path


def load_runs(specs):
    """specs: list of "label=path" or "path". Returns [(label, summary_dict)]."""
    runs = []
    for spec in specs:
        if "=" in spec:
            label, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            label = None
        path = resolve_summary_path(raw_path)
        if not path.exists():
            print(f"WARNING: skipping missing summary: {path}")
            continue
        with open(path) as f:
            summary = json.load(f)
        if label is None:
            # Fall back to the run-dir name (parent of final_summary.json).
            label = path.parent.name
        runs.append((label, summary))
    return runs


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def print_table(runs):
    label_width = max([len("controller")] + [len(label) for label, _ in runs])
    header = "controller".ljust(label_width)
    for _key, head, width in COLUMNS:
        header += "  " + head.rjust(width)
    print(header)
    print("-" * len(header))
    for label, summary in runs:
        row = label.ljust(label_width)
        for key, _head, width in COLUMNS:
            row += "  " + format_value(summary.get(key)).rjust(width)
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+",
                        help="One or more 'label=path' or 'path' to a final_summary.json or run dir.")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print("No summaries found.")
        return
    print_table(runs)


if __name__ == "__main__":
    main()
