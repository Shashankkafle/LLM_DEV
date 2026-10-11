"""Collect surge_analysis.json files into one CSV, one row per
(config, controller, seed, surge), so a rate or hold sweep reads as a table.

    python tools/surge_sweep_table.py --analyses "logs/surge_family/**/surge_analysis.json" \
        --out report_sheet/surge_sweep.csv
"""
import argparse
import csv
import glob
import json
import os

COLUMNS = ["config_alias", "controller", "seed", "surge", "rate_veh_h", "mult",
           "window_start", "window_end", "hold_s", "stream_departed",
           "stream_arrived_by_window_end", "stream_arrived_by_end",
           "approach_peak_halting", "control_approach_peak_halting",
           "oversaturated", "queue_growing", "pending_growing",
           "recovery_time_s", "max_pending_in_window",
           "network_extra_arrived_in_window", "network_excess_peak_smoothed",
           "network_excess_at_end_smoothed", "network_max_pending_in_window"]


def rows_from_report(report):
    network = report["network"]
    for surge in report["surges"]:
        peak = surge.get("approach_peak_halting") or {}
        control_peak = surge.get("control_approach_peak_halting") or {}
        yield {
            "config_alias": report["config_alias"], "controller": report["controller"],
            "seed": report["seed"], "surge": surge["name"],
            "rate_veh_h": surge.get("rate_veh_h"), "mult": surge.get("mult"),
            "window_start": surge["window"][0], "window_end": surge["window"][1],
            "hold_s": surge["hold"][1] - surge["hold"][0],
            "stream_departed": surge.get("stream_departed"),
            "stream_arrived_by_window_end": surge.get("stream_arrived_by_window_end"),
            "stream_arrived_by_end": surge.get("stream_arrived_by_end"),
            "approach_peak_halting": peak.get("peak"),
            "control_approach_peak_halting": control_peak.get("peak"),
            "oversaturated": surge.get("oversaturated"),
            "queue_growing": surge.get("queue_growing"),
            "pending_growing": surge.get("pending_growing"),
            "recovery_time_s": surge.get("recovery_time_s"),
            "max_pending_in_window": surge.get("max_pending_in_window"),
            "network_extra_arrived_in_window": network["extra_arrived_in_window"],
            "network_excess_peak_smoothed": network["excess_peak_smoothed"],
            "network_excess_at_end_smoothed": network["excess_at_end_smoothed"],
            "network_max_pending_in_window": network["max_pending_in_window"],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyses", required=True, help="glob of surge_analysis.json files")
    parser.add_argument("--out", default="report_sheet/surge_sweep.csv")
    args = parser.parse_args(argv)
    rows = []
    for path in sorted(glob.glob(args.analyses, recursive=True)):
        with open(path, encoding="utf-8") as f:
            rows.extend(rows_from_report(json.load(f)))
    rows.sort(key=lambda r: (r["controller"], r["config_alias"], r["seed"] or 0, r["surge"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()
