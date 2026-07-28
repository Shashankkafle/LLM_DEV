"""Recompute completion_rate for runs recorded before the loaded-vehicle fix.

Runs written before 2026-07-28 undercount the denominator: MetricsRecorder only
polled getLoadedIDList() after each simulation step, so the batch of vehicles
SUMO builds while loading the route file (before the first step -- ids 0-9 on
hangzhou_real) was never counted. The rate was therefore too high, and could
print above 1.0.

This script re-derives it from fields those runs already contain:

    arrived     = total_departed_vehicles - still_running_at_end   (recorder,
                                                obstacle-filtered already)
    denominator = sumo_vehicles_loaded - obstacle vehicles         (SUMO)

SUMO counts a blockage's obstacle vehicle as both loaded and inserted (verified:
a C3 run reads loaded="2984" against a clean 2983), so it has to come back out
of the denominator to match the recorder-side numerator. The obstacle enters at
the blockage's "activated" event -- not at "obstacle_placed", which never fires
if placement stays deferred -- and only for the obstacle_vehicle method;
speed_restriction blockages add no vehicle.

Usage:
    python recompute_completion_rate.py [--logs-dir logs] [--csv out.csv]
"""
import argparse
import csv
import json
from pathlib import Path

from configurations import BLOCKAGE_EVENTS_FILENAME, FINAL_SUMMARY_FILENAME

OBSTACLE_METHOD = "obstacle_vehicle"


def load_events(run_dir):
    events_path = run_dir / BLOCKAGE_EVENTS_FILENAME
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def scored_episode_events(events):
    """Keep only the episode whose SUMO session wrote the statistics file.

    CoLight shares one event log across every training round plus the eval
    episode, tagging training events with a round number; the eval session --
    the scored one -- writes none. Fall back to the last round when every event
    carries one (a training-only log).
    """
    untagged = [e for e in events if "round" not in e]
    if untagged:
        return untagged
    rounds = [e["round"] for e in events]
    if not rounds:
        return []
    last_round = max(rounds)
    return [e for e in events if e["round"] == last_round]


def count_obstacle_vehicles(run_dir):
    """How many vehicles the blockage machinery added to SUMO this episode."""
    activated = [
        e for e in scored_episode_events(load_events(run_dir))
        if e.get("event") == "activated" and e.get("method") == OBSTACLE_METHOD
    ]
    return len({e["blockage_id"] for e in activated})


def recompute(summary, obstacle_count):
    """Corrected rate, or (None, reason) when the run lacks the fields."""
    departed = summary.get("total_departed_vehicles")
    still_running = summary.get("still_running_at_end")
    sumo_loaded = summary.get("sumo_vehicles_loaded")
    if departed is None or still_running is None:
        return None, "no recorder vehicle accounting"
    if not sumo_loaded:
        return None, "no sumo_vehicles_loaded (train-mode or pre-statistics run)"
    denominator = sumo_loaded - obstacle_count
    if denominator <= 0:
        return None, f"denominator {denominator} after removing {obstacle_count} obstacle(s)"
    return round((departed - still_running) / denominator, 4), None


def collect_rows(logs_dir):
    rows = []
    for summary_path in sorted(logs_dir.rglob(FINAL_SUMMARY_FILENAME)):
        run_dir = summary_path.parent
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            rows.append({"run": str(run_dir), "note": "unreadable final_summary.json"})
            continue
        obstacle_count = count_obstacle_vehicles(run_dir)
        corrected, reason = recompute(summary, obstacle_count)
        reported = summary.get("completion_rate")
        rows.append({
            "run": str(run_dir.relative_to(logs_dir)),
            "controller": summary.get("controller"),
            "scenario": summary.get("blockage_scenario"),
            "seed": summary.get("seed"),
            "recorder_loaded": summary.get("total_loaded_vehicles"),
            "sumo_loaded": summary.get("sumo_vehicles_loaded"),
            "obstacles": obstacle_count,
            "denominator": (summary["sumo_vehicles_loaded"] - obstacle_count
                            if summary.get("sumo_vehicles_loaded") else None),
            "arrived": (summary["total_departed_vehicles"] - summary["still_running_at_end"]
                        if summary.get("total_departed_vehicles") is not None
                        and summary.get("still_running_at_end") is not None else None),
            "completion_rate_reported": reported,
            "completion_rate_corrected": corrected,
            "delta": (round(corrected - reported, 4)
                      if corrected is not None and reported is not None else None),
            "note": reason or "",
        })
    return rows


def print_table(rows):
    header = f"{'run':52s} {'obs':>3s} {'denom':>6s} {'arrived':>7s} {'reported':>9s} {'corrected':>9s} {'delta':>8s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("completion_rate_corrected") is None:
            print(f"{row['run'][:52]:52s} {'-':>3s} {'-':>6s} {'-':>7s} "
                  f"{_fmt(row.get('completion_rate_reported')):>9s} {'skipped':>9s} "
                  f"{'':>8s}  {row['note']}")
            continue
        print(f"{row['run'][:52]:52s} {row['obstacles']:3d} {row['denominator']:6d} "
              f"{row['arrived']:7d} {_fmt(row['completion_rate_reported']):>9s} "
              f"{row['completion_rate_corrected']:9.4f} {_fmt(row['delta']):>8s}")


def print_summary(rows):
    corrected = [r for r in rows if r.get("completion_rate_corrected") is not None]
    skipped = len(rows) - len(corrected)
    changed = [r for r in corrected if r["delta"] not in (None, 0.0)]
    over_one = [r for r in corrected
                if (r["completion_rate_reported"] or 0) > 1.0]
    print(f"\n{len(rows)} runs scanned, {len(corrected)} recomputed, {skipped} skipped.")
    print(f"{len(changed)} runs change; "
          f"{len(over_one)} had a reported rate above 1.0.")
    if changed:
        worst = max(changed, key=lambda r: abs(r["delta"]))
        print(f"largest correction: {worst['delta']:+.4f} on {worst['run']}")


def _fmt(value):
    return "-" if value is None else f"{value:.4f}"


def write_csv(rows, path):
    fieldnames = ["run", "controller", "scenario", "seed", "recorder_loaded",
                  "sumo_loaded", "obstacles", "denominator", "arrived",
                  "completion_rate_reported", "completion_rate_corrected",
                  "delta", "note"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", default="logs", type=Path)
    parser.add_argument("--csv", type=Path, help="also write the full table here")
    args = parser.parse_args()

    rows = collect_rows(args.logs_dir)
    print_table(rows)
    print_summary(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
