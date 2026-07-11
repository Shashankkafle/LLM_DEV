"""Offline traffic-metric report from SUMO output files.


  - ATT (s): average travel time, from tripinfo `duration`.
  - AWT (s): average waiting time, from tripinfo `waitingTime` -- the cumulative
    time a vehicle's speed was below 0.1 m/s over its whole trip. (NOTE: this is
    a different quantity from CityFlow/LLMLight's "AWT", which is a time-averaged
    snapshot of current per-link waiting and is typically ~3x smaller.)
  - AQL (m): average queue length in METRES, from the per-lane `queueing_length`
    in SUMO's --queue-output, averaged over every (lane, timestep) sample.

ATT/AWT are each reported two ways:
  - "completed": only vehicles that reached their destination (arrival >= 0).
  - "all":       every vehicle in tripinfo, including those still running at the
                 horizon (written by --tripinfo-output.write-unfinished and
                 charged their time-so-far). This is the population-faithful
                 number that does not silently drop late departures.

Run on a finished run directory:
    python -m utils.cal_offline --run-dir logs/<run_dir>
or point at files directly with --tripinfo / --queue / --route / --sumo-config.
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from configurations import SUMO_TRIPINFO_FILENAME, SUMO_QUEUE_FILENAME


def _parse_tripinfo(tripinfo_file):
    """Read durations and waiting times from a SUMO tripinfo file.

    Returns (completed, all_, completed_count), where completed/all_ are dicts
    with 'durations' and 'waiting_times' lists. A trip counts as completed when
    its `arrival` attribute is >= 0 -- SUMO writes arrival="-1" for vehicles
    still running at the horizon under --tripinfo-output.write-unfinished.
    """
    completed = {"durations": [], "waiting_times": []}
    all_ = {"durations": [], "waiting_times": []}
    completed_count = 0

    for _event, elem in ET.iterparse(tripinfo_file, events=("end",)):
        if elem.tag != "tripinfo":
            continue
        try:
            duration = float(elem.get("duration", 0.0))
            waiting = float(elem.get("waitingTime", 0.0))
            arrival = float(elem.get("arrival", -1))
        except ValueError:
            elem.clear()
            continue

        all_["durations"].append(duration)
        all_["waiting_times"].append(waiting)
        if arrival >= 0:
            completed_count += 1
            completed["durations"].append(duration)
            completed["waiting_times"].append(waiting)
        elem.clear()

    return completed, all_, completed_count


def _parse_queue(queue_file):
    """Average per-lane `queueing_length` (metres) over every (lane, timestep)
    sample in a SUMO --queue-output file. Returns None if the file is absent."""
    if not queue_file or not os.path.exists(queue_file):
        return None
    total = 0.0
    count = 0
    for _event, elem in ET.iterparse(queue_file, events=("end",)):
        if elem.tag == "lane":
            try:
                total += float(elem.get("queueing_length", 0.0))
                count += 1
            except ValueError:
                pass
        elif elem.tag == "data":
            elem.clear()  # free each timestep block as we go (queue files are big)
    return (total / count) if count else None


def _route_files_from_sumocfg(sumo_config):
    """Resolve the route-file path(s) a .sumocfg references, relative to it."""
    if not sumo_config or not os.path.exists(sumo_config):
        return []
    try:
        root = ET.parse(sumo_config).getroot()
    except ET.ParseError:
        return []
    node = root.find(".//route-files")
    if node is None or not node.get("value"):
        return []
    base = Path(sumo_config).parent
    return [str(base / name.strip()) for name in node.get("value").split(",") if name.strip()]


def _count_demand(route_files):
    """Total scheduled vehicles across route file(s) (vehicle/trip/flow tags)."""
    total = 0
    for route_file in route_files:
        if not os.path.exists(route_file):
            continue
        for _event, elem in ET.iterparse(route_file, events=("end",)):
            if elem.tag in ("vehicle", "trip", "flow"):
                total += 1
            elem.clear()
    return total


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def cal_offline(run_dir=None, tripinfo_file=None, queue_file=None,
                route_files=None, sumo_config=None, log_file=None,
                episode_tag=""):
    """Compute and report  ATT/AWT/AQL from SUMO output files.

    Pass a run_dir (uses the standard filenames inside it) or explicit file
    paths. The completion-rate denominator is the scheduled demand: from
    route_files, else the route file(s) referenced by sumo_config, else a
    fallback to the tripinfo vehicle count. Returns the metrics dict and appends
    a human-readable report to log_file.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        tripinfo_file = tripinfo_file or str(run_dir / SUMO_TRIPINFO_FILENAME)
        queue_file = queue_file or str(run_dir / SUMO_QUEUE_FILENAME)
        log_file = log_file or str(run_dir / "offline_report.log")

    if not tripinfo_file or not os.path.exists(tripinfo_file):
        raise FileNotFoundError(f"tripinfo file not found: {tripinfo_file}")

    if route_files is None:
        route_files = _route_files_from_sumocfg(sumo_config)

    completed, all_, completed_count = _parse_tripinfo(tripinfo_file)
    aql_m = _parse_queue(queue_file)

    demand = _count_demand(route_files) if route_files else 0
    if demand == 0:
        demand = len(all_["durations"])  # fall back to inserted-vehicle count

    metrics = {
        "ATT_completed_s": round(_mean(completed["durations"]), 2),
        "AWT_completed_s": round(_mean(completed["waiting_times"]), 2),
        "ATT_all_s": round(_mean(all_["durations"]), 2),
        "AWT_all_s": round(_mean(all_["waiting_times"]), 2),
        "AQL_m": round(aql_m, 2) if aql_m is not None else None,
        "completed_vehicles": completed_count,
        "total_demand": demand,
        "completion_rate": round(completed_count / demand, 4) if demand else None,
    }

    if metrics["AQL_m"] is not None:
        aql_line = f"  AQL  (queue metres) : {metrics['AQL_m']:.2f} m"
    else:
        aql_line = "  AQL  (queue metres) : n/a (run with --queue-output)"

    if metrics["completion_rate"] is not None:
        comp_line = (f"  Completion          : {completed_count} / {demand} "
                     f"({100 * metrics['completion_rate']:.2f}%)")
    else:
        comp_line = "  Completion          : n/a"

    lines = [
        "",
        "=" * 72,
        "Offline metrics report" + (f" [{episode_tag}]" if episode_tag else ""),
        "=" * 72,
        f"  tripinfo : {tripinfo_file}",
        f"  queue    : {queue_file}",
        "",
        f"  ATT (completed)     : {metrics['ATT_completed_s']:.2f} s",
        f"  AWT (completed)     : {metrics['AWT_completed_s']:.2f} s",
        f"  ATT (all, w/ unfin) : {metrics['ATT_all_s']:.2f} s",
        f"  AWT (all, w/ unfin) : {metrics['AWT_all_s']:.2f} s",
        aql_line,
        comp_line,
        "=" * 72,
    ]
    report = "\n".join(lines)
    print(report)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Offline ATT/AWT/AQL report from SUMO output files."
    )
    parser.add_argument("--run-dir", help="Run dir holding tripinfo.xml / queue_output.xml.")
    parser.add_argument("--tripinfo", help="Path to tripinfo.xml (overrides --run-dir).")
    parser.add_argument("--queue", help="Path to queue_output.xml (overrides --run-dir).")
    parser.add_argument("--route", action="append", default=None,
                        help="Route file for the demand denominator (repeatable).")
    parser.add_argument("--sumo-config", help="Read route file(s) from this .sumocfg for demand.")
    parser.add_argument("--episode-tag", default="")
    args = parser.parse_args()

    if not args.run_dir and not args.tripinfo:
        parser.error("provide --run-dir or --tripinfo")

    cal_offline(
        run_dir=args.run_dir,
        tripinfo_file=args.tripinfo,
        queue_file=args.queue,
        route_files=args.route,
        sumo_config=args.sumo_config,
        episode_tag=args.episode_tag,
    )


if __name__ == "__main__":
    main()
