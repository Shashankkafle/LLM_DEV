"""Evaluation metrics of finished runs, derived offline from the run folder
(LOGGING_CHECKLIST.md, the "derive" items).

    python tools/eval_metrics.py <run_dir> [<run_dir> ...]

Reads only the run folder plus the run's net file and route file (found via
the sumocfg its manifest names), writes <run_dir>/eval_metrics.json and
prints one line per run. Older run folders need a
`tools/verify_flags.py --replay` first, for extents.csv, ground_truth.csv and
network_metrics.csv.

arrival_rates
    veh/h per 5-min bin on each surge entry lane, each blocked lane and each
    lane feeding a blocked lane. Source lane_entries.csv; runs logged before
    it fall back to tripinfo departLane, which only covers lanes no
    connection leads into (boundary entry lanes) and misses lane changes
    after insertion. Other lanes are then null.
time_to_recovery_s
    First cycle starting at or after the end of the surge / blockage window
    in which every extent is N, minus the window end. Null without a window
    or when the run never recovers.
peak_vehicles_in_system
    Largest per-step count in network_timeseries.csv.
spillback_count
    Lane greens with SOSI > 0 on any movement, or a queue at green start or
    end >= 0.95 of the lane's storage.
side_street_delay_s
    Mean tripinfo waitingTime of the vehicles whose path uses no lane that
    was flagged at any time in the run. A route uses a lane of one of its
    edges when the lane connects to the route's next edge (any lane of the
    last edge).
detector
    The movement flags against ground_truth.csv, per lane-cycle. A lane is
    flagged in a cycle when any movement leaving it is active at cycle close.
    recall = flagged / gt-oversaturated (null with none), false positives =
    flagged lane-cycles with gt false.
regime_slices
    Per extent and per regime: cycles, ATT contribution and mean
    vehicles_in_system (network_metrics.csv joined to extents.csv on
    cycle_start). The ATT contribution of a cycle is the vehicle-seconds
    spent in the network during it over the vehicles departed in the run, so
    the cycles of a run sum to its ATT, in-flight vehicles counted. Per
    regime, a cycle counts when any extent is in that regime (so a cycle can
    count under two); N is the cycles with no extent outside N.
plan_transitions
    Rows of plan_changes.csv; null when the run did not write it.
scenario_gate
    utils/scenario_gate.py.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import (  # noqa: E402
    EVAL_METRICS_FILENAME,
    EXTENTS_FILENAME,
    GROUND_TRUTH_FILENAME,
    LANE_ENTRIES_FILENAME,
    MOVEMENT_FLAGS_FILENAME,
    NETWORK_METRICS_FILENAME,
    NETWORK_TIMESERIES_FILENAME,
    OBSTACLE_VEHICLE_PREFIX,
    PLAN_CHANGES_FILENAME,
    SUMO_TRIPINFO_FILENAME,
)
from utils.diagnoser import DEFAULT_CYCLE_S  # noqa: E402
from utils.network_topology import read_network, read_vehicle_routes  # noqa: E402
from utils.regime import NORMAL  # noqa: E402
from utils.run_logs import (  # noqa: E402
    blockage_window_ends,
    blocked_lanes,
    input_paths,
    parse_float,
    read_csv_rows,
    read_lane_greens,
    read_manifest,
)
from utils.scenario_gate import scenario_gate_record  # noqa: E402

ARRIVAL_BIN_S = 300
SPILLBACK_OCCUPANCY = 0.95
SECONDS_PER_HOUR = 3600


# --- arrival rates ------------------------------------------------------------------

def arrival_target_lanes(manifest, run_dir, network):
    """lane -> sorted roles (surge_entry, blocked, feeder)."""
    roles = defaultdict(set)
    for surge in (manifest.get("surge") or {}).get("surges", []):
        for lane in network.edge_lanes.get(surge["route"][0], []):
            roles[lane].add("surge_entry")
    for lane in blocked_lanes(run_dir):
        roles[lane].add("blocked")
        for feeder in network.feeders(lane):
            roles[feeder].add("feeder")
    return {lane: sorted(lane_roles) for lane, lane_roles in sorted(roles.items())}


def binned_rates(times, horizon_s):
    n_bins = int(horizon_s // ARRIVAL_BIN_S) + 1
    counts = [0] * n_bins
    for t in times:
        counts[min(int(t // ARRIVAL_BIN_S), n_bins - 1)] += 1
    return [count * SECONDS_PER_HOUR / ARRIVAL_BIN_S for count in counts]


def lane_entry_times(run_dir):
    """lane -> entry times, one per vehicle, from lane_entries.csv."""
    times = defaultdict(list)
    for row in read_csv_rows(run_dir / LANE_ENTRIES_FILENAME):
        times[row["lane"]].extend([int(row["t"])] * int(row["entered"]))
    return times


def depart_times_by_lane(tripinfos):
    times = defaultdict(list)
    for trip in tripinfos:
        times[trip["departLane"]].append(float(trip["depart"]))
    return times


def arrival_rates(manifest, run_dir, network, tripinfos, horizon_s):
    targets = arrival_target_lanes(manifest, run_dir, network)
    if (run_dir / LANE_ENTRIES_FILENAME).exists():
        source, times = "lane_entries", lane_entry_times(run_dir)
    else:
        source, times = "tripinfo_departLane", depart_times_by_lane(tripinfos)
    rates = {}
    for lane, roles in targets.items():
        # Departures are a lane's only arrivals when no connection leads into it.
        covered = source == "lane_entries" or not network.feeders(lane)
        rates[lane] = {"roles": roles, "source": source if covered else None,
                       "veh_h": binned_rates(times[lane], horizon_s) if covered else None}
    return {"bin_s": ARRIVAL_BIN_S, "lanes": rates}


# --- recovery, peak, spillback ----------------------------------------------------------

def event_window_end(manifest, run_dir):
    ends = [surge["window"][1] for surge in (manifest.get("surge") or {}).get("surges", [])]
    ends += blockage_window_ends(run_dir)
    return max(ends) if ends else None


def non_normal_cycles(extent_rows):
    return {int(row["cycle_start"]) for row in extent_rows if row["regime"] != NORMAL}


def time_to_recovery(cycle_starts, extent_rows, window_end):
    """(seconds, cycle_start) or (None, None)."""
    if window_end is None:
        return None, None
    busy = non_normal_cycles(extent_rows)
    for cycle_start in sorted(cycle_starts):
        if cycle_start >= window_end and cycle_start not in busy:
            return cycle_start - window_end, cycle_start
    return None, None


def is_spillback(green, storage_ratio=SPILLBACK_OCCUPANCY):
    queues = [ratio for ratio in (green.ratio_start, green.ratio_end) if ratio is not None]
    return ((green.max_sosi is not None and green.max_sosi > 0)
            or any(ratio >= storage_ratio for ratio in queues))


# --- side-street delay ----------------------------------------------------------------

def read_tripinfos(path):
    trips = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "tripinfo":
            if not element.get("id").startswith(OBSTACLE_VEHICLE_PREFIX):
                trips.append(dict(element.attrib))
            element.clear()
    return trips


def lane_next_edges(network):
    """lane -> edges its connections lead to."""
    next_edges = defaultdict(set)
    for source, target in network.connections:
        next_edges[source].add(network.lane_edge.get(target))
    return next_edges


def lanes_used(route, network, next_edges):
    """Lanes a vehicle on this route can be on: the lanes of each edge that
    connect to the next edge, every lane of the last edge."""
    next_edge = dict(zip(route, route[1:]))
    used = set()
    for edge in route:
        for lane in network.edge_lanes.get(edge, []):
            if edge not in next_edge or next_edge[edge] in next_edges[lane]:
                used.add(lane)
    return used


def ever_flagged_lanes(flag_rows):
    return {row["lane"] for row in flag_rows if row["active"] == "1"}


def side_street_delay(tripinfos, routes, network, flagged):
    next_edges = lane_next_edges(network)
    waits = []
    for trip in tripinfos:
        route = routes.get(trip["id"])
        if route is None:
            continue
        if lanes_used(route, network, next_edges) & flagged:
            continue
        waits.append(float(trip["waitingTime"]))
    return {"vehicles": len(waits),
            "mean_waiting_s": round(sum(waits) / len(waits), 2) if waits else None}


# --- detector -------------------------------------------------------------------------

def flagged_lanes_by_cycle(flag_rows, cycle_starts, cycle_s):
    """cycle_start -> lanes with an active movement at that cycle's close,
    carried forward from each movement's last green."""
    rows = sorted(flag_rows, key=lambda row: int(row["t_close"]))
    active = {}
    flagged = {}
    index = 0
    for cycle_start in sorted(cycle_starts):
        cycle_end = cycle_start + cycle_s
        while index < len(rows) and int(rows[index]["t_close"]) < cycle_end:
            row = rows[index]
            active[(row["lane"], row["target_lane"])] = row["active"] == "1"
            index += 1
        flagged[cycle_start] = {lane for (lane, _), on in active.items() if on}
    return flagged


def detector_scores(ground_truth_rows, flagged_by_cycle):
    true_positive = false_negative = false_positive = 0
    for row in ground_truth_rows:
        oversaturated = row["gt_oversat"] == "1"
        flagged = row["lane"] in flagged_by_cycle.get(int(row["cycle_start"]), set())
        true_positive += oversaturated and flagged
        false_negative += oversaturated and not flagged
        false_positive += flagged and not oversaturated
    gt_positive = true_positive + false_negative
    n_flagged = true_positive + false_positive
    return {
        "lane_cycles": len(ground_truth_rows),
        "gt_positive_lane_cycles": gt_positive,
        "flagged_lane_cycles": n_flagged,
        "recall": round(true_positive / gt_positive, 4) if gt_positive else None,
        "precision": round(true_positive / n_flagged, 4) if n_flagged else None,
        "false_positive_lane_cycles": false_positive,
    }


# --- regime slices ----------------------------------------------------------------------

def vehicle_seconds_by_cycle(timeseries_rows, cycle_s, step_s):
    """A step t belongs to the cycle the Diagnoser closes on or after it:
    t // cycle_s."""
    totals = defaultdict(float)
    for row in timeseries_rows:
        totals[int(float(row["t"])) // cycle_s * cycle_s] += int(row["vehicles_in_network"]) * step_s
    return totals


def slice_summary(cycles, att_by_cycle, vehicles_by_cycle):
    vehicles = [vehicles_by_cycle[c] for c in cycles if vehicles_by_cycle.get(c) is not None]
    return {
        "cycles": len(cycles),
        "att_contribution_s": round(sum(att_by_cycle.get(c, 0.0) for c in cycles), 2),
        "mean_vehicles_in_system": round(sum(vehicles) / len(vehicles), 1) if vehicles else None,
    }


def regime_slices(extent_rows, cycle_starts, att_by_cycle, vehicles_by_cycle):
    extent_cycles = defaultdict(lambda: defaultdict(list))
    root_lanes = {}
    regime_cycles = defaultdict(set)
    for row in extent_rows:
        cycle_start = int(row["cycle_start"])
        extent_cycles[row["extent_id"]][row["regime"]].append(cycle_start)
        root_lanes[row["extent_id"]] = row.get("root_lane", "")
        if row["regime"] != NORMAL:
            regime_cycles[row["regime"]].add(cycle_start)
    regime_cycles[NORMAL] = set(cycle_starts) - non_normal_cycles(extent_rows)
    per_extent = {
        extent_id: {"root_lane": root_lanes[extent_id],
                    "regimes": {regime: slice_summary(cycles, att_by_cycle, vehicles_by_cycle)
                                for regime, cycles in sorted(regimes.items())}}
        for extent_id, regimes in extent_cycles.items()}
    per_regime = {regime: slice_summary(sorted(cycles), att_by_cycle, vehicles_by_cycle)
                  for regime, cycles in sorted(regime_cycles.items())}
    return {"per_regime": per_regime, "per_extent": per_extent}


# --- one run ----------------------------------------------------------------------------

def cycle_settings(manifest):
    args = manifest.get("args") or {}
    step_s = (manifest.get("sumo") or {}).get("step_length_s") or 1.0
    return args.get("cycle_s") or DEFAULT_CYCLE_S, step_s


def row_count(path):
    return len(read_csv_rows(path)) if path.exists() else None


def evaluate(run_dir):
    manifest = read_manifest(run_dir)
    cycle_s, step_s = cycle_settings(manifest)
    net_path, route_path = input_paths(manifest)
    if net_path is None or route_path is None:
        raise FileNotFoundError(f"{run_dir}: the run's sumocfg is not on this machine")
    network = read_network(net_path)

    flag_rows = read_csv_rows(run_dir / MOVEMENT_FLAGS_FILENAME)
    extent_rows = read_csv_rows(run_dir / EXTENTS_FILENAME)
    ground_truth_rows = read_csv_rows(run_dir / GROUND_TRUTH_FILENAME)
    network_rows = read_csv_rows(run_dir / NETWORK_METRICS_FILENAME)
    timeseries_rows = read_csv_rows(run_dir / NETWORK_TIMESERIES_FILENAME)
    tripinfos = read_tripinfos(run_dir / SUMO_TRIPINFO_FILENAME)

    cycle_starts = [int(row["cycle_start"]) for row in network_rows]
    horizon_s = int(float(timeseries_rows[-1]["t"]))
    departed = int(timeseries_rows[-1]["departed_cum"])
    att_by_cycle = {cycle: seconds / departed for cycle, seconds in
                    vehicle_seconds_by_cycle(timeseries_rows, cycle_s, step_s).items()}
    vehicles_by_cycle = {int(row["cycle_start"]): parse_float(row["vehicles_in_system"])
                         for row in network_rows}
    window_end = event_window_end(manifest, run_dir)
    recovery_s, recovery_cycle = time_to_recovery(cycle_starts, extent_rows, window_end)
    flagged_by_cycle = flagged_lanes_by_cycle(
        flag_rows, {int(row["cycle_start"]) for row in ground_truth_rows}, cycle_s)

    return {
        "run_dir": str(run_dir),
        "cycle_s": cycle_s,
        "arrival_rates": arrival_rates(manifest, run_dir, network, tripinfos, horizon_s),
        "event_window_end_s": window_end,
        "time_to_recovery_s": recovery_s,
        "recovery_cycle_start": recovery_cycle,
        "peak_vehicles_in_system": max(int(row["vehicles_in_network"])
                                       for row in timeseries_rows),
        "spillback_count": sum(1 for green in read_lane_greens(run_dir) if is_spillback(green)),
        "side_street_delay": side_street_delay(tripinfos, read_vehicle_routes(route_path),
                                               network, ever_flagged_lanes(flag_rows)),
        "detector": detector_scores(ground_truth_rows, flagged_by_cycle),
        "att_s": round(sum(att_by_cycle.values()), 2),
        "regime_slices": regime_slices(extent_rows, cycle_starts, att_by_cycle,
                                       vehicles_by_cycle),
        "plan_transitions": row_count(run_dir / PLAN_CHANGES_FILENAME),
        "scenario_gate": scenario_gate_record(run_dir),
    }


def summary_line(run_dir, metrics):
    detector = metrics["detector"]
    recall = "n/a" if detector["recall"] is None else f"{detector['recall']:.3f}"
    recovery = metrics["time_to_recovery_s"]
    return (f"{run_dir.name}: recall={recall} "
            f"fp_lane_cycles={detector['false_positive_lane_cycles']} "
            f"time_to_recovery={'n/a' if recovery is None else f'{recovery}s'} "
            f"peak_vehicles={metrics['peak_vehicles_in_system']} "
            f"spillback_count={metrics['spillback_count']} "
            f"gate={str(metrics['scenario_gate']['passed']).lower()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run dirs to evaluate.")
    return parser.parse_args()


def main():
    for run_dir in parse_args().run_dirs:
        metrics = evaluate(run_dir)
        (run_dir / EVAL_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2),
                                                      encoding="utf-8")
        print(summary_line(run_dir, metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
