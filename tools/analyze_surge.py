"""Measure one surge run against its same-seed control.

Reads network_timeseries.csv, edge_timeseries.csv and surge_timeseries.csv
from both run directories plus the scenario's sidecar (for each surge's
window and approach edge) and writes <run>/surge_analysis.json with, per
surge: throughput in the window, peak approach queue, an oversaturation
flag, recovery time and the insertion backlog; plus network-level excess
occupancy over the control.

    python tools/analyze_surge.py --run logs/x/ft_cross_r600_seed1_... \
        --control logs/x/ft_h2_steady_seed1_...
    python tools/analyze_surge.py estimate --capacity 900 --base-rate 300 \
        --surge-rate 1200 --duration 1200
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import (EDGE_TIMESERIES_FILENAME, NETWORK_TIMESERIES_FILENAME,  # noqa: E402
                            RUN_MANIFEST_FILENAME, SURGE_TIMESERIES_FILENAME)
from experiments import SURGE_SCENARIOS  # noqa: E402
from utils import surge_analysis as sa  # noqa: E402

ANALYSIS_FILENAME = "surge_analysis.json"


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / RUN_MANIFEST_FILENAME).read_text())
    surge_path = run_dir / SURGE_TIMESERIES_FILENAME
    return {
        "dir": run_dir,
        "manifest": manifest,
        "seed": (manifest.get("environment") or {}).get("seed"),
        "steps": (manifest.get("environment") or {}).get("simulation_steps"),
        "config": Path((manifest.get("environment") or {}).get("simulation_config") or "").name,
        "network": sa.read_timeseries(run_dir / NETWORK_TIMESERIES_FILENAME),
        "edges": sa.read_edge_timeseries(run_dir / EDGE_TIMESERIES_FILENAME),
        "surges": sa.read_surge_timeseries(surge_path) if surge_path.exists() else {},
    }


def scenario_for_config(config_basename):
    for alias, info in SURGE_SCENARIOS.items():
        if Path(info["sumocfg"]).name == config_basename:
            return alias, json.loads((REPO / info["path"]).read_text(encoding="utf-8"))
    return None, None


def load_sidecar(scenario_path, run):
    if scenario_path:
        scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
        alias = scenario.get("config_alias")
    else:
        alias, scenario = scenario_for_config(run["config"])
        if scenario is None:
            sys.exit(f"analyze_surge: no surge scenario matches {run['config']}; pass --scenario")
    sidecar = json.loads((REPO / scenario["sidecar"]).read_text(encoding="utf-8"))
    return alias, sidecar


def check_pairing(run, control):
    if run["seed"] != control["seed"]:
        print(f"warning: seeds differ (run {run['seed']}, control {control['seed']})")
    if run["steps"] != control["steps"]:
        print(f"warning: steps differ (run {run['steps']}, control {control['steps']})")


def approach_series(run, edge):
    return sa.series(run["edges"], f"halting:{edge}")


def surge_pending_series(run, name, times):
    rows = run["surges"].get(name, [])
    by_t = {row["t"]: row["pending"] for row in rows}
    return [by_t.get(t, 0.0) for t in times]


def analyze_surge(entry, run, control, options):
    window = tuple(entry["window"])
    hold = (window[0] + entry["ramp"], window[1] - entry["ramp"])
    edge = entry.get("approach_edge")
    result = {"name": entry["name"], "mode": entry["mode"], "window": list(window),
              "hold": list(hold), "approach_edge": edge, "rate_veh_h": entry.get("rate_veh_h"),
              "mult": entry.get("mult")}
    result.update(surge_stream_counts(run, entry["name"], window))
    if edge is None:
        result["note"] = "no approach edge in the sidecar; queue metrics skipped"
        return result
    times, halting = approach_series(run, edge)
    _control_times, control_halting = approach_series(control, edge)
    pending = surge_pending_series(run, entry["name"], times)
    result["approach_peak_halting"] = sa.peak_in_window(times, halting, window)
    result["control_approach_peak_halting"] = sa.peak_in_window(times, control_halting, window)
    result.update(sa.oversaturated(times, halting, pending, hold, options.smooth,
                                   options.growth_min))
    result.update(sa.recovery_time(times, halting, control_halting, window[1],
                                   options.smooth, options.tolerance, options.slack,
                                   options.sustain))
    result["max_pending_in_window"] = max(
        (p for t, p in zip(times, pending) if window[0] <= t <= window[1]), default=0.0)
    return result


def surge_stream_counts(run, name, window):
    rows = run["surges"].get(name)
    if not rows:
        return {"stream_departed": None, "stream_arrived_by_window_end": None,
                "stream_arrived_by_end": None}
    return {"stream_departed": rows[-1]["departed_cum"],
            "stream_arrived_by_window_end": sa.cumulative_delta(rows, "arrived_cum", 0, window[1]),
            "stream_arrived_by_end": rows[-1]["arrived_cum"]}


def analyze(run, control, sidecar, alias, options):
    windows = [tuple(s["window"]) for s in sidecar["surges"]] or [(0, run["steps"] or 0)]
    union = (min(w[0] for w in windows), max(w[1] for w in windows))
    report = {
        "run_dir": str(run["dir"]), "control_dir": str(control["dir"]),
        "config_alias": alias, "seed": run["seed"], "steps": run["steps"],
        "controller": run["manifest"].get("controller"),
        "surge_window_union": list(union),
        "network": {
            **sa.throughput_in_window(run["network"], control["network"], union),
            **sa.excess_over_control(run["network"], control["network"], options.smooth),
            "max_pending_in_window": sa.max_in_window(run["network"], "not_inserted_pending", union),
            "control_max_pending_in_window": sa.max_in_window(control["network"],
                                                              "not_inserted_pending", union),
        },
        "surges": [analyze_surge(entry, run, control, options) for entry in sidecar["surges"]],
    }
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    sub = parser.add_subparsers(dest="command")
    estimate = sub.add_parser("estimate", help="deterministic queueing estimate")
    estimate.add_argument("--capacity", type=float, required=True, help="veh/h the approach discharges")
    estimate.add_argument("--base-rate", type=float, required=True, help="veh/h baseline demand")
    estimate.add_argument("--surge-rate", type=float, required=True, help="veh/h during the surge")
    estimate.add_argument("--duration", type=float, required=True, help="surge hold in seconds")
    return parser


def add_run_arguments(parser):
    parser.add_argument("--run", help="surge run directory")
    parser.add_argument("--control", help="same-seed control run directory")
    parser.add_argument("--scenario", help="scenario JSON (default: resolved from the manifest)")
    parser.add_argument("--smooth", type=int, default=300, help="trailing mean window, s")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--slack", type=float, default=1.0, help="absolute vehicles of slack")
    parser.add_argument("--sustain", type=int, default=300, help="seconds recovered before it counts")
    parser.add_argument("--growth-min", type=float, default=2.0,
                        help="vehicles of growth over the last third of the hold that count as rising")
    parser.add_argument("--out", help=f"default <run>/{ANALYSIS_FILENAME}")


def main(argv=None):
    options = build_parser().parse_args(argv)
    if options.command == "estimate":
        estimate = sa.queueing_estimate(options.capacity, options.base_rate,
                                        options.surge_rate, options.duration)
        print(json.dumps(estimate, indent=2))
        return estimate
    if not options.run or not options.control:
        sys.exit("analyze_surge: --run and --control are required")
    run, control = load_run(options.run), load_run(options.control)
    check_pairing(run, control)
    alias, sidecar = load_sidecar(options.scenario, run)
    report = analyze(run, control, sidecar, alias, options)
    out = Path(options.out) if options.out else run["dir"] / ANALYSIS_FILENAME
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
