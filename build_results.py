"""Aggregate run results from run_manifest.json + final_summary.json.

Replaces build_report_sheet.py and build_blockage_report_sheet.py with one
manifest-driven collector. Run identity (controller, routes, timing, seed,
blockage) is read from each run's manifest -- directory names are never parsed,
so blockage-vs-clean and every seed are just columns, not naming conventions.

Produces two sheets under report_sheet/:
  <scope>_runs.csv       one row per run found (the physical inventory)
  <scope>_by_config.csv  one row per config (seeds collapsed to mean/std/n),
                         with each blockage arm's ATT delta vs its clean
                         baseline, expressed as a multiple of the clean seed
                         spread -- the acceptance-rule number.

    python build_results.py                          # all of logs/
    python build_results.py --experiment mp_blockage_sweep
"""
import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path

from configurations import LOGS_DIR_NAME, RUN_MANIFEST_FILENAME
from experiments import identity_from_manifest, identity_key

OUT_DIR = "report_sheet"
FINAL_SUMMARY = "final_summary.json"

# sheet column name -> final_summary.json key
METRICS = [
    ("att_paper", "cityflow_clock_att_s"),
    ("awt_paper", "average_per_decision_wait_s"),
    ("aql", "average_queue_length"),
    ("att_internal", "cityflow_style_att_s"),
    ("awt_total", "cityflow_style_awt_s"),
    ("finished", "sumo_vehicles_finished"),
    ("not_inserted", "sumo_vehicles_not_inserted"),
    ("teleports", "sumo_teleports_total"),
    ("completion_rate", "completion_rate"),
    ("valid_resp_rate", "valid_response_rate"),
    ("parse_err_rate", "parse_error_rate"),
    ("halluc_rate", "hallucination_rate"),
]
# metrics summarised per config (mean/std across seeds)
AGG_METRICS = ["att_internal", "att_paper", "awt_paper", "aql",
               "completion_rate", "finished", "not_inserted"]


def dataset_routes(config_basename):
    name = (config_basename or "").replace(".sumocfg", "")
    city = next((c for c in ("hangzhou", "jinan", "newyork", "atlanta")
                 if c in name), "")
    grid = re.search(r"_(\d+_\d+)_", name)
    dataset = "_".join(p for p in (city, grid.group(1) if grid else "") if p) or name
    if "cfphys" in name:
        routes = "cfphys"
    elif "5816" in name:
        routes = "real5816"
    else:
        routes = "real"
    return dataset, routes


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def group_of(run_dir, logs_dir):
    rel = run_dir.relative_to(logs_dir)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def row_for_run(run_dir, logs_root):
    manifest = load_json(run_dir / RUN_MANIFEST_FILENAME)
    summary = load_json(run_dir / FINAL_SUMMARY)
    env = manifest.get("environment") or {}
    blk = manifest.get("blockage") or {}
    dataset, routes = dataset_routes(Path(env.get("simulation_config") or "").name)
    reused = load_json(run_dir / "reused_from.json")

    row = {
        "run_dir": str(run_dir),
        "experiment": group_of(run_dir, logs_root),
        "status": manifest.get("status") or ("ok" if summary else "incomplete"),
        "reused_from": reused.get("reused_from", ""),
        "controller": manifest.get("controller", ""),
        "dataset": dataset,
        "routes": routes,
        "intersection_config": env.get("intersection_config", ""),
        "green_s": (env.get("signal_timing") or {}).get("default_green_duration", ""),
        "steps": env.get("simulation_steps", ""),
        "seed": env.get("seed"),
        "blockage": blk.get("scenario_name", "none") if blk else "none",
        "hide_info": blk.get("hide_blockage_info", False) if blk else False,
        "info_scope": blk.get("blockage_info_scope", "") if blk else "",
        "num_rounds": (manifest.get("args") or {}).get("num_rounds", ""),
        "sumo_version": (manifest.get("sumo") or {}).get("version", ""),
        "date": manifest.get("started_at", ""),
        "identity": identity_key(identity_from_manifest(manifest)),
        "completed": manifest.get("status") == "completed" and bool(summary),
    }
    for col, key in METRICS:
        value = summary.get(key)
        row[col] = "" if value is None else value
    return row


def collect_rows(scan_dir, logs_root):
    rows = []
    for manifest_path in sorted(Path(scan_dir).glob(f"**/{RUN_MANIFEST_FILENAME}")):
        rows.append(row_for_run(manifest_path.parent, logs_root))
    return rows


def _config_key(row):
    """Identity minus seed -- groups replicate seeds of one config."""
    return (row["controller"], row["dataset"], row["routes"],
            row["intersection_config"], row["steps"], row["blockage"],
            row["hide_info"], row["info_scope"], row["num_rounds"])


def _family_key(row):
    """Config minus the blockage -- links a blockage arm to its clean baseline."""
    return (row["controller"], row["dataset"], row["routes"],
            row["intersection_config"], row["steps"], row["hide_info"],
            row["info_scope"], row["num_rounds"])


def dedupe_completed(rows):
    """One row per (identity incl seed), newest kept: collapses reused copies
    and any accidental same-seed re-runs so a seed is never double-counted."""
    best = {}
    for row in rows:
        if not row["completed"]:
            continue
        key = row["identity"]
        if key not in best or row["date"] > best[key]["date"]:
            best[key] = row
    return list(best.values())


def _floats(rows, col):
    out = []
    for row in rows:
        try:
            out.append(float(row[col]))
        except (TypeError, ValueError):
            pass
    return out


def aggregate_configs(completed_rows):
    by_config = {}
    for row in completed_rows:
        by_config.setdefault(_config_key(row), []).append(row)

    configs = []
    for key, runs in by_config.items():
        sample = runs[0]
        seeds = sorted({r["seed"] for r in runs}, key=lambda s: (s is None, s))
        entry = {
            "controller": sample["controller"], "dataset": sample["dataset"],
            "routes": sample["routes"],
            "intersection_config": sample["intersection_config"],
            "steps": sample["steps"], "blockage": sample["blockage"],
            "hide_info": sample["hide_info"], "info_scope": sample["info_scope"],
            "num_rounds": sample["num_rounds"],
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
            "_family": _family_key(sample),
        }
        for col in AGG_METRICS:
            values = _floats(runs, col)
            entry[f"{col}_mean"] = round(statistics.mean(values), 3) if values else ""
            entry[f"{col}_std"] = (round(statistics.stdev(values), 3)
                                   if len(values) >= 2 else "")
        configs.append(entry)
    _add_blockage_deltas(configs)
    return configs


def _add_blockage_deltas(configs):
    """For each blockage arm, ATT delta vs its clean (none) baseline, and that
    delta as a multiple of the clean seed spread (the acceptance number)."""
    clean = {c["_family"]: c for c in configs if c["blockage"] == "none"}
    for c in configs:
        c["delta_att_internal"] = ""
        c["delta_over_clean_spread"] = ""
        if c["blockage"] == "none":
            continue
        base = clean.get(c["_family"])
        if not base or base["att_internal_mean"] == "" or c["att_internal_mean"] == "":
            continue
        delta = c["att_internal_mean"] - base["att_internal_mean"]
        c["delta_att_internal"] = round(delta, 3)
        spread = base["att_internal_std"]
        if isinstance(spread, (int, float)) and spread > 0:
            c["delta_over_clean_spread"] = round(delta / spread, 1)


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


RUN_COLUMNS = (["run_dir", "experiment", "status", "reused_from", "controller",
                "dataset", "routes", "intersection_config", "green_s", "steps",
                "seed", "blockage", "hide_info", "info_scope", "num_rounds",
                "sumo_version", "date"] + [c for c, _ in METRICS])

CONFIG_COLUMNS = (["controller", "dataset", "routes", "intersection_config",
                   "steps", "blockage", "hide_info", "info_scope", "num_rounds",
                   "n_seeds", "seeds"]
                  + [f"{m}_{stat}" for m in AGG_METRICS for stat in ("mean", "std")]
                  + ["delta_att_internal", "delta_over_clean_spread"])


def print_summary(configs):
    def fmt(mean, std):
        if mean == "":
            return "   n/a"
        return f"{mean:7.1f}" + (f"±{std:.1f}" if std != "" else "")

    def sort_key(family):
        return tuple("" if x is None else str(x) for x in family)

    for family in sorted({c["_family"] for c in configs}, key=sort_key):
        arms = [c for c in configs if c["_family"] == family]
        head = arms[0]
        print(f"\n{head['controller']}  {head['dataset']}/{head['routes']}  "
              f"({head['intersection_config']}, {head['steps']} steps)")
        for c in sorted(arms, key=lambda x: (x["blockage"] != "none", x["blockage"])):
            line = (f"  {c['blockage']:26s} n={c['n_seeds']}  "
                    f"ATT={fmt(c['att_internal_mean'], c['att_internal_std'])}  "
                    f"AQL={fmt(c['aql_mean'], c['aql_std'])}  "
                    f"compl={fmt(c['completion_rate_mean'], c['completion_rate_std'])}")
            if c["delta_att_internal"] != "":
                mult = c["delta_over_clean_spread"]
                line += f"  ΔATT={c['delta_att_internal']:+.1f}"
                line += f" ({mult}x spread)" if mult != "" else ""
            print(line)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str,
                        help="Aggregate only logs/<experiment>/ into "
                             "<experiment>_*.csv (default: all of logs/)")
    parser.add_argument("--logs_dir", type=str, default=LOGS_DIR_NAME)
    return parser.parse_args()


def main(args):
    logs_dir = Path(args.logs_dir)
    scope_dir = logs_dir / args.experiment if args.experiment else logs_dir
    scope = args.experiment or "all"
    if not scope_dir.exists():
        print(f"No such logs dir: {scope_dir}")
        return

    rows = collect_rows(scope_dir, logs_dir)
    completed = dedupe_completed(rows)
    configs = aggregate_configs(completed)

    write_csv(os.path.join(OUT_DIR, f"{scope}_runs.csv"), RUN_COLUMNS, rows)
    write_csv(os.path.join(OUT_DIR, f"{scope}_by_config.csv"), CONFIG_COLUMNS, configs)

    print(f"Scanned {len(rows)} run(s) under {scope_dir} "
          f"({len(completed)} completed, {len(configs)} configs).")
    print_summary(configs)
    print(f"\nWrote {OUT_DIR}/{scope}_runs.csv and {OUT_DIR}/{scope}_by_config.csv")


if __name__ == "__main__":
    main(parse_args())
