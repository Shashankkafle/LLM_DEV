"""Regenerate report_sheet/blockage_results.csv from logs/blockage_final_*
run directories (the blockage report suite: FT/MP/CoLight/Adv-CoLight with a
lane-blockage scenario, CoLight evaluated from the clean suite's weights).

Same behavior as build_report_sheet.py: placeholders for expected-but-missing
runs, hand-written notes survive regeneration, incomplete runs are flagged.
Adds blockage provenance columns: blockage_scenario, weights_dir, eval_round.

Usage:
    python build_blockage_report_sheet.py
"""
import csv
import glob
import json
import os
import re
import socket

LOGS_DIR = "logs"
OUT_CSV = os.path.join("report_sheet", "blockage_results.csv")

CONTROLLER_TOKENS = {
    "ft30": "fixedtime",
    "ft60": "fixedtime (60s green)",
    "mp30": "maxpressure",
    "colight": "colight",
    "advcolight": "advanced_colight",
    "lightgpt13b": "lightgpt13b",
    "qwen05b": "qwen05b",
}


def load_json(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_arg(manifest, key):
    """CLI args live top-level in old replay_meta.json but nested under
    'args' in run_manifest.json (run-record v2). Check both."""
    if manifest.get(key) is not None:
        return manifest[key]
    return (manifest.get("args") or {}).get(key)

COLUMNS = [
    "run_dir", "date", "machine", "controller", "dataset", "routes",
    "green_s", "steps", "seed", "rounds", "blockage_scenario", "weights_dir",
    "eval_round", "sumo_version", "status",
    "att_paper", "awt_paper", "aql", "att_internal", "awt_total",
    "finished", "not_inserted", "teleports", "completion_rate",
    "valid_resp_rate", "parse_err_rate", "halluc_rate", "notes",
]

EXPECTED_RUNS = [
    ("blockage_final_ft30_hz1_cfphys", "suite"),
    ("blockage_final_mp30_hz1_cfphys", "suite"),
    ("blockage_final_ft30_jinan1_cfphys", "suite"),
    ("blockage_final_mp30_jinan1_cfphys", "suite"),
    ("blockage_final_colight_hz1_cfphys", "suite (eval from clean weights)"),
    ("blockage_final_advcolight_hz1_cfphys", "suite (eval from clean weights)"),
    ("blockage_final_colight_jinan1_cfphys", "suite (eval from clean weights)"),
    ("blockage_final_advcolight_jinan1_cfphys", "suite (eval from clean weights)"),
    ("blockage_final_lightgpt13b_hz1_cfphys", "manual LLM run"),
    ("blockage_final_lightgpt13b_jinan1_cfphys", "manual LLM run"),
]


def parse_test_name(dir_name):
    """blockage_final_<controller>_<dataset>_<routes>_<timestamp> -> parts."""
    base = re.sub(r"_\d{8}_\d{6}$", "", dir_name)
    m = re.match(r"blockage_final_([a-z0-9]+)_([a-z0-9]+)_([a-z0-9]+)$", base)
    if not m:
        return base, "", "", ""
    token, dataset, routes = m.groups()
    return base, CONTROLLER_TOKENS.get(token, token), dataset, routes


def row_for_run(run_dir):
    dir_name = os.path.basename(run_dir)
    summary = load_json(run_dir, "final_summary.json")
    manifest = load_json(run_dir, "run_manifest.json") or load_json(run_dir, "replay_meta.json")
    _, controller, dataset, routes = parse_test_name(dir_name)

    def s(key):
        value = summary.get(key)
        return "" if value is None else value

    def arg(key):
        value = manifest_arg(manifest, key)
        return "" if value is None else value

    scenario = arg("blockage_scenario")
    return {
        "run_dir": dir_name,
        "date": summary.get("run_started_at") or manifest.get("started_at") or "",
        "machine": socket.gethostname(),
        "controller": manifest.get("controller") or arg("variant") or controller,
        "dataset": dataset,
        "routes": routes,
        "green_s": 30,
        "steps": arg("simulation_steps") or s("simulation_steps"),
        "seed": arg("seed"),
        "rounds": arg("num_rounds"),
        "blockage_scenario": os.path.basename(scenario) if scenario else "",
        "weights_dir": arg("weights_dir"),
        "eval_round": arg("eval_round"),
        "sumo_version": s("sumo_version"),
        "status": "ok" if summary else "incomplete (no final_summary.json)",
        "att_paper": s("cityflow_clock_att_s"),
        "awt_paper": s("average_per_decision_wait_s"),
        "aql": s("average_queue_length"),
        "att_internal": s("cityflow_style_att_s"),
        "awt_total": s("cityflow_style_awt_s"),
        "finished": s("sumo_vehicles_finished"),
        "not_inserted": s("sumo_vehicles_not_inserted"),
        "teleports": s("sumo_teleports_total"),
        "completion_rate": s("completion_rate"),
        "valid_resp_rate": s("valid_response_rate"),
        "parse_err_rate": s("parse_error_rate"),
        "halluc_rate": s("hallucination_rate"),
        "notes": "",
    }


def placeholder_row(test_name, note):
    _, controller, dataset, routes = parse_test_name(test_name)
    row = {col: "" for col in COLUMNS}
    row.update({
        "controller": controller, "dataset": dataset, "routes": routes,
        "green_s": 30, "steps": 3600,
        "status": "not run yet", "notes": f"{note}: {test_name}",
    })
    return row


def load_existing_notes():
    if not os.path.exists(OUT_CSV):
        return {}
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        return {r["run_dir"]: r.get("notes", "") for r in csv.DictReader(f)
                if r.get("run_dir") and r.get("notes")}


def main():
    kept_notes = load_existing_notes()
    rows = []
    seen_test_names = set()

    for run_dir in sorted(glob.glob(os.path.join(LOGS_DIR, "blockage_final_*"))):
        if not os.path.isdir(run_dir):
            continue
        row = row_for_run(run_dir)
        row["notes"] = kept_notes.get(row["run_dir"], "")
        rows.append(row)
        base, _, _, _ = parse_test_name(os.path.basename(run_dir))
        seen_test_names.add(base)

    for test_name, note in EXPECTED_RUNS:
        if test_name not in seen_test_names:
            rows.append(placeholder_row(test_name, note))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    done = sum(1 for r in rows if r["status"] == "ok")
    print(f"Wrote {OUT_CSV}: {len(rows)} rows ({done} completed runs, "
          f"{len(rows) - done} placeholders/incomplete).")


if __name__ == "__main__":
    main()
