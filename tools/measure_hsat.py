"""Estimate saturation headway h_sat from per-green lane records.

    python tools/measure_hsat.py <run_dir> [<run_dir> ...] [--csv out.csv]

Reads <run_dir>/oversaturation_indices.csv (read-only) and uses greens of
through (lane index 1) and left (lane index 2) lanes whose queue never
cleared. Lane index 0 is the uncontrolled right turn and is never used.

A green is skipped for the first reason that applies, in this order:
    first_green   the lane's first green of the run
    blocked       blocked_flag set (a blockage covered the lane)
    spillback     sosi > 0 on any connection of the lane
    queue_cleared tosi == 0 (the queue at green start fully discharged)
    zero_exits    no stop-line exits in the window

For a used green, window = t_allred - t_green_start (green + yellow) and
    h = (window - lost) / exited_green
for lost = 3 s (main) and lost = 0 s (sensitivity).

The runs carry no per-step stop-line exit series, so exits in the first
`lost` seconds cannot be removed: exited_green is the whole-window count.
The per-step exit_count of the shadow log includes lane changes and is not
used.

Only runs logged after exits were split by kind (C1b) are accepted, i.e.
those whose CSV has an lc_out_green column; before that, exited_green also
counted lane changes.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

INDICES_FILENAME = "oversaturation_indices.csv"
POST_C1B_COLUMN = "lc_out_green"
LANE_ROLES = {"1": "through", "2": "left"}
ROLE_ORDER = ["through", "left"]
LOST_VALUES_S = [3, 0]
SKIP_REASONS = ["first_green", "blocked", "spillback", "queue_cleared", "zero_exits"]
ALL_RUNS_LABEL = "ALL"

CSV_COLUMNS = ["run", "tls", "lane", "role", "t_green_start", "t_green_end", "t_allred",
               "window_s", "green_s", "tosi", "max_sosi", "blocked_flag", "first_green",
               "exited_green", "skip_reason"] + [f"h_lost{lost}" for lost in LOST_VALUES_S]


# --- reading -------------------------------------------------------------------------

def lane_role(lane):
    """SUMO lane ids are '<edge id>_<lane index>'."""
    return LANE_ROLES.get(lane.rsplit("_", 1)[1])


def read_indices_rows(run_dir):
    path = run_dir / INDICES_FILENAME
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if POST_C1B_COLUMN not in reader.fieldnames:
            sys.exit(f"{path}: no {POST_C1B_COLUMN} column -- logged before C1b, "
                     f"exited_green there includes lane changes")
        return list(reader)


def parse_optional_float(text):
    return None if text == "" else float(text)


def max_sosi(sosi_values):
    """sosi is blank for greens too short to score; those count as no spillback."""
    known = [value for value in sosi_values if value is not None]
    return max(known) if known else None


def group_greens(run_label, rows):
    """One record per (tls, lane, green) for through and left lanes. Green-level
    columns repeat on every connection row; sosi is per connection."""
    greens = {}
    for row in rows:
        role = lane_role(row["lane"])
        if role is None:
            continue
        key = (row["tls"], row["lane"], int(row["t_green_start"]))
        if key not in greens:
            greens[key] = {
                "run": run_label, "tls": row["tls"], "lane": row["lane"], "role": role,
                "t_green_start": int(row["t_green_start"]),
                "t_green_end": int(row["t_green_end"]),
                "t_allred": int(row["t_allred"]),
                "green_s": int(row["green_s"]),
                "tosi": parse_optional_float(row["tosi"]),
                "blocked_flag": row["blocked_flag"] == "1",
                "first_green": row["first_green"] == "1",
                "exited_green": int(row["exited_green"]),
                "sosi_values": [],
            }
        greens[key]["sosi_values"].append(parse_optional_float(row["sosi"]))
    for green in greens.values():
        green["max_sosi"] = max_sosi(green.pop("sosi_values"))
        green["window_s"] = green["t_allred"] - green["t_green_start"]
    return list(greens.values())


def load_greens(run_dir):
    return group_greens(run_dir.name, read_indices_rows(run_dir))


# --- selection and h_sat ---------------------------------------------------------------

def skip_reason(green):
    if green["first_green"]:
        return "first_green"
    if green["blocked_flag"]:
        return "blocked"
    if green["max_sosi"] is not None and green["max_sosi"] > 0:
        return "spillback"
    if not green["tosi"] > 0:
        return "queue_cleared"
    if green["exited_green"] == 0:
        return "zero_exits"
    return None


def saturation_headway(green, lost_s):
    return (green["window_s"] - lost_s) / green["exited_green"]


def evaluate_green(green):
    green["skip_reason"] = skip_reason(green)
    used = green["skip_reason"] is None
    for lost in LOST_VALUES_S:
        green[f"h_lost{lost}"] = saturation_headway(green, lost) if used else None
    return green


# --- summary ------------------------------------------------------------------------------

def quartiles(values):
    """(Q1, Q3), linear interpolation between order statistics."""
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def summarize(values):
    if not values:
        return {"n": 0}
    q1, q3 = quartiles(values)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "q1": q1, "q3": q3, "iqr": q3 - q1}


def skip_counts(greens):
    counts = {reason: 0 for reason in SKIP_REASONS}
    for green in greens:
        if green["skip_reason"] is not None:
            counts[green["skip_reason"]] += 1
    return counts


def format_number(value):
    return "-" if value is None else f"{value:.3f}"


def print_role_table(label, greens):
    print(f"\n=== {label} ===")
    header = f"{'role':<8} {'lost_s':>6} {'n_used':>6} {'mean':>7} {'median':>7} " \
             f"{'Q1':>7} {'Q3':>7} {'IQR':>7}"
    print(header)
    print("-" * len(header))
    for role in ROLE_ORDER:
        role_greens = [g for g in greens if g["role"] == role]
        for lost in LOST_VALUES_S:
            values = [g[f"h_lost{lost}"] for g in role_greens if g["skip_reason"] is None]
            stats = summarize(values)
            print(f"{role:<8} {lost:>6} {stats['n']:>6} "
                  f"{format_number(stats.get('mean')):>7} "
                  f"{format_number(stats.get('median')):>7} "
                  f"{format_number(stats.get('q1')):>7} "
                  f"{format_number(stats.get('q3')):>7} "
                  f"{format_number(stats.get('iqr')):>7}")
    print("skipped greens (first matching reason; same for every lost value):")
    for role in ROLE_ORDER:
        role_greens = [g for g in greens if g["role"] == role]
        counts = skip_counts(role_greens)
        listed = "  ".join(f"{reason}={counts[reason]}" for reason in SKIP_REASONS)
        print(f"  {role:<8} total={len(role_greens)}  {listed}")


# --- output -------------------------------------------------------------------------------

def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_per_green_csv(path, greens):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for green in greens:
            writer.writerow([csv_cell(green[column]) for column in CSV_COLUMNS])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("hsat_per_green.csv"),
                        help="per-green output CSV (default: ./hsat_per_green.csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_greens = {run_dir.name: [evaluate_green(g) for g in load_greens(run_dir)]
                  for run_dir in args.run_dirs}
    print("note: no per-step stop-line exits in these runs; h uses the whole-window "
          "exited_green count (exits in the first `lost` seconds are not removed)")
    for label, greens in run_greens.items():
        print_role_table(label, greens)
    all_greens = [green for greens in run_greens.values() for green in greens]
    if len(run_greens) > 1:
        print_role_table(ALL_RUNS_LABEL, all_greens)
    write_per_green_csv(args.csv, all_greens)
    print(f"\nper-green CSV: {args.csv} ({len(all_greens)} greens)")


if __name__ == "__main__":
    main()
