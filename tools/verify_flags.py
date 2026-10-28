"""Check a run's movement flags, regimes, extents and evaluation logs offline.

    python tools/verify_flags.py <run_dir> [<run_dir> ...]
    python tools/verify_flags.py --replay <run_dir> --out <dir>

Verify mode replays the run's logged per-green inputs (movement_flags.csv +
oversaturation_indices.csv) and logged network state (edge_timeseries.csv,
lane_halting.csv, blockage_events.jsonl) through the live MovementFlags,
Diagnoser and EvaluationLogs, and diffs the result against the run's own
movement_flags.csv (flag state columns, plus blocked_event_t, q_start and
storage where the run logged them), regime.csv, extents.csv, ground_truth.csv
and network_metrics.csv. It prints the first differing row of each file and
exits non-zero on any difference.

Replay mode writes those five files into <dir> as the live code would have
written them for the logged inputs, copies the run's other input and
evaluation files beside them, and writes a run_manifest.json carrying the
resolved settings, so <dir> is itself a run folder that can be verified and
evaluated (tools/eval_metrics.py). This is how older runs are re-flagged
under the current rule without SUMO. Give <dir> a new path: replaying onto
the run dir itself overwrites its original flag columns.

Thresholds and cycle come from the run's manifest args where set, else from
the current defaults (FlagConfig, --cycle_s). Runs logged before q_start and
storage were columns get them from the net file (storage = lane length / 7.5
m, q_start = ratio_start x storage); runs before lane_halting.csv get an
empty side_street_halted.

The logged inputs are rounded to 4 decimals. A live green whose SOSI x
green_s lands within rounding of L can flag differently from its replay.
"""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import (  # noqa: E402
    BLOCKAGE_EVENTS_FILENAME,
    BLOCKAGE_SCENARIO_COPY_FILENAME,
    COMMANDS_FILENAME,
    EDGE_TIMESERIES_FILENAME,
    EXTENTS_FILENAME,
    FINAL_SUMMARY_FILENAME,
    GROUND_TRUTH_FILENAME,
    LANE_ENTRIES_FILENAME,
    LANE_HALTING_FILENAME,
    MOVEMENT_FLAGS_FILENAME,
    NETWORK_METRICS_FILENAME,
    NETWORK_TIMESERIES_FILENAME,
    OVERSATURATION_INDICES_FILENAME,
    PLAN_CHANGES_FILENAME,
    REGIME_FILENAME,
    SUMO_TRIPINFO_FILENAME,
    SURGE_TIMESERIES_FILENAME,
)
from utils.diagnoser import DEFAULT_CYCLE_S, Diagnoser  # noqa: E402
from utils.evaluation_logs import EvaluationLogs  # noqa: E402
from utils.lane_state_log import LoggedNetworkState  # noqa: E402
from utils.movement_flags import FlagConfig, MovementFlags  # noqa: E402
from utils.network_topology import load_surge_spec, resolve_sumocfg  # noqa: E402
from utils.run_logs import load_network, read_manifest  # noqa: E402

RUN_MANIFEST_FILENAME = "run_manifest.json"
REPLAY_MANIFEST_FILENAME = "replay_manifest.json"
FLAG_STATE_COLUMNS = ["skipped", "entry_count", "clean_count", "active",
                      "raised_at", "branch"]
# Compared only when the live file has them (runs before they were logged don't).
LOGGED_INPUT_COLUMNS = ["q_start", "storage", "blocked_event_t"]
MOVEMENT_KEY = ["t_close", "tls", "lane", "target_lane"]
# Copied into a replay dir so it is a complete run folder for verify and
# eval_metrics. The five replayed files are written, not copied.
RUN_FILES_TO_COPY = [
    OVERSATURATION_INDICES_FILENAME, EDGE_TIMESERIES_FILENAME,
    NETWORK_TIMESERIES_FILENAME, SURGE_TIMESERIES_FILENAME, LANE_HALTING_FILENAME,
    LANE_ENTRIES_FILENAME, BLOCKAGE_EVENTS_FILENAME, BLOCKAGE_SCENARIO_COPY_FILENAME,
    SUMO_TRIPINFO_FILENAME, FINAL_SUMMARY_FILENAME, COMMANDS_FILENAME,
    PLAN_CHANGES_FILENAME,
]
BLOCKAGE_FLAG_EVENTS = ("activated", "deactivated")


# --- settings ---------------------------------------------------------------------

def manifest_args(run_dir):
    for name in (RUN_MANIFEST_FILENAME, REPLAY_MANIFEST_FILENAME):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")).get("args") or {}
    return {}


def resolve_settings(run_dir):
    """(FlagConfig, cycle_s) the run used, falling back to the defaults for
    anything its manifest leaves unset (the v3 batch logged null)."""
    args = manifest_args(run_dir)
    overrides = {f.name: args[f.name] for f in fields(FlagConfig)
                 if args.get(f.name) is not None}
    return FlagConfig(**overrides), args.get("cycle_s") or DEFAULT_CYCLE_S


# --- logged greens -------------------------------------------------------------------

@dataclass
class LoggedGreen:
    """The IntervalResult fields MovementFlags and EvaluationLogs read,
    rebuilt from the logs."""
    tls: str
    lane: str
    allred: int
    green_s: int
    tosi: object
    ratio_start: object
    ratio_end: object
    halted_count_start: object
    halted_count_end: int
    capacity: object
    discharged: int
    blocked_flag: bool
    held_over: int
    stop_halted_steps: int
    no_exit_steps: int
    sosi: dict = field(default_factory=dict)
    target_halted_steps: dict = field(default_factory=dict)


def parse_float(text):
    return None if text == "" else float(text)


def parse_int(text):
    return None if text == "" else int(text)


def parse_link_indices(text):
    return [int(index) for index in text.split(";")] if text else []


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def read_green_lengths(run_dir):
    """(t_allred, tls, lane, target_lane) -> green_s."""
    _, rows = read_csv(run_dir / OVERSATURATION_INDICES_FILENAME)
    return {(row["t_allred"], row["tls"], row["lane"], row["target_lane"]): int(row["green_s"])
            for row in rows}


def queue_counts(row, network):
    """(q_start, storage) from the row's own columns, or for runs logged
    before them, from the net: storage = lane capacity, q_start =
    ratio_start x storage (exact: ratio_start is q_start / storage to 4
    decimals and storage is far below 10^4)."""
    if row.get("storage", "") != "":
        return parse_int(row["q_start"]), int(row["storage"])
    if network is None or row["lane"] not in network.lane_length:
        return None, None
    storage = network.storage(row["lane"])
    ratio_start = parse_float(row["ratio_start"])
    return (None if ratio_start is None else round(ratio_start * storage)), storage


def new_logged_green(row, green_s, network):
    q_start, storage = queue_counts(row, network)
    return LoggedGreen(
        tls=row["tls"], lane=row["lane"], allred=int(row["t_close"]), green_s=green_s,
        tosi=parse_float(row["tosi"]), ratio_start=parse_float(row["ratio_start"]),
        ratio_end=parse_float(row["ratio_end"]), halted_count_start=q_start,
        halted_count_end=int(row["halted_count_end"]), capacity=storage,
        discharged=int(row["discharged"]),
        blocked_flag=row["blocked"] == "1", held_over=int(row["held_over"]),
        stop_halted_steps=int(row["sosi_c1"]), no_exit_steps=int(row["sosi_c2"]))


def load_logged_greens(run_dir, network=None):
    """One (LoggedGreen, tracker stub) per closed lane green, in log order.
    The stub carries the only tracker field MovementFlags reads. network is
    only needed for runs logged without the q_start and storage columns."""
    green_lengths = read_green_lengths(run_dir)
    _, rows = read_csv(run_dir / MOVEMENT_FLAGS_FILENAME)
    greens = {}
    for row in rows:
        target = row["target_lane"]
        green_key = (row["t_close"], row["tls"], row["lane"])
        if green_key not in greens:
            green_s = green_lengths[(*green_key, target)]
            greens[green_key] = (new_logged_green(row, green_s, network),
                                 SimpleNamespace(target_link_indices={}))
        green, tracker = greens[green_key]
        green.sosi[target] = parse_float(row["sosi"])
        green.target_halted_steps[target] = int(row["sosi_c3"])
        tracker.target_link_indices[target] = parse_link_indices(row["link_indices"])
    return list(greens.values())


def has_queue_columns(run_dir):
    columns, _ = read_csv(run_dir / MOVEMENT_FLAGS_FILENAME)
    return "storage" in columns


def load_blockage_events(run_dir):
    """(sim_time, lane) per raise or clear of a blockage event flag -- the
    same pairs the BlockageManager hands MovementFlags live."""
    path = run_dir / BLOCKAGE_EVENTS_FILENAME
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["event"] in BLOCKAGE_FLAG_EVENTS:
            events.append((record["sim_time"], record["lane_id"]))
    return events


# --- replay -------------------------------------------------------------------------

def feed_greens(diagnoser, greens, last_t=None):
    """Step the cycle clock second by second up to the run's last second,
    closing each green at its t_close before the clock reads that second --
    the live order. last_t defaults to the last t_close."""
    by_close = defaultdict(list)
    for green, tracker in greens:
        by_close[green.allred].append((green, tracker))
    end = max([*by_close, last_t if last_t is not None else 0])
    for t in range(end + 1):
        for green, tracker in by_close.get(t, ()):
            diagnoser.on_interval_closed(tracker, green)
        diagnoser.update(t)


def copy_run_files(run_dir, out_dir):
    if run_dir.resolve() == out_dir.resolve():
        return
    for name in RUN_FILES_TO_COPY:
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, out_dir / name)


def replayed_manifest(run_dir, config, cycle_s):
    """The run's manifest with the settings the replay used in args and the
    surge spec filled in when the run predates it."""
    manifest = read_manifest(run_dir)
    manifest["args"] = {**(manifest.get("args") or {}), **vars(config), "cycle_s": cycle_s}
    if manifest.get("surge") is None:
        sumocfg = resolve_sumocfg(manifest)
        manifest["surge"] = load_surge_spec(sumocfg) if sumocfg is not None else None
    manifest["replay_of"] = str(run_dir)
    return manifest


def replay(run_dir, out_dir, write_run_folder=True):
    """Replay run_dir into out_dir. write_run_folder=False writes only the
    five replayed files (all verify needs)."""
    config, cycle_s = resolve_settings(run_dir)
    network = None if has_queue_columns(run_dir) else load_network(read_manifest(run_dir))
    greens = load_logged_greens(run_dir, network)
    network_state = LoggedNetworkState(run_dir)
    blockage_events = load_blockage_events(run_dir)
    manifest = replayed_manifest(run_dir, config, cycle_s) if write_run_folder else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_run_folder:
        copy_run_files(run_dir, out_dir)
    flags = MovementFlags(config, out_dir, blockage_events)
    diagnoser = Diagnoser(flags, cycle_s, out_dir, EvaluationLogs(out_dir, network_state))
    try:
        feed_greens(diagnoser, greens, network_state.last_t())
    finally:
        diagnoser.close()
        flags.close()
    if manifest is not None:
        (out_dir / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")


# --- diff ---------------------------------------------------------------------------

def movement_sort_key(key):
    t_close, tls, lane, target = key
    return int(t_close), tls, lane, target


def first_flag_difference(live, replayed, columns):
    for key in sorted(replayed, key=movement_sort_key):
        for column in columns:
            if live[key][column] != replayed[key][column]:
                where = ", ".join(f"{c}={v}" for c, v in zip(MOVEMENT_KEY, key))
                return (f"{where}: {column} live={live[key][column]!r} "
                        f"replay={replayed[key][column]!r}")
    return None


def diff_flags(live_path, replay_path):
    """Returns (rows compared, first difference or None)."""
    live_columns, live_rows = read_csv(live_path)
    _, replay_rows = read_csv(replay_path)
    live = {tuple(row[c] for c in MOVEMENT_KEY): row for row in live_rows}
    replayed = {tuple(row[c] for c in MOVEMENT_KEY): row for row in replay_rows}
    if live.keys() != replayed.keys():
        only_live = sorted(live.keys() - replayed.keys(), key=movement_sort_key)
        only_replay = sorted(replayed.keys() - live.keys(), key=movement_sort_key)
        return len(replay_rows), (f"row keys differ: {len(only_live)} only live "
                                  f"(first {only_live[:1]}), {len(only_replay)} only "
                                  f"replay (first {only_replay[:1]})")
    missing = [c for c in FLAG_STATE_COLUMNS if c not in live_columns]
    present = [c for c in FLAG_STATE_COLUMNS + LOGGED_INPUT_COLUMNS if c in live_columns]
    problems = [f"live file has no column(s) {missing}"] if missing else []
    first = first_flag_difference(live, replayed, present)
    if first is not None:
        problems.append(first)
    return len(replay_rows), "; ".join(problems) or None


def diff_table(live_path, replay_path):
    """Row-by-row diff of two CSVs. Returns (rows compared, first difference or None)."""
    if not live_path.exists():
        return 0, "live file missing"
    live_columns, live_rows = read_csv(live_path)
    replay_columns, replay_rows = read_csv(replay_path)
    if live_columns != replay_columns:
        return len(replay_rows), f"columns differ: live {live_columns} replay {replay_columns}"
    for index, (live_row, replay_row) in enumerate(zip(live_rows, replay_rows)):
        if live_row != replay_row:
            return len(replay_rows), (f"row {index + 1}: live {list(live_row.values())} "
                                      f"replay {list(replay_row.values())}")
    if len(live_rows) != len(replay_rows):
        return len(replay_rows), (f"row count live {len(live_rows)} "
                                  f"replay {len(replay_rows)}")
    return len(replay_rows), None


TABLE_FILENAMES = [REGIME_FILENAME, EXTENTS_FILENAME, GROUND_TRUTH_FILENAME,
                   NETWORK_METRICS_FILENAME]


def verify_run(run_dir):
    """[(filename, rows compared, first difference or None)] for the five files."""
    with tempfile.TemporaryDirectory() as tmp:
        replay_dir = Path(tmp)
        replay(run_dir, replay_dir, write_run_folder=False)
        results = [(MOVEMENT_FLAGS_FILENAME,
                    *diff_flags(run_dir / MOVEMENT_FLAGS_FILENAME,
                                replay_dir / MOVEMENT_FLAGS_FILENAME))]
        results += [(name, *diff_table(run_dir / name, replay_dir / name))
                    for name in TABLE_FILENAMES]
    return results


def first_difference(results):
    """'<file>: <difference>' for the first file that differs, or None."""
    for name, _, difference in results:
        if difference is not None:
            return f"{name}: {difference}"
    return None


def verification_record(run_dir):
    """The manifest's verification block: {passed, first_diff}."""
    diff = first_difference(verify_run(run_dir))
    return {"passed": diff is None, "first_diff": diff}


def verify(run_dir):
    """Prints one line per run (and one per differing file) and returns True
    when all five files match."""
    results = verify_run(run_dir)
    ok = all(difference is None for _, _, difference in results)
    summary = ", ".join(f"{name} {count} rows" for name, count, _ in results)
    print(f"{'OK  ' if ok else 'FAIL'} {run_dir.name}: {summary}")
    for name, _, difference in results:
        if difference is not None:
            print(f"     {name}: {difference}")
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", type=Path, help="Run dirs to verify.")
    parser.add_argument("--replay", type=Path, help="Run dir to replay instead.")
    parser.add_argument("--out", type=Path, help="Where --replay writes its files.")
    args = parser.parse_args()
    if args.replay is not None and (args.out is None or args.run_dirs):
        parser.error("--replay takes one run dir and needs --out")
    if args.replay is None and not args.run_dirs:
        parser.error("give run dirs to verify, or --replay <run_dir> --out <dir>")
    return args


def main():
    args = parse_args()
    if args.replay is not None:
        replay(args.replay, args.out)
        print(f"replayed {args.replay} -> {args.out}")
        return 0
    results = [verify(run_dir) for run_dir in args.run_dirs]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
