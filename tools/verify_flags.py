"""Check a run's movement flags, network regime and extents offline.

    python tools/verify_flags.py <run_dir> [<run_dir> ...]
    python tools/verify_flags.py --replay <run_dir> --out <dir>

Verify mode replays the run's logged per-green inputs (movement_flags.csv +
oversaturation_indices.csv) through the live MovementFlags and Diagnoser, and
diffs the result against the run's own movement_flags.csv (flag state
columns), regime.csv and extents.csv. It prints the first differing row of
each file and exits non-zero on any difference.

Replay mode writes the three files the live code would have written for those
logged inputs into <dir>, plus a copy of oversaturation_indices.csv, so <dir>
can itself be verified. This is how older runs are re-flagged under the
current rule without SUMO.

Thresholds and cycle come from the run's manifest args where set, else from
the current defaults (FlagConfig, --cycle_s).

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
    EXTENTS_FILENAME,
    MOVEMENT_FLAGS_FILENAME,
    OVERSATURATION_INDICES_FILENAME,
    REGIME_FILENAME,
)
from utils.diagnoser import DEFAULT_CYCLE_S, Diagnoser  # noqa: E402
from utils.movement_flags import FlagConfig, MovementFlags  # noqa: E402

RUN_MANIFEST_FILENAME = "run_manifest.json"
REPLAY_MANIFEST_FILENAME = "replay_manifest.json"
FLAG_STATE_COLUMNS = ["skipped", "entry_count", "clean_count", "active",
                      "raised_at", "branch"]
MOVEMENT_KEY = ["t_close", "tls", "lane", "target_lane"]


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
    """The IntervalResult fields MovementFlags reads, rebuilt from the logs."""
    tls: str
    lane: str
    allred: int
    green_s: int
    tosi: object
    ratio_start: object
    ratio_end: object
    halted_count_end: int
    discharged: int
    blocked_flag: bool
    held_over: int
    stop_halted_steps: int
    no_exit_steps: int
    sosi: dict = field(default_factory=dict)
    target_halted_steps: dict = field(default_factory=dict)


def parse_float(text):
    return None if text == "" else float(text)


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


def new_logged_green(row, green_s):
    return LoggedGreen(
        tls=row["tls"], lane=row["lane"], allred=int(row["t_close"]), green_s=green_s,
        tosi=parse_float(row["tosi"]), ratio_start=parse_float(row["ratio_start"]),
        ratio_end=parse_float(row["ratio_end"]),
        halted_count_end=int(row["halted_count_end"]), discharged=int(row["discharged"]),
        blocked_flag=row["blocked"] == "1", held_over=int(row["held_over"]),
        stop_halted_steps=int(row["sosi_c1"]), no_exit_steps=int(row["sosi_c2"]))


def load_logged_greens(run_dir):
    """One (LoggedGreen, tracker stub) per closed lane green, in log order.
    The stub carries the only tracker field MovementFlags reads."""
    green_lengths = read_green_lengths(run_dir)
    _, rows = read_csv(run_dir / MOVEMENT_FLAGS_FILENAME)
    greens = {}
    for row in rows:
        target = row["target_lane"]
        green_key = (row["t_close"], row["tls"], row["lane"])
        if green_key not in greens:
            green_s = green_lengths[(*green_key, target)]
            greens[green_key] = (new_logged_green(row, green_s),
                                 SimpleNamespace(target_link_indices={}))
        green, tracker = greens[green_key]
        green.sosi[target] = parse_float(row["sosi"])
        green.target_halted_steps[target] = int(row["sosi_c3"])
        tracker.target_link_indices[target] = parse_link_indices(row["link_indices"])
    return list(greens.values())


# --- replay -------------------------------------------------------------------------

def feed_greens(diagnoser, greens):
    """Step the cycle clock second by second, closing each green at its
    t_close before the clock reads that second -- the live order."""
    by_close = defaultdict(list)
    for green, tracker in greens:
        by_close[green.allred].append((green, tracker))
    for t in range(max(by_close) + 1):
        for green, tracker in by_close.get(t, ()):
            diagnoser.on_interval_closed(tracker, green)
        diagnoser.update(t)


def replay(run_dir, out_dir):
    config, cycle_s = resolve_settings(run_dir)
    greens = load_logged_greens(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = MovementFlags(config, out_dir)
    diagnoser = Diagnoser(flags, cycle_s, out_dir)
    try:
        feed_greens(diagnoser, greens)
    finally:
        diagnoser.close()
        flags.close()
    shutil.copy(run_dir / OVERSATURATION_INDICES_FILENAME,
                out_dir / OVERSATURATION_INDICES_FILENAME)
    replay_manifest = {"replay_of": str(run_dir),
                       "args": {**vars(config), "cycle_s": cycle_s}}
    (out_dir / REPLAY_MANIFEST_FILENAME).write_text(
        json.dumps(replay_manifest, indent=2), encoding="utf-8")


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
    present = [c for c in FLAG_STATE_COLUMNS if c in live_columns]
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


def verify(run_dir):
    """Prints one line per file and returns True when all three match."""
    with tempfile.TemporaryDirectory() as tmp:
        replay_dir = Path(tmp)
        replay(run_dir, replay_dir)
        results = [
            (MOVEMENT_FLAGS_FILENAME, *diff_flags(run_dir / MOVEMENT_FLAGS_FILENAME,
                                                  replay_dir / MOVEMENT_FLAGS_FILENAME)),
            (REGIME_FILENAME, *diff_table(run_dir / REGIME_FILENAME,
                                          replay_dir / REGIME_FILENAME)),
            (EXTENTS_FILENAME, *diff_table(run_dir / EXTENTS_FILENAME,
                                           replay_dir / EXTENTS_FILENAME)),
        ]
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
