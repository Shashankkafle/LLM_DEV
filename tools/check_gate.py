"""Counts and invariants for a finished ruletable run, from its logs only.

    python tools/check_gate.py <run_dir>

Prints command / plan-change / selector-row counts, selector outcomes, the
lag from a plan's issue to its effect, and ATT. Then checks:

- D1: every plan, and every realized cycle, is exactly one cycle long;
- D2: every planned and realized green is within [min_green, max_green];
- D3: each plan change happens when the first phase of the order turns green,
  and every realized cycle runs exactly one plan, the one in force at its
  start, in plan order;
- D4: each selector decision window starts at the receiver's first green at
  or after max(effective time of the plan in force, previous decision point);
- commands.csv effective_at and plan_changes.csv t agree one to one, and
  every selector evaluation sits on the cycle-close clock (t = -1 mod cycle).

Phase order, base splits and timings come from the run manifest's
intersection_config and cycle_s. Realized greens come from
phase_sequences/*.jsonl (a green starts at the event that sets the phase's
green string and ends at the event that sets its yellow), not from the
plans, so the checks test what SUMO actually ran. Exits non-zero on any
failure.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from runner_baselines import default_cyclic_plan  # noqa: E402

MAX_FAILURES_SHOWN = 40


class Timing:
    """The run's phase order and timings, from its manifest."""

    def __init__(self, manifest):
        conf = INTERSECTION_CONFIGS[manifest["args"]["intersection_config"]]
        settings = conf["global_settings"]
        base_plan = default_cyclic_plan(conf)
        self.order = list(base_plan.order)
        self.base_splits = dict(base_plan.splits)
        self.transition_s = settings["yellow_duration"] + settings["red_duration"]
        self.min_green = settings["min_green"]
        # Manifests written before --max_green existed have no max_green in args.
        self.max_green = manifest["args"].get("max_green", settings["max_green"])
        self.cycle_s = int(manifest["args"]["cycle_s"])


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def realized_greens(run, order):
    """{tls: [(phase, start, end)]} in time order."""
    greens = {}
    for path in sorted((run / "phase_sequences").glob("*_phase_sequence.jsonl")):
        tls = path.name.replace("_phase_sequence.jsonl", "")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        tls_greens = []
        # The start phase is green from t=0 without an event.
        open_phase, open_start = order[0], 0
        for event in events:
            name, state = event["new_phase_name"], event["new_phase"]
            if name in order and "y" in state:
                assert name == open_phase, (tls, event)
                tls_greens.append((open_phase, open_start, event["step"]))
                open_phase = None
            elif name in order:
                open_phase, open_start = name, event["step"]
        greens[tls] = tls_greens
    return greens


def cycles_of(greens, order):
    """Split one tls's greens into cycles, each starting at order[0]; a
    trailing incomplete cycle is dropped."""
    cycles, current = [], []
    for green in greens:
        if green[0] == order[0] and current:
            cycles.append(current)
            current = []
        current.append(green)
    if len(current) == len(order):
        cycles.append(current)
    return cycles


def print_counts(commands, changes, rows, manifest, summary):
    print("commands.csv rows:", len(commands))
    print("plan_changes.csv rows:", len(changes))
    print("selector_decisions.csv rows:", len(rows))
    print("outcomes:", dict(sorted(Counter(row["outcome"] for row in rows).items())))
    print("commands by primitive:", dict(Counter(c["primitive"] for c in commands)))
    print("transition types:", dict(Counter(c["transition_type"] for c in changes)))
    print("verification:", manifest.get("verification"))
    print("ATT average_travel_time_s:", summary.get("average_travel_time_s"),
          " cityflow_style_att_s:", summary.get("cityflow_style_att_s"),
          " wall_clock_s:", summary.get("run_wall_clock_s"))
    lags = Counter(int(c["effective_at"]) - int(c["t_issued"]) for c in commands)
    print("effective_at - t_issued:", dict(lags))


def check_planned_splits(changes, timing):
    """D1/D2 on the plans themselves."""
    failures = []
    for change in changes:
        for splits_key, cycle_key in (("old_splits", "old_cycle"), ("new_splits", "new_cycle")):
            splits = json.loads(change[splits_key])
            total = sum(splits.values()) + timing.transition_s * len(splits)
            if total != timing.cycle_s or int(change[cycle_key]) != timing.cycle_s:
                failures.append(f"D1 {change['tls']} t={change['t']} {splits_key} sums to {total}")
            if not all(timing.min_green <= green <= timing.max_green for green in splits.values()):
                failures.append(f"D2 {change['tls']} t={change['t']} {splits_key} {splits}")
    return failures


def plan_timelines(changes, timing):
    """{tls: [(effective t, splits)]}, starting with the base plan at t=-1."""
    timelines = defaultdict(lambda: [(-1, timing.base_splits)])
    for change in changes:
        timelines[change["tls"]].append((int(change["t"]), json.loads(change["new_splits"])))
    return timelines


def check_commands_match_changes(commands, changes):
    command_keys = sorted((c["tls"], int(c["effective_at"])) for c in commands)
    change_keys = sorted((c["tls"], int(c["t"])) for c in changes)
    if command_keys != change_keys:
        return ["commands.effective_at != plan_changes.t"]
    return []


def check_change_times(changes, greens, timing):
    """D3, part one: a plan change is a first-phase green start."""
    first_green_starts = {(tls, start) for tls, tls_greens in greens.items()
                          for phase, start, _ in tls_greens if phase == timing.order[0]}
    failures = []
    for change in changes:
        tls, t = change["tls"], int(change["t"])
        if (tls, t) not in first_green_starts:
            failures.append(f"D3 {tls} plan change t={t} is not a {timing.order[0]} green start")
    off_boundary = [(c["tls"], int(c["t"])) for c in changes if int(c["t"]) % timing.cycle_s]
    print(f"plan changes not at t = 0 mod {timing.cycle_s}:", off_boundary)
    return failures


def check_realized_cycles(greens, timelines, timing):
    """D2 on realized greens; D1 and D3 (part two) on realized cycles."""
    failures = []
    n_greens = n_cycles = 0
    for tls, tls_greens in greens.items():
        for phase, start, end in tls_greens:
            n_greens += 1
            if not timing.min_green <= end - start <= timing.max_green:
                failures.append(f"D2 realized {tls} {phase} {start}-{end}")
        for cycle in cycles_of(tls_greens, timing.order):
            n_cycles += 1
            cycle_start = cycle[0][1]
            in_force = [splits for t, splits in timelines[tls] if t <= cycle_start][-1]
            realized = {phase: end - start for phase, start, end in cycle}
            if [green[0] for green in cycle] != timing.order or realized != in_force:
                failures.append(f"D3 {tls} cycle at {cycle_start}: realized {realized} "
                                f"plan {in_force}")
            if cycle_start > 0:
                length = cycle[-1][2] + timing.transition_s - cycle_start
                if length != timing.cycle_s:
                    failures.append(f"D1 realized {tls} cycle at {cycle_start} is {length} s")
    print(f"realized greens checked: {n_greens}, cycles checked: {n_cycles}")
    return failures


def check_evaluation_clock(rows, timing):
    eval_times = sorted({int(row["t"]) for row in rows})
    off_clock = [t for t in eval_times if (t + 1) % timing.cycle_s]
    if off_clock:
        return [f"evaluate rows off the cycle-close clock: {off_clock[:5]}"]
    return []


def check_decision_windows(rows, greens, timelines):
    """D4. Only decision-point rows carry a window."""
    failures = []
    last_decision = {}
    n_windows = 0
    for row in rows:
        if row["window_first_t"] == "":
            continue
        n_windows += 1
        tls, t, phase = row["intersection"], int(row["t"]), row["phase"]
        effective = max(effective_t for effective_t, _ in timelines[tls] if effective_t < t)
        floor = max(effective, last_decision.get(tls, -1))
        receiver_starts = [start for p, start, _ in greens[tls] if p == phase and start >= floor]
        first = int(row["window_first_t"])
        if first < effective:
            failures.append(f"D4 {tls} t={t}: window_first_t {first} < effective_t {effective}")
        if not receiver_starts or first != receiver_starts[0]:
            failures.append(f"D4 {tls} t={t}: window_first_t {first}, expected "
                            f"{receiver_starts[:1]} (floor {floor})")
        last_decision[tls] = t
    print(f"decision windows checked: {n_windows}")
    return failures


def main(run_dir):
    run = Path(run_dir)
    commands = read_csv(run / "commands.csv")
    changes = read_csv(run / "plan_changes.csv")
    rows = read_csv(run / "selector_decisions.csv")
    manifest = json.loads((run / "run_manifest.json").read_text())
    summary = json.loads((run / "final_summary.json").read_text())
    timing = Timing(manifest)

    print_counts(commands, changes, rows, manifest, summary)
    greens = realized_greens(run, timing.order)
    timelines = plan_timelines(changes, timing)
    failures = (check_planned_splits(changes, timing)
                + check_commands_match_changes(commands, changes)
                + check_change_times(changes, greens, timing)
                + check_realized_cycles(greens, timelines, timing)
                + check_evaluation_clock(rows, timing)
                + check_decision_windows(rows, greens, timelines))

    print("FAILURES:" if failures else "ALL INVARIANTS HOLD")
    for failure in failures[:MAX_FAILURES_SHOWN]:
        print("  ", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
