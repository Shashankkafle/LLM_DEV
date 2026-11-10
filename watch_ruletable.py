"""Run the ruletable arm and print what one intersection sees and does.

Nothing here changes control. WatchedController is the real
RuleTableController with print calls added around its methods. Each one calls
super() unchanged, so the run is the same as
`runner_baselines.py --controller ruletable --movement_flags`, and it writes
the same CSVs.

    uv run python watch_ruletable.py --tls intersection_2_2 --simulation_steps 1800

Any other runner_baselines.py option (--simulation_config, --theta_in, ...)
is passed through. Leave out --tls to get the list of intersection ids.
"""

import argparse
import sys

import runner_baselines
from ruletable_controller import RuleTableController, SelectorFeed, phase_lanes_of


class RememberingLog:
    """Wraps the real selector_decisions.csv log and keeps a copy of every row,
    so the watcher can print the rows one evaluate produced."""

    def __init__(self, real_log):
        self.real_log = real_log
        self.rows = []

    @property
    def rows_written(self):
        return len(self.rows)

    def write(self, rows):
        self.real_log.write(rows)
        self.rows.extend(rows)

    def close(self):
        self.real_log.close()


class PrintingFeed(SelectorFeed):
    """Prints every closed green of the watched intersection: the per-lane
    inputs behind the flags, then each movement's flag state after the
    green (the Diagnoser has already updated it when super() returns)."""

    def __init__(self, diagnoser, selector, movement_flags, tls):
        super().__init__(diagnoser, selector)
        self.movement_flags = movement_flags
        self.tls = tls

    def on_interval_closed(self, tracker, result):
        super().on_interval_closed(tracker, result)
        if result.tls != self.tls:
            return
        print(f"  green closed  {result.lane:<14} t={result.green_start}-{result.green_end}"
              f"  tosi={result.tosi}  ratio_start={result.ratio_start}"
              f"  (halted_at_start={result.halted_count_start} / storage={result.capacity})"
              f"  held_over={result.held_over}"
              f"  entered_red={result.entered_red}  exited_green={result.exited_green}")
        for target, sosi in result.sosi.items():
            state = self.movement_flags.flags[(result.lane, target)]
            print(f"      flag -> {target:<14} sosi={sosi}  entry_count={state.entry_count}"
                  f"  clean_count={state.clean_count}  active={state.active}"
                  f"  branch={state.branch or '-'}  raised_at={state.raised_at}")


class WatchedController(RuleTableController):

    def __init__(self, conf, ctx, flag_config, tls):
        super().__init__(conf, ctx, flag_config)
        if tls not in ctx.handlers:
            raise SystemExit(f"--tls {tls!r} not found; intersections: {sorted(ctx.handlers)}")
        self.tls = tls
        phase_lanes = phase_lanes_of(tls, ctx.oversaturation.trackers, conf)
        self.my_lanes = {lane for lanes in phase_lanes.values() for lane in lanes}
        ctx.oversaturation.listener = PrintingFeed(ctx.diagnoser, self.selector,
                                                   ctx.movement_flags, tls)
        self.decisions_log = RememberingLog(self.decisions_log)
        print(f"Flag parameters: {flag_config}")
        print("  raise: tosi > 0 and ratio_start >= theta_in, for n_in greens in a row"
              " (or SOSI loss >= sosi_loss_s)")
        print("  clear: k_flag clean greens in a row"
              " (tosi == 0 and ratio_start < theta_out and SOSI loss < sosi_loss_s)")
        print(f"Watching {tls}")
        for phase, lanes in phase_lanes.items():
            print(f"  {phase}: {lanes}")
        print(f"  start plan: {self.current_plan(tls).splits}")

    def choose(self, intersection_id, handler):
        decision, state = super().choose(intersection_id, handler)
        if intersection_id == self.tls:
            t = self.ctx.env.get_current_step()
            phase, green = decision
            print(f"t={t:>5}  DECISION  {handler.current_phase} -> {phase} for {green}s\n\n")
        return decision, state

    def _apply_pending(self, tls):
        pending = self.pending[tls]
        super()._apply_pending(tls)
        if tls == self.tls:
            t = self.ctx.env.get_current_step()
            print(f"t={t:>5}  PLAN INSTALLED (issued t={pending.t_issued}): {pending.plan.splits}")

    def on_cycle_closed(self, cycle_start, network_row, extent_rows):
        t = self.ctx.env.get_current_step()
        print(f"\n\n===================== cycle {cycle_start}-{cycle_start + self.ctx.diagnoser.cycle_s} closed"
              f" at t={t}  network regime={network_row['regime']}"
              f" (active flags={network_row['n_active']}) =====")
        self.print_flags()
        self.print_extents(extent_rows)
        super().on_cycle_closed(cycle_start, network_row, extent_rows)

    def print_flags(self):
        flags = {movement: branch
                 for movement, branch in self.ctx.movement_flags.active_branches().items()
                 if movement[0] in self.my_lanes}
        print(f"  -=-=-=-=flags here: {flags if flags else 'none'}")

    def print_extents(self, extent_rows):
        touching = [row for row in extent_rows if self.my_lanes & set(row.members)]
        if not touching:
            print("  extents here: none")
        for row in touching:
            print(f"  extent {row.extent_id}: regime={row.regime}  members={sorted(row.members)}"
                  f"  E={row.entries} X={row.exits}")

    def _evaluate(self, t, tls, extents, branches):
        rows_before = self.decisions_log.rows_written
        super()._evaluate(t, tls, extents, branches)
        if tls != self.tls:
            return
        new_rows = self.decisions_log.rows[rows_before:]
        if not new_rows:
            print("  SELECTOR: no rows (nothing to act on)")
        for row in new_rows:
            print(f"  SELECTOR: outcome={row['outcome']}  extent={row['extent_id']}"
                  f"  regime={row['regime']}  phase={row['phase']}"
                  f"  dg_requested={row['dg_requested']}  dg_given={row['dg_given']}"
                  f"  donors={row['donors']}")
            print(f"            plan {row['plan_before']} -> {row['plan_after']}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tls", default=None)
    own_args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--controller", "ruletable", "--movement_flags",
                "--test_name", "watch_ruletable", *rest]
    runner_args = runner_baselines.parse_args()

    def build_watched(name, conf, ctx, flag_config):
        tls = own_args.tls or "(none given)"
        return WatchedController(conf, ctx, flag_config, tls)

    runner_baselines.build_run_controller = build_watched
    runner_baselines.main(runner_args)


if __name__ == "__main__":
    main()
