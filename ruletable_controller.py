"""The "ruletable" arm: the A1 cyclic executor with the rule-table selector
(utils/selector.py) changing its green splits, driven by the live Diagnoser.

Only wiring lives here; the executor, diagnoser and selector are used as they
are. The timing, per intersection:

- every closed lane-green goes to selector.on_green_closed, after the
  Diagnoser (and so the flags) has seen it;
- when the Diagnoser closes a cycle (its flags, extents and regime updated),
  selector.evaluate runs for every intersection;
- a returned plan waits until that intersection next starts a cycle (the
  decision that hands the green back to the first phase of the order) and is
  installed there with CyclicController.set_plan, so no cycle ever mixes two
  plans. That decision's time is the plan's effective time, given to the
  selector with mark_plan_applied.

The base plan's cycle must equal the Diagnoser's cycle and the selector only
moves seconds between greens, so every intersection's cycle boundary falls a
fixed few seconds before each Diagnoser cycle close, and greens fall into the
same cycles for both.
"""

import csv
import dataclasses

from configurations import SELECTOR_DECISIONS_FILENAME
from runner_baselines import CyclicController, default_cyclic_plan
from utils.selector import (ADD, EXPIRE, NEW_COMMAND, ROW_COLUMNS, ExtentView, GreenRecord,
                            IntersectionLanes, Plan, RuleTableSelector, lane_branches)

PLAN_OUTCOMES = (NEW_COMMAND, ADD, EXPIRE)


# --- selector inputs built from the run ------------------------------------------

def phase_lanes_of(tls, trackers, conf):
    """{phase: lanes} for one signal: a lane is in a phase when one of its links
    is 'G' in that phase's green string."""
    lanes = {phase: [] for phase in conf["phases"]}
    for tracker in trackers:
        if tracker.tls != tls:
            continue
        for phase, phase_conf in conf["phases"].items():
            if any(phase_conf["green"][index] == "G" for index in tracker.link_indices):
                lanes[phase].append(tracker.lane)
    return lanes


def selector_plan(cyclic_plan):
    return Plan(order=tuple(cyclic_plan.order), splits=dict(cyclic_plan.splits),
                offset=cyclic_plan.offset)


def starts_new_cycle(plan, current_phase):
    """True when CyclicController.choose will next serve the first phase of
    the order, i.e. this decision opens a new cycle."""
    if current_phase not in plan.order:
        return True
    return plan.order.index(current_phase) == len(plan.order) - 1


# --- feeding closed greens to the selector ------------------------------------------

class SelectorFeed:
    """Takes the OversaturationIndices listener slot in front of the
    Diagnoser: forwards every callback to it, then hands each closed green to
    the selector."""

    def __init__(self, diagnoser, selector):
        self.diagnoser = diagnoser
        self.selector = selector

    def on_green_start(self, tracker):
        self.diagnoser.on_green_start(tracker)

    def on_step(self, tracker, step):
        self.diagnoser.on_step(tracker, step)

    def on_interval_closed(self, tracker, result):
        self.diagnoser.on_interval_closed(tracker, result)
        self.selector.on_green_closed(GreenRecord.from_interval_result(result))


# --- selector_decisions.csv ----------------------------------------------------------

class SelectorDecisionsLog:
    """Every selector row, appended and flushed as soon as it is produced."""

    def __init__(self, records_dir):
        self._file = open(records_dir / SELECTOR_DECISIONS_FILENAME, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=ROW_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, rows):
        if not rows:
            return
        self._writer.writerows(rows)
        self._file.flush()

    def close(self):
        self._file.close()


# --- the controller ---------------------------------------------------------------

@dataclasses.dataclass
class PendingPlan:
    plan: object       # CyclicPlan
    t_issued: int
    row: dict          # the selector row that returned the plan


def issuing_row(rows):
    issuing = [row for row in rows if row["outcome"] in PLAN_OUTCOMES]
    if len(issuing) != 1:
        raise RuntimeError(f"expected one plan-returning selector row, got {issuing}")
    return issuing[0]


class RuleTableController:
    """Same choose() contract as the other baselines. Built after setup_run,
    since it plugs into the run's Diagnoser and OversaturationIndices."""

    def __init__(self, conf, ctx, flag_config):
        if ctx.diagnoser is None:
            raise ValueError("ruletable needs the live Diagnoser (--movement_flags)")
        self.ctx = ctx
        self.cyclic = CyclicController(conf, default_cyclic_plan(conf))
        if self.cyclic.default_plan.cycle_length != ctx.diagnoser.cycle_s:
            raise ValueError(
                f"base plan cycle {self.cyclic.default_plan.cycle_length} s != diagnoser "
                f"cycle {ctx.diagnoser.cycle_s} s; the selector's cycles would not line up")
        trackers = ctx.oversaturation.trackers
        intersections = [IntersectionLanes(tls, phase_lanes_of(tls, trackers, conf))
                         for tls in ctx.handlers]
        self.selector = RuleTableSelector(intersections, flag_config)
        self.pending = {}
        self.decisions_log = SelectorDecisionsLog(ctx.records_dir)
        ctx.oversaturation.listener = SelectorFeed(ctx.diagnoser, self.selector)
        ctx.diagnoser.cycle_listener = self.on_cycle_closed

    def current_plan(self, tls):
        return self.cyclic.plans.get(tls, self.cyclic.default_plan)

    # --- decisions ----------------------------------------------------------------

    def choose(self, intersection_id, handler):
        if (intersection_id in self.pending
                and starts_new_cycle(self.current_plan(intersection_id), handler.current_phase)):
            self._apply_pending(intersection_id)
        return self.cyclic.choose(intersection_id, handler)

    def _apply_pending(self, tls):
        pending = self.pending.pop(tls)
        t = self.ctx.env.get_current_step()
        executor_log = self.ctx.executor_log
        self.cyclic.set_plan(tls, pending.plan, t, executor_log)
        self.selector.mark_plan_applied(tls, t)
        row = pending.row
        watch_condition = None
        if row["outcome"] != EXPIRE:
            watch_condition = {"receiver_phase": row["phase"],
                               "decision_window_greens": self.selector.decision_window_greens}
        executor_log.log_command(
            t_issued=pending.t_issued, tls=tls, primitive=row["outcome"],
            parameters={"phase": row["phase"], "dg_given": row["dg_given"],
                        "splits": pending.plan.splits},
            watch_condition=watch_condition, effective_at=t, expiry=None,
            extent_id=row["extent_id"], regime=row["regime"])

    # --- Diagnoser cycle_listener ---------------------------------------------------

    def on_cycle_closed(self, cycle_start, network_row, extent_rows):
        t = self.ctx.env.get_current_step()
        # The Diagnoser also closes the run's last, partial cycle at teardown;
        # that is no cycle boundary, so nothing is evaluated on it.
        if t + 1 < cycle_start + self.ctx.diagnoser.cycle_s:
            return
        extents = [ExtentView.from_extent_row(row, self.ctx.diagnoser.lane_tls)
                   for row in extent_rows]
        branches = lane_branches(self.ctx.movement_flags.active_branches())
        for tls in self.ctx.handlers:
            self._evaluate(t, tls, extents, branches)

    def _evaluate(self, t, tls, extents, branches):
        current = self.current_plan(tls)
        new_plan, rows = self.selector.evaluate(t, tls, selector_plan(current), extents,
                                                branches)
        self.decisions_log.write(rows)
        if new_plan is None:
            return
        if tls in self.pending:
            raise RuntimeError(f"{tls}: selector returned a plan at t={t} while the one "
                               f"issued at t={self.pending[tls].t_issued} is not applied yet")
        cyclic_plan = dataclasses.replace(current, splits=dict(new_plan.splits))
        self.pending[tls] = PendingPlan(cyclic_plan, t, issuing_row(rows))

    def close(self):
        self.decisions_log.close()
