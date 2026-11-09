"""Tests for the ruletable arm's wiring (ruletable_controller.py): when a
selector plan is installed, what gets logged, and when the selector runs.

    uv run --with pytest python -m pytest tests/test_ruletable_controller.py
    python tests/test_ruletable_controller.py     # without pytest

Fake run context; no SUMO.
"""
import csv
import dataclasses
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import (  # noqa: E402
    COMMANDS_FILENAME, INTERSECTION_CONFIGS, PLAN_CHANGES_FILENAME, SELECTOR_DECISIONS_FILENAME,
)
from ruletable_controller import (  # noqa: E402
    PendingPlan, RuleTableController, SelectorDecisionsLog, starts_new_cycle,
)
from runner_baselines import default_cyclic_plan  # noqa: E402
from utils.executor_log import ExecutorLog  # noqa: E402
from utils.movement_flags import FlagConfig  # noqa: E402
from utils.selector import NEW_COMMAND, ROW_COLUMNS  # noqa: E402

CONF = INTERSECTION_CONFIGS["three_lane"]
TLS = "tls"


def fake_ctx(records_dir, t=0):
    return SimpleNamespace(
        records_dir=records_dir,
        handlers={TLS: None},
        env=SimpleNamespace(get_current_step=lambda: t),
        diagnoser=SimpleNamespace(cycle_s=140, cycle_listener=None, lane_tls={}),
        oversaturation=SimpleNamespace(trackers=[], listener=None),
        movement_flags=SimpleNamespace(active_branches=lambda: {}),
        executor_log=ExecutorLog(records_dir),
    )


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def issued_plan():
    base = default_cyclic_plan(CONF)
    plan = dataclasses.replace(base, splits=dict(base.splits, ETWT=40, NTST=20))
    row = {"outcome": NEW_COMMAND, "phase": "ETWT", "dg_given": 10, "extent_id": 1,
           "regime": "L"}
    return plan, row


def test_starts_new_cycle_only_after_the_last_phase():
    plan = default_cyclic_plan(CONF)
    assert [starts_new_cycle(plan, phase) for phase in plan.order] == [False, False, False, True]
    assert starts_new_cycle(plan, "not_a_phase")


def test_pending_plan_waits_for_the_cycle_boundary():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = fake_ctx(Path(tmp), t=275)
        controller = RuleTableController(CONF, ctx, FlagConfig())
        applied = []
        controller.selector.mark_plan_applied = lambda tls, t: applied.append((tls, t))
        plan, row = issued_plan()
        controller.pending[TLS] = PendingPlan(plan, t_issued=139, row=row)

        controller.choose(TLS, SimpleNamespace(current_phase="ETWT"))
        assert TLS in controller.pending and applied == []

        decision, _state = controller.choose(TLS, SimpleNamespace(current_phase="NLSL"))
        assert decision == ("ETWT", 40)
        assert controller.cyclic.plans[TLS] == plan and TLS not in controller.pending
        assert applied == [(TLS, 275)]
        ctx.executor_log.close()
        controller.close()
        changes = read_rows(Path(tmp) / PLAN_CHANGES_FILENAME)
        commands = read_rows(Path(tmp) / COMMANDS_FILENAME)
        assert [change["t"] for change in changes] == ["275"]
        assert [(c["t_issued"], c["effective_at"], c["primitive"]) for c in commands] == [
            ("139", "275", NEW_COMMAND)]


def test_partial_last_cycle_is_not_evaluated():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = fake_ctx(Path(tmp), t=150)
        controller = RuleTableController(CONF, ctx, FlagConfig())
        calls = []
        controller.selector.evaluate = lambda *args: calls.append(args) or (None, [])
        ctx.diagnoser.cycle_listener(140, {}, [])
        assert calls == []
        ctx.env.get_current_step = lambda: 279
        ctx.diagnoser.cycle_listener(140, {}, [])
        assert len(calls) == 1 and calls[0][0] == 279
        ctx.executor_log.close()
        controller.close()


def test_selector_rows_are_on_disk_as_soon_as_written():
    with tempfile.TemporaryDirectory() as tmp:
        log = SelectorDecisionsLog(Path(tmp))
        row = {column: "" for column in ROW_COLUMNS}
        row.update(t=139, intersection=TLS, outcome=NEW_COMMAND)
        log.write([row])
        rows = read_rows(Path(tmp) / SELECTOR_DECISIONS_FILENAME)
        log.close()
        assert [(r["t"], r["outcome"]) for r in rows] == [("139", NEW_COMMAND)]


if __name__ == "__main__":
    failed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
