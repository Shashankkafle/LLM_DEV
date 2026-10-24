"""Shared scaffolding for the SUMO controller runners.

Every controller (LLM, FixedTime/MaxPressure, CoLight eval) is compared on the
same seam: identical run-directory layout, recorders, environment flags, and
decision loop -- only the decision rule differs. This module holds that shared
scaffolding in one place so the runners cannot drift apart: early-exit
behavior, AWT sampling cadence, and the close-then-summarize order live here.
"""

import shutil
from datetime import datetime
from pathlib import Path

import traci

from sumo_env import SumoEnv
from utils.phase_handler import PhaseHandler
from utils.metrics_recorder import MetricsRecorder
from utils.replay_recorder import ReplayRecorder
from utils.oversaturation_indices import OversaturationIndices
from utils.movement_flags import FlagConfig, MovementFlags
from utils.blockage_manager import BlockageManager, load_scenario
from utils import llm_activity
from utils.run_manifest import add_sumo_runtime, save_manifest, finalize_manifest
from configurations import (
    DEFAULT_START_PHASE,
    LOGS_DIR_NAME,
    PHASE_SEQUENCES_DIR_NAME,
    BLOCKAGE_SCENARIO_COPY_FILENAME,
    BLOCKAGE_EVENTS_FILENAME,
)


def build_blockage_manager(scenario_path):
    """Load a blockage scenario and build its manager.

    Returns (scenario_dict, manager), or (None, None) when no scenario is
    requested -- so every runner wires blockages identically.
    """
    if not scenario_path:
        return None, None
    scenario = load_scenario(scenario_path)
    return scenario, BlockageManager(scenario["blockages"],
                                     scenario_name=scenario["scenario_name"])


def add_movement_flag_args(parser):
    """The --movement_flags CLI, shared by every runner that uses setup_run.
    Threshold defaults are None so FlagConfig stays the one place they live."""
    parser.add_argument("--movement_flags", action="store_true",
                        help="Also write movement_flags.csv: per-movement "
                             "oversaturation flags with hysteresis, built on "
                             "TOSI/SOSI. Observer only; control is unchanged.")
    parser.add_argument("--theta_in", type=float, default=None,
                        help="Queue/storage ratio a flag needs to raise.")
    parser.add_argument("--theta_out", type=float, default=None,
                        help="Ratio a green must fall below to count as clean "
                             "(<= theta_in - 0.10).")
    parser.add_argument("--n_in", type=int, default=None,
                        help="Consecutive entry greens needed to raise a flag.")
    parser.add_argument("--k_flag", type=int, default=None,
                        help="Consecutive clean greens needed to clear a flag.")


def movement_flag_config_from_args(args):
    """FlagConfig when --movement_flags is set, else None. getattr because
    tests build the args Namespace by hand."""
    if not getattr(args, "movement_flags", False):
        return None
    overrides = {name: getattr(args, name, None)
                 for name in ("theta_in", "theta_out", "n_in", "k_flag")}
    return FlagConfig(**{name: value for name, value in overrides.items()
                         if value is not None})


def create_run_dirs(test_name, run_group=None, logs_dir=None):
    """Create a timestamped records dir under logs/ plus its phase_sequences
    subdir. Returns (records_dir, phase_sequence_dir).

    run_group, when given, nests the run under logs/<run_group>/ so a sweep's
    runs sit together (run_matrix groups by experiment). The aggregator reads
    identity from run_manifest.json, so this grouping is for humans, not
    load-bearing.

    logs_dir replaces the default logs/ root. run_matrix watches the directory
    it was given for the run dir to appear, so if the runner wrote somewhere
    else the sweep would report every run as failed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(logs_dir or LOGS_DIR_NAME)
    if run_group:
        base = base / run_group
    # The timestamp has one-second resolution, so two runs of the same name
    # launched together (a parallel sweep) would otherwise share a directory
    # and interleave their decision logs. exist_ok=False makes the mkdir an
    # atomic claim; on a collision, take the next suffix.
    base.mkdir(parents=True, exist_ok=True)
    for suffix in ("", *(f"_{i}" for i in range(2, 100))):
        records_dir = base / f"{test_name}_{timestamp}{suffix}"
        try:
            records_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        phase_sequence_dir = records_dir / PHASE_SEQUENCES_DIR_NAME
        phase_sequence_dir.mkdir()
        return records_dir, phase_sequence_dir
    raise RuntimeError(f"Could not claim a run directory under {base}")


class RunContext:
    """Everything a runner needs after setup: dirs, recorders, a started env,
    and one PhaseHandler per intersection."""

    def __init__(self, records_dir, env, recorder, replay_recorder, handlers,
                 oversaturation, movement_flags=None):
        self.records_dir = records_dir
        self.env = env
        self.recorder = recorder
        self.replay_recorder = replay_recorder
        self.handlers = handlers
        self.oversaturation = oversaturation
        self.movement_flags = movement_flags


def setup_run(conf, test_name, simulation_config, run_meta, use_gui=False,
              seed=None, verbose_metrics=False, blockage_manager=None,
              manifest=None, run_group=None, logs_dir=None,
              oversat_shadow=False, oversat_listener=None, extra_sumo_args=(),
              movement_flag_config=None):
    """Build the run stack every controller shares.

    run_meta is written to replay_meta.json (and the manifest, when given, to
    run_manifest.json) up front so a crashed run still leaves enough on disk
    to be replayed and re-scored.

    movement_flag_config (a FlagConfig) turns on MovementFlags, which takes
    the single oversaturation listener slot.
    """
    if movement_flag_config is not None and oversat_listener is not None:
        raise ValueError("movement flags and oversat_listener both need the "
                         "single OversaturationIndices listener slot")
    records_dir, phase_sequence_dir = create_run_dirs(test_name, run_group,
                                                      logs_dir)
    if manifest is not None:
        save_manifest(records_dir, manifest)
    # Copy the scenario into the run dir so the run stays reproducible even if
    # the original scenario file later changes.
    if run_meta.get("blockage_scenario"):
        shutil.copy(run_meta["blockage_scenario"],
                    records_dir / BLOCKAGE_SCENARIO_COPY_FILENAME)
    if blockage_manager is not None:
        blockage_manager.set_event_log(records_dir / BLOCKAGE_EVENTS_FILENAME)
    replay_recorder = ReplayRecorder(record_dir=records_dir, meta=run_meta)
    recorder = MetricsRecorder(run_dir=records_dir, verbose=verbose_metrics,
                               phase_names=list(conf["phases"].keys()),
                               sumo_config=simulation_config,
                               blockage_manager=blockage_manager,
                               run_info=run_meta)
    env = SumoEnv(sumo_config=simulation_config, use_gui=use_gui,
                  phase_sequence_dir=phase_sequence_dir,
                  intersection_config=conf, output_dir=records_dir, seed=seed,
                  blockage_manager=blockage_manager,
                  extra_sumo_args=extra_sumo_args)
    env.start_simulation()
    recorder.record_initial_load()
    movement_flags = None
    if movement_flag_config is not None:
        movement_flags = MovementFlags(movement_flag_config, records_dir)
        oversat_listener = movement_flags
    oversaturation = OversaturationIndices(records_dir, blockage_manager,
                                           shadow=oversat_shadow,
                                           listener=oversat_listener)
    if manifest is not None:
        add_sumo_runtime(manifest, env.cmd)
        save_manifest(records_dir, manifest)
    handlers = {
        intersection_id: PhaseHandler(env=env, conf=conf,
                                      intersection_id=intersection_id,
                                      start_phase=DEFAULT_START_PHASE,
                                      replay_recorder=replay_recorder)
        for intersection_id in env.get_intersections()
    }
    return RunContext(records_dir, env, recorder, replay_recorder, handlers,
                      oversaturation, movement_flags)


def _close_and_summarize(ctx, status):
    """Close SUMO first (it flushes the statistics file on close), then write
    the final summary -- and finalize the manifest no matter what, so a run dir
    is never left claiming status "running" even if closing or summarizing
    raises. Shared by both control loops so the teardown order cannot drift.
    """
    try:
        ctx.oversaturation.close()
        if ctx.movement_flags is not None:
            ctx.movement_flags.close()
        ctx.env.close()
        summary = ctx.recorder.save_final_summary()
    finally:
        finalize_manifest(ctx.records_dir, status)
        llm_activity.close()
    return summary


def run_control_loop(ctx, simulation_steps, decide):
    """Drive the simulation with per-intersection decisions. Returns the final
    summary dict.

    ``decide(step, intersection_id, handler)`` -> next phase name, called
    whenever that intersection's green window ends. The loop owns the shared
    measurement behavior:
      - one network-wide AWT sample per decision point,
      - early exit once SUMO expects no more vehicles,
      - close-then-summarize teardown (also on a crash, so a partial run still
        leaves an honest record; the manifest then carries status "crashed").
    """
    status = "crashed"
    try:
        for step in range(simulation_steps): 
            ctx.env.step()
            ctx.recorder.record_step_summary(step)
            # Read-only; must run before handler.step() so it sees the signal
            # state that governed the step just simulated.
            ctx.oversaturation.update(ctx.env.get_current_step())
            llm_activity.set_step(step)
            for intersection_id, handler in ctx.handlers.items():
                handler.step()
                if handler.switch_phase:
                    next_phase = decide(step, intersection_id, handler)
                    ctx.recorder.record_decision_wait()
                    handler.activate_phase(next_phase)
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected at step {step}, stopping early.")
                break
        status = "completed"
    finally:
        summary = _close_and_summarize(ctx, status)
    return summary


def run_control_loop_batched(ctx, simulation_steps, decide_batch):
    """Like run_control_loop, but every intersection switching on the same step
    is handed to ``decide_batch`` together, so a batched controller can run
    their decisions as one inference call. Returns the final summary dict.

    ``decide_batch(step, pending)`` -> {intersection_id: next_phase}, where
    pending is the list of (intersection_id, handler) switching this step;
    decide_batch owns all per-decision recording.

    This is behaviourally identical to run_control_loop: the simulation advances
    exactly once per step (``env.step()``) before any decision, and setting a
    phase changes only a signal -- it never moves vehicles -- so reading every
    switching intersection's state up front, before any phase is applied, yields
    the same inputs as deciding them one at a time. The shared measurement
    behavior (one AWT sample per decision, early exit, close-then-summarize) is
    preserved: record_decision_wait() is still called once per decision.
    """
    status = "crashed"
    try:
        for step in range(simulation_steps):
            ctx.env.step()
            ctx.recorder.record_step_summary(step)
            # Read-only; must run before handler.step() so it sees the signal
            # state that governed the step just simulated.
            ctx.oversaturation.update(ctx.env.get_current_step())
            llm_activity.set_step(step)
            pending = []
            for intersection_id, handler in ctx.handlers.items():
                handler.step()
                if handler.switch_phase:
                    pending.append((intersection_id, handler))
            if pending:
                next_phases = decide_batch(step, pending)
                for intersection_id, handler in pending:
                    ctx.recorder.record_decision_wait()
                    handler.activate_phase(next_phases[intersection_id])
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected at step {step}, stopping early.")
                break
        status = "completed"
    finally:
        summary = _close_and_summarize(ctx, status)
    return summary
