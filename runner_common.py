"""Shared scaffolding for the SUMO controller runners.

Every controller (LLM, FixedTime/MaxPressure, CoLight eval) is compared on the
same seam: identical run-directory layout, recorders, environment flags, and
decision loop -- only the decision rule differs. This module holds that shared
scaffolding in one place so the runners cannot drift apart: early-exit
behavior, AWT sampling cadence, and the close-then-summarize order live here.
"""

from datetime import datetime
from pathlib import Path

import traci

from sumo_env import SumoEnv
from utils.phase_handler import PhaseHandler
from utils.metrics_recorder import MetricsRecorder
from utils.replay_recorder import ReplayRecorder
from utils.blockage_manager import BlockageManager, load_scenario
from configurations import (
    DEFAULT_START_PHASE,
    LOGS_DIR_NAME,
    PHASE_SEQUENCES_DIR_NAME,
)


def build_blockage_manager(scenario_path):
    """Load a blockage scenario and build its manager.

    Returns (scenario_dict, manager), or (None, None) when no scenario is
    requested -- so every runner wires blockages identically.
    """
    if not scenario_path:
        return None, None
    scenario = load_scenario(scenario_path)
    return scenario, BlockageManager(scenario["blockages"])


def create_run_dirs(test_name):
    """Create a timestamped records dir under logs/ plus its phase_sequences
    subdir. Returns (records_dir, phase_sequence_dir)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = Path(LOGS_DIR_NAME) / f"{test_name}_{timestamp}"
    phase_sequence_dir = records_dir / PHASE_SEQUENCES_DIR_NAME
    phase_sequence_dir.mkdir(parents=True, exist_ok=True)
    return records_dir, phase_sequence_dir


class RunContext:
    """Everything a runner needs after setup: dirs, recorders, a started env,
    and one PhaseHandler per intersection."""

    def __init__(self, records_dir, env, recorder, replay_recorder, handlers):
        self.records_dir = records_dir
        self.env = env
        self.recorder = recorder
        self.replay_recorder = replay_recorder
        self.handlers = handlers


def setup_run(conf, test_name, simulation_config, run_meta, use_gui=False,
              seed=None, verbose_metrics=False, blockage_manager=None):
    """Build the run stack every controller shares.

    run_meta is written to replay_meta.json up front so a crashed run still
    leaves enough on disk to be replayed and re-scored.
    """
    records_dir, phase_sequence_dir = create_run_dirs(test_name)
    replay_recorder = ReplayRecorder(record_dir=records_dir, meta=run_meta)
    recorder = MetricsRecorder(run_dir=records_dir, verbose=verbose_metrics,
                               phase_names=list(conf["phases"].keys()),
                               sumo_config=simulation_config,
                               blockage_manager=blockage_manager)
    env = SumoEnv(sumo_config=simulation_config, use_gui=use_gui,
                  phase_sequence_dir=phase_sequence_dir,
                  intersection_config=conf, output_dir=records_dir, seed=seed,
                  blockage_manager=blockage_manager)
    env.start_simulation()
    handlers = {
        intersection_id: PhaseHandler(env=env, conf=conf,
                                      intersection_id=intersection_id,
                                      start_phase=DEFAULT_START_PHASE,
                                      replay_recorder=replay_recorder)
        for intersection_id in env.get_intersections()
    }
    return RunContext(records_dir, env, recorder, replay_recorder, handlers)


def run_control_loop(ctx, simulation_steps, decide):
    """Drive the simulation with per-intersection decisions. Returns the final
    summary dict.

    ``decide(step, intersection_id, handler)`` -> next phase name, called
    whenever that intersection's green window ends. The loop owns the shared
    measurement behavior:
      - one network-wide AWT sample per decision point,
      - early exit once SUMO expects no more vehicles,
      - close SUMO first (it flushes the statistics file on close), then write
        the final summary -- also on a crash, so a partial run still leaves an
        honest record.
    """
    try:
        for step in range(simulation_steps):
            ctx.env.step()
            ctx.recorder.record_step_summary(step)
            for intersection_id, handler in ctx.handlers.items():
                handler.step()
                if handler.switch_phase:
                    next_phase = decide(step, intersection_id, handler)
                    ctx.recorder.record_decision_wait()
                    handler.activate_phase(next_phase)
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected at step {step}, stopping early.")
                break
    finally:
        ctx.env.close()
        summary = ctx.recorder.save_final_summary()
    return summary
