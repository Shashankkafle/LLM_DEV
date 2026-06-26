from datetime import datetime
from pathlib import Path

from sumo_env import SumoEnv
from utils.prompt_builder import getPrompt
from models_inference.LLM.open_llm import LLM_Inference
from utils.phase_handler import PhaseHandler
from configurations import (
    INTERSECTION_CONFIGS,
    DEFAULT_INTERSECTION_CONFIG_NAME,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_START_PHASE,
    LLM_DEFAULT_PATH,
    SIGNAL_REGEX,
    LOGS_DIR_NAME,
    PHASE_SEQUENCES_DIR_NAME,
)
from utils.metrics_recorder import MetricsRecorder
import time
import re
import argparse

from utils.replay_recorder import ReplayRecorder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="rough_test")
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--llm_path", type=str, default=LLM_DEFAULT_PATH)
    parser.add_argument("--use_gui", type=bool, default=False)
    parser.add_argument(
        "--intersection_config",
        type=str,
        choices=list(INTERSECTION_CONFIGS.keys()),
        default=DEFAULT_INTERSECTION_CONFIG_NAME,
        help="Which intersection config (from configurations.INTERSECTION_CONFIGS) to use.",
    )

    return parser.parse_args()

def parse_llm_signal(raw_text):
    """
         extracts the requested signal using regex.
    """
    match = re.search(SIGNAL_REGEX, raw_text, re.IGNORECASE)
    if match:
            return match.group(1).strip().upper()
    return None


def state_is_empty(state_dict):
    """True when no vehicles are present on any controlled lane of the intersection.

    Uses the per-lane counts (``lane_states``) so each lane is counted once. An
    empty intersection has no decision to make, so keeping the current phase is
    the correct, deliberate action -- not a parse failure. We detect this from
    the state we already have, rather than asking the LLM to recognise "zero".
    """
    lane_states = state_dict.get("lane_states", {})
    for lane in lane_states.values():
        if lane.get("early_queued", 0) > 0:
            return False
        if any(count > 0 for count in lane.get("segments", {}).values()):
            return False
    return True

def mock_llm_inference(prompt, conf):
    """
    Mock function to simulate LLM inference.
    """
    phases = list(conf.get("phases").keys())
    return f"<signal>{phases[len(prompt) % len(phases)]}</signal>"
        

def main(args):
    print("Starting main function...")
    # Intersection config selected for this run (phases, signal strings, timings).
    conf = INTERSECTION_CONFIGS[args.intersection_config]

    # Initialize SUMO environment
    # Initialize LLM
    llm = LLM_Inference(llm_path=args.llm_path)
    llm.initialize_llm()

    base_log_dir = LOGS_DIR_NAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = Path(base_log_dir) / f"{args.test_name}_{timestamp}"
    phase_sequence_dir = records_dir / PHASE_SEQUENCES_DIR_NAME
    phase_sequence_dir.mkdir(parents=True, exist_ok=True)
    original_run_details = {
        "test_name": args.test_name,
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "llm_path": args.llm_path,
        "intersection_config": args.intersection_config,
    }
    run_replay_recorder = ReplayRecorder(record_dir=records_dir, meta=original_run_details)
    recorder = MetricsRecorder(
        run_dir=records_dir, verbose=True, phase_names=list(conf["phases"].keys())
    )
    env = SumoEnv(
        sumo_config=args.simulation_config, use_gui=args.use_gui,
        phase_sequence_dir=phase_sequence_dir, intersection_config=conf,
        output_dir=records_dir,
    )
    env.start_simulation()


    intersection_phase_handlers = {}
    intersections = env.get_intersections()
    for intersection_id in intersections:
        intersection_phase_handlers[intersection_id] = PhaseHandler( env=env, conf=conf, intersection_id=intersection_id, start_phase=DEFAULT_START_PHASE,replay_recorder=run_replay_recorder)
    for step in range(args.simulation_steps):
        print(f"Simulation step: {step}")
        env.step()
        recorder.record_step_summary(step)
        for intersection_id, handler in intersection_phase_handlers.items():
            handler.step()
            if handler.switch_phase:

                state_data = env.get_state(intersection_id)
                previous_phase = handler.current_phase

                # Classify every decision point into exactly one of three outcomes
                # so that "nothing to do" is never confused with "model failed".
                if state_is_empty(state_data):
                    # No vehicles waiting or approaching: holding the current phase
                    # is correct. Skip the (expensive) LLM call entirely.
                    decision_type = "no_action_empty"
                    prompt = None
                    llm_output = None
                    latency_ms = 0.0
                    extracted_signal = None
                    next_phase = previous_phase
                else:
                    prompt = getPrompt(state_dict=state_data)
                    start_time = time.time()
                    llm_output = llm.inference(prompt)
                    # llm_output = mock_llm_inference(prompt, conf)
                    latency_ms = (time.time() - start_time) * 1000

                    # Raw parse result: None if no <signal> tag, otherwise the
                    # tag's contents (which may still be an invalid phase name).
                    extracted_signal = parse_llm_signal(llm_output)
                    if extracted_signal in conf["phases"]:
                        decision_type = "llm_decision"
                        next_phase = extracted_signal
                    else:
                        # Non-empty intersection but the model gave no usable answer
                        # (truncated / no tag) or named an invalid phase. Hold the
                        # current phase, but record this as a genuine failure.
                        decision_type = "fallback_parse_error"
                        next_phase = previous_phase
                        print(f"[Warning] No valid signal ('{extracted_signal}') at step {step}, "
                              f"intersection {intersection_id}. Holding {previous_phase}.")

                handler.activate_phase(next_phase)

                recorder.record_decision(
                    step=step, state_dict=state_data, prompt=prompt,
                    llm_output=llm_output, previous_phase=previous_phase,
                    final_phase=next_phase, decision_type=decision_type,
                    latency_ms=latency_ms,
                    extracted_signal=extracted_signal,
                    intersection_id=intersection_id
                )

    # Close SUMO first so it flushes the statistics file, then summarize (the
    # summary parses that file for population-faithful metrics).
    env.close()
    recorder.save_final_summary()
    # Replay events + metadata were streamed to disk during the run; nothing
    # left to flush here.


        


if __name__ == "__main__":
    args = parse_args()
    main(args)    