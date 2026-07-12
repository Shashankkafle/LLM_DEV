"""LLM traffic-signal controller on the shared runner seam (runner_common).

Each time an intersection's green window ends, the current state is rendered
into a prompt, the LLM picks the next phase, and the decision (with its raw
output and parse outcome) is logged through MetricsRecorder.
"""

import argparse
import re
import time

from models_inference.LLM.open_llm import LLM_Inference
from runner_common import setup_run, run_control_loop, build_blockage_manager
from utils.prompt_builder import get_prompt
from configurations import (
    INTERSECTION_CONFIGS,
    DEFAULT_INTERSECTION_CONFIG_NAME,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_SIMULATION_CONFIG,
    LLM_DEFAULT_PATH,
    SIGNAL_REGEX,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="rough_test")
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--llm_path", type=str, default=LLM_DEFAULT_PATH)
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="SUMO random seed. Default keeps SUMO's fixed "
                             "default (deterministic reruns).")
    parser.add_argument("--blockage_scenario", type=str, default=None,
                        help="Path to a blockage scenario JSON (see "
                             "simulations/single_intersection/scenarios/). "
                             "Omit for no blockages.")
    parser.add_argument("--hide_blockage_info", action="store_true",
                        help="Ablation: inject the blockages physically but "
                             "keep them out of the prompt. Separates 'the LLM "
                             "uses the incident information' from 'the LLM "
                             "reacts to the queue numbers'.")
    parser.add_argument(
        "--intersection_config",
        type=str,
        choices=list(INTERSECTION_CONFIGS.keys()),
        default=DEFAULT_INTERSECTION_CONFIG_NAME,
        help="Which intersection config (from configurations.INTERSECTION_CONFIGS) to use.",
    )
    return parser.parse_args()


def parse_llm_signal(raw_text):
    """Extracts the requested signal from the LLM output using SIGNAL_REGEX."""
    match = re.search(SIGNAL_REGEX, raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    return None


# Tokens a model may return to mean "no phase change is needed" rather than
# naming one of the four phases. LightGPT-style models emit e.g.
# <signal>None</signal> when the intersection they see is empty. This is a
# deliberate "hold the current phase" decision, NOT a hallucination.
NO_ACTION_SIGNALS = {"NONE", "NULL", "NO_ACTION", "NO_CHANGE", "NOCHANGE", "KEEP", "HOLD", "STAY"}


def is_no_action_signal(extracted_signal):
    """True if the model explicitly declared that no phase change is needed."""
    if extracted_signal is None:
        return False
    token = extracted_signal.strip().upper().replace(" ", "_").replace("-", "_")
    return token in NO_ACTION_SIGNALS


def state_is_empty(state_dict):
    """True when no vehicles are present on the phase-controlled lanes.

    Bases emptiness on ``movement_states`` -- the exact lanes that feed the LLM
    prompt -- so "empty" means the model would see an all-zero state. (The
    always-green right-turn lanes are deliberately excluded: they are not part of
    any phase and never appear in the prompt, so cars there are irrelevant to the
    phase decision.) An empty intersection has no decision to make, so holding the
    current phase is the correct, deliberate action -- not a parse failure -- and
    we detect it from the state we already have rather than spending an LLM call.
    """
    movement_states = state_dict.get("movement_states", {})
    for phase_directions in movement_states.values():
        for direction in phase_directions.values():
            if not isinstance(direction, dict):
                continue
            if direction.get("early_queued", 0) > 0:
                return False
            if any(count > 0 for count in direction.get("segments", {}).values()):
                return False
    return True


def main(args):
    conf = INTERSECTION_CONFIGS[args.intersection_config]

    llm = LLM_Inference(llm_path=args.llm_path)
    llm.initialize_llm()

    _, blockage_manager = build_blockage_manager(args.blockage_scenario)
    run_meta = {
        "test_name": args.test_name,
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "llm_path": args.llm_path,
        "intersection_config": args.intersection_config,
        "seed": args.seed,
        "blockage_scenario": args.blockage_scenario,
        "hide_blockage_info": args.hide_blockage_info,
    }
    ctx = setup_run(conf, args.test_name, args.simulation_config, run_meta,
                    use_gui=args.use_gui, seed=args.seed, verbose_metrics=True,
                    blockage_manager=blockage_manager)

    def decide(step, intersection_id, handler):
        state_data = ctx.env.get_state(intersection_id)
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
            blockages = (None if args.hide_blockage_info
                         else ctx.env.describe_blockages(intersection_id))
            prompt = get_prompt(state_dict=state_data, blockages=blockages)
            start_time = time.time()
            llm_output = llm.inference(prompt)
            latency_ms = (time.time() - start_time) * 1000

            # Raw parse result: None if no <signal> tag, otherwise the
            # tag's contents (which may still be an invalid phase name).
            extracted_signal = parse_llm_signal(llm_output)
            if extracted_signal in conf["phases"]:
                decision_type = "llm_decision"
                next_phase = extracted_signal
            elif is_no_action_signal(extracted_signal):
                # Model explicitly judged no phase change is needed
                # (e.g. <signal>None</signal>). Hold the current phase;
                # this is a valid decision, not a failure.
                decision_type = "llm_no_action"
                next_phase = previous_phase
            else:
                # Non-empty intersection but the model gave no usable answer
                # (truncated / no tag) or named an invalid phase. Hold the
                # current phase, but record this as a genuine failure.
                decision_type = "fallback_parse_error"
                next_phase = previous_phase
                print(f"[Warning] No valid signal ('{extracted_signal}') at step {step}, "
                      f"intersection {intersection_id}. Holding {previous_phase}.")

        ctx.recorder.record_decision(
            step=step, state_dict=state_data, prompt=prompt,
            llm_output=llm_output, previous_phase=previous_phase,
            final_phase=next_phase, decision_type=decision_type,
            latency_ms=latency_ms,
            extracted_signal=extracted_signal,
            intersection_id=intersection_id,
        )
        return next_phase

    run_control_loop(ctx, args.simulation_steps, decide)


if __name__ == "__main__":
    main(parse_args())
