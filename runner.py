"""LLM traffic-signal controller on the shared runner seam (runner_common).

Each time an intersection's green window ends, the current state is rendered
into a prompt, the LLM picks the next phase, and the decision (with its raw
output and parse outcome) is logged through MetricsRecorder.
"""

import argparse
import re
import time

from models_inference.LLM.open_llm import LLM_Inference
from models_inference.LLM.openrouter_llm import OpenRouter_Inference, OPENROUTER_PREFIX
from runner_common import (
    setup_run, run_control_loop, run_control_loop_batched, build_blockage_manager,
)
from utils.prompt_builder import get_prompt
from utils.run_manifest import build_manifest, save_manifest
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
    parser.add_argument("--run_group", type=str, default=None,
                        help="Nest the run under logs/<run_group>/ (used by "
                             "run_matrix.py to group a sweep's runs).")
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--llm_path", type=str, default=LLM_DEFAULT_PATH,
                        help="Local model directory, or "
                             "'openrouter:<provider>/<model>' to run the "
                             "decisions against an OpenRouter-hosted model "
                             "(needs OPENROUTER_API_KEY).")
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
                             "keep them out of the prompt entirely (both the "
                             "approach and exit sections). Separates 'the LLM "
                             "uses the incident information' from 'the LLM "
                             "reacts to the queue numbers'.")
    parser.add_argument("--blockage_info_scope", type=str,
                        choices=("both", "approach", "exit"),
                        default="both",
                        help="Which controllers hear about a blockage: "
                             "'approach' informs only the intersection whose "
                             "approach lane is blocked (pre-existing behavior), "
                             "'exit' only the intersection whose exit road is "
                             "blocked, 'both' informs both. Ignored under "
                             "--hide_blockage_info.")
    parser.add_argument(
        "--intersection_config",
        type=str,
        choices=list(INTERSECTION_CONFIGS.keys()),
        default=DEFAULT_INTERSECTION_CONFIG_NAME,
        help="Which intersection config (from configurations.INTERSECTION_CONFIGS) to use.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Decide one intersection at a time (the pre-batching path). Default "
             "batches every intersection switching on the same step into one "
             "inference call; --sequential forces the old behavior.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Generation cap per decision (default: configurations."
             "LLM_MAX_NEW_TOKENS). A chain-of-thought model spends its "
             "reasoning tokens out of this same budget, so it needs a far "
             "larger value than the fine-tuned arms -- too low and the model "
             "returns an empty completion, which the runner can only hold on.")
    parser.add_argument(
        "--request_timeout", type=int, default=None,
        help="Per-request timeout in seconds for the OpenRouter backend "
             "(default: configurations.LLM_REQUEST_TIMEOUT_S). Raise it for a "
             "slow thinking model: a timeout is retried, so a low value pays "
             "for several full generations and still fails.")
    parser.add_argument(
        "--max_batch_size", type=int, default=0,
        help="Cap the per-step inference batch (0 = no cap, one batch per step). "
             "Lower it if a wide batch pressures GPU memory; results are "
             "unaffected (each sequence is decoded independently).")
    return parser.parse_args()


def build_llm(llm_path, max_new_tokens=None, request_timeout=None):
    """Local HuggingFace model by default; an 'openrouter:<model>' path routes
    the decisions to the hosted API instead. Both satisfy the same interface,
    so nothing downstream of here knows which backend it is talking to.

    request_timeout only reaches the hosted backend -- local generation has no
    socket to time out."""
    if llm_path.startswith(OPENROUTER_PREFIX):
        return OpenRouter_Inference(llm_path, max_new_tokens=max_new_tokens,
                                    timeout_s=request_timeout)
    return LLM_Inference(llm_path=llm_path, max_new_tokens=max_new_tokens)


def _chunk(items, size):
    """Split items into consecutive chunks of at most `size` (size <= 0 -> one
    chunk holding everything)."""
    if size and size > 0:
        return [items[i:i + size] for i in range(0, len(items), size)]
    return [items]


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

    llm = build_llm(args.llm_path, max_new_tokens=args.max_new_tokens,
                    request_timeout=args.request_timeout)
    llm.initialize_llm()

    _, blockage_manager = build_blockage_manager(args.blockage_scenario)
    run_meta = {
        "test_name": args.test_name,
        "controller": "llm",
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "llm_path": args.llm_path,
        "intersection_config": args.intersection_config,
        "seed": args.seed,
        "blockage_scenario": args.blockage_scenario,
        "hide_blockage_info": args.hide_blockage_info,
        "blockage_info_scope": args.blockage_info_scope,
        # Recorded so final_summary consumers can tell batched runs apart:
        # under batching, a decision's inference_latency_ms is the shared batch
        # wall-time, not a per-call measurement.
        "batched": not getattr(args, "sequential", False),
        "max_batch_size": getattr(args, "max_batch_size", 0),
    }
    manifest = build_manifest("llm", args, args.intersection_config, conf)
    # getattr guards: smoke tests swap in stub LLMs without describe()/last_usage.
    describe = getattr(llm, "describe", None)
    manifest["llm"] = describe() if describe else None
    ctx = setup_run(conf, args.test_name, args.simulation_config, run_meta,
                    use_gui=args.use_gui, seed=args.seed, verbose_metrics=True,
                    blockage_manager=blockage_manager, manifest=manifest,
                    run_group=args.run_group)

    def capture_example_formatted_prompt():
        """Store the first fully-templated prompt (system prompt + chat
        template) in the manifest: together with the per-decision user prompts
        it makes every prompt the model ever saw reconstructable."""
        if manifest["llm"] is None or "example_formatted_prompt" in manifest["llm"]:
            return
        formatted = getattr(llm, "last_formatted_prompt", None)
        if formatted is None:
            return
        manifest["llm"]["example_formatted_prompt"] = formatted
        save_manifest(ctx.records_dir, manifest)

    def prepare_decision(intersection_id, handler):
        """Everything up to (but excluding) the LLM call: read the state, decide
        whether the intersection is empty, and build its prompt. Shared by the
        sequential and batched drivers so the two cannot diverge.

        The structured blockage facts are always computed, even when the prompt
        hides them, so the record shows what the model was NOT told.
        """
        state_data = ctx.env.get_state(intersection_id)
        blockage_facts = ctx.env.describe_blockages(intersection_id)
        exit_blockage_facts = ctx.env.describe_exit_blockages(intersection_id)
        req = {
            "intersection_id": intersection_id,
            "state_data": state_data,
            "previous_phase": handler.current_phase,
            "blockage_facts": blockage_facts,
            "exit_blockage_facts": exit_blockage_facts,
            "empty": state_is_empty(state_data),
            "prompt": None,
            "blockage_info_in_prompt": None,
        }
        if not req["empty"]:
            if args.hide_blockage_info:
                blockages = exit_blockages = None
            else:
                scope = args.blockage_info_scope
                blockages = (blockage_facts
                             if scope in ("both", "approach") else None)
                exit_blockages = (exit_blockage_facts
                                  if scope in ("both", "exit") else None)
            req["blockage_info_in_prompt"] = bool(blockages) or bool(exit_blockages)
            req["prompt"] = get_prompt(state_dict=state_data, blockages=blockages,
                                       exit_blockages=exit_blockages)
        return req

    def finalize_decision(step, req, llm_output, latency_ms, token_usage, error,
                          reasoning=None):
        """Classify the outcome, record the decision, return the next phase.
        Shared by both drivers. Every decision point maps to exactly one
        decision_type so "nothing to do" is never confused with "model failed".
        """
        previous_phase = req["previous_phase"]
        intersection_id = req["intersection_id"]
        if req["empty"]:
            # No vehicles waiting or approaching: holding is correct; no LLM
            # call was made.
            decision_type = "no_action_empty"
            extracted_signal = None
            next_phase = previous_phase
        elif error is not None:
            # Inference itself failed (CUDA error, OOM, ...). Hold the current
            # phase and keep the run going; the error is recorded per decision
            # and counted in the final summary.
            decision_type = "inference_error"
            extracted_signal = None
            next_phase = previous_phase
            print(f"[Warning] LLM inference failed at step {step}, "
                  f"intersection {intersection_id}: {error}. "
                  f"Holding {previous_phase}.")
        else:
            # Raw parse result: None if no <signal> tag, otherwise the tag's
            # contents (which may still be an invalid phase name).
            extracted_signal = parse_llm_signal(llm_output)
            if extracted_signal in conf["phases"]:
                decision_type = "llm_decision"
                next_phase = extracted_signal
            elif is_no_action_signal(extracted_signal):
                # Model explicitly judged no phase change is needed
                # (e.g. <signal>None</signal>). A valid "hold", not a failure.
                decision_type = "llm_no_action"
                next_phase = previous_phase
            else:
                # Non-empty intersection but no usable answer (truncated / no
                # tag) or an invalid phase name. Hold, but record a real failure.
                decision_type = "fallback_parse_error"
                next_phase = previous_phase
                print(f"[Warning] No valid signal ('{extracted_signal}') at step {step}, "
                      f"intersection {intersection_id}. Holding {previous_phase}.")

        ctx.recorder.record_decision(
            step=step, state_dict=req["state_data"], prompt=req["prompt"],
            llm_output=llm_output, previous_phase=previous_phase,
            final_phase=next_phase, decision_type=decision_type,
            latency_ms=latency_ms, extracted_signal=extracted_signal,
            intersection_id=intersection_id,
            blockage_facts=req["blockage_facts"],
            exit_blockage_facts=req["exit_blockage_facts"],
            blockage_info_in_prompt=req["blockage_info_in_prompt"],
            token_usage=token_usage, error=error, reasoning=reasoning,
        )
        return next_phase

    def decide(step, intersection_id, handler):
        """Sequential driver: one intersection, one LLM call."""
        req = prepare_decision(intersection_id, handler)
        if req["empty"]:
            return finalize_decision(step, req, None, 0.0, None, None)
        error = None
        token_usage = None
        reasoning = None
        start_time = time.time()
        try:
            llm_output = llm.inference(req["prompt"])
        except Exception as exc:
            llm_output = None
            error = repr(exc)
        latency_ms = (time.time() - start_time) * 1000
        if error is None:
            token_usage = getattr(llm, "last_usage", None)
            reasoning = getattr(llm, "last_reasoning", None)
            capture_example_formatted_prompt()
        return finalize_decision(step, req, llm_output, latency_ms,
                                 token_usage, error, reasoning)

    def infer_single(prompt):
        """Run one prompt through the model, timed, isolating its own failure --
        the sequential path's inference, reused as the per-prompt fallback."""
        start_time = time.time()
        try:
            output = llm.inference(prompt)
            usage = getattr(llm, "last_usage", None)
            reasoning = getattr(llm, "last_reasoning", None)
            error = None
            capture_example_formatted_prompt()
        except Exception as exc:
            output, usage, reasoning, error = None, None, None, repr(exc)
        latency_ms = (time.time() - start_time) * 1000
        return {"output": output, "usage": usage, "reasoning": reasoning,
                "latency_ms": latency_ms, "error": error}

    def infer_chunk(prompts):
        """Run one chunk as a single batched call. On ANY batch-level failure
        (OOM, or a backend returning the wrong number of outputs/usages) fall
        back to running the chunk one prompt at a time, so only genuinely
        failing intersections error out -- matching the sequential path's
        per-intersection isolation instead of failing the whole cohort.
        Returns one result dict per prompt, in order.
        """
        start_time = time.time()
        try:
            outputs = llm.inference_batch(prompts)
            usages = getattr(llm, "last_usage_batch", None) or [None] * len(prompts)
            # A backend without reasoning support simply has none to report; a
            # short list would silently misalign, so only a full one is used.
            reasonings = getattr(llm, "last_reasoning_batch", None)
            if not reasonings or len(reasonings) != len(prompts):
                reasonings = [None] * len(prompts)
            if len(outputs) != len(prompts) or len(usages) != len(prompts):
                raise ValueError(
                    f"inference_batch returned {len(outputs)} outputs / "
                    f"{len(usages)} usages for {len(prompts)} prompts")
            latency_ms = (time.time() - start_time) * 1000
            capture_example_formatted_prompt()
            return [{"output": o, "usage": u, "reasoning": r,
                     "latency_ms": latency_ms, "error": None}
                    for o, u, r in zip(outputs, usages, reasonings)]
        except Exception as exc:
            print(f"[Warning] Batched inference failed ({exc!r}); falling back "
                  f"to per-prompt for {len(prompts)} intersection(s).")
            return [infer_single(p) for p in prompts]

    def decide_batch(step, pending):
        """Batched driver: every intersection switching this step. The non-empty
        prompts run as batched inference (chunked by --max_batch_size); empty
        intersections never reach the model. Returns {intersection_id: phase}.

        Each queried decision records its chunk's shared wall-clock as
        inference_latency_ms (the calls ran together), flagged via run_meta.
        """
        reqs = [prepare_decision(iid, handler) for iid, handler in pending]
        query_reqs = [r for r in reqs if not r["empty"]]

        results = []
        for chunk in _chunk(query_reqs, getattr(args, "max_batch_size", 0)):
            if chunk:
                results.extend(infer_chunk([r["prompt"] for r in chunk]))

        next_phases = {}
        query_idx = 0
        for req in reqs:
            if req["empty"]:
                next_phase = finalize_decision(step, req, None, 0.0, None, None)
            else:
                res = results[query_idx]
                query_idx += 1
                next_phase = finalize_decision(
                    step, req, res["output"], res["latency_ms"], res["usage"],
                    res["error"], res.get("reasoning"))
            next_phases[req["intersection_id"]] = next_phase
        return next_phases

    if getattr(args, "sequential", False):
        run_control_loop(ctx, args.simulation_steps, decide)
    else:
        run_control_loop_batched(ctx, args.simulation_steps, decide_batch)


if __name__ == "__main__":
    main(parse_args())
