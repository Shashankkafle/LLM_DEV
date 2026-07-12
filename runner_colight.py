"""SUMO training + eval loop for the lifted CoLight agent.

Replaces the source CityFlow generator/pipeline/construct_sample/updater flow with a
SUMO-native loop that reuses this repo's SumoEnv, PhaseHandler, MetricsRecorder, and
ReplayRecorder. Mirrors runner.py's structure; does not modify runner.py.

Per round (episode): reset sim -> drive SumoEnv + PhaseHandler -> build state via
state_features -> choose_action -> accumulate transitions in the exact 7-tuple shape
prepare_Xs_Y expects -> prepare_Xs_Y -> train_network -> periodic target-net copy ->
save_network. EVAL: load weights, run one greedy episode through MetricsRecorder.

See PORTING_PLAN.md for the full contract.
"""

import os
# Must precede any tensorflow import so `tensorflow.keras` resolves to Keras 2.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import traci

from sumo_env import SumoEnv
from runner_common import create_run_dirs, build_blockage_manager
from utils.tf_device import configure_tf_devices
from utils.phase_handler import PhaseHandler
from utils.metrics_recorder import MetricsRecorder
from utils.replay_recorder import ReplayRecorder
from utils import state_features as sf
from configurations import (
    INTERSECTION_CONFIGS,
    DEFAULT_SIMULATION_STEPS,
    DEFAULT_START_PHASE,
    PHASE_SEQUENCES_DIR_NAME,
    BLOCKAGE_SCENARIO_COPY_FILENAME,
    COLIGHT_AGENT_CONF,
    COLIGHT_FEATURES,
    ADVANCED_COLIGHT_AGENT_CONF,
    ADVANCED_COLIGHT_FEATURES,
    COLIGHT_NUM_ROUNDS,
    COLIGHT_TOP_K_ADJACENCY,
    COLIGHT_NUM_LANE_FEATURES,
    COLIGHT_REWARD_QUEUE_COEFF,
    COLIGHT_DEFAULT_SIMULATION_CONFIG,
    COLIGHT_DEFAULT_INTERSECTION_CONFIG_NAME,
    COLIGHT_WEIGHTS_DIR_NAME,
    COLIGHT_TRAINING_PROGRESS_FILENAME,
    COLIGHT_MODEL_METADATA_FILENAME,
)
# Importing the agent activates legacy Keras + the unsafe-deserialization toggle.
from models_inference.RL.colight_agent import CoLightAgent


# Each variant = (feature list, agent conf). Both use the SAME lifted CoLightAgent;
# only the features differ. adjacency_matrix MUST stay last (the agent drops it with
# LIST_STATE_FEATURE[:-1] when building the feature vector).
VARIANTS = {
    "colight": (COLIGHT_FEATURES, COLIGHT_AGENT_CONF),
    "advanced_colight": (ADVANCED_COLIGHT_FEATURES, ADVANCED_COLIGHT_AGENT_CONF),
}


class _NullReplayRecorder:
    """Satisfies PhaseHandler's replay interface without writing files (training)."""

    def record_phase_change(self, *args, **kwargs):
        pass


def build_agent_confs(conf, num_agents, weights_dir, features, agent_conf):
    """Assemble the three dicts the lifted CoLightAgent expects from our config."""
    phase_map = sf.build_phase_onehot(conf)
    dic_traffic_env_conf = {
        "NUM_INTERSECTIONS": num_agents,
        "TOP_K_ADJACENCY": COLIGHT_TOP_K_ADJACENCY,
        "PHASE": phase_map,
        "BINARY_PHASE_EXPANSION": True,
        "NUM_LANE_FEATURES": COLIGHT_NUM_LANE_FEATURES,
        "LIST_STATE_FEATURE": features,
    }
    dic_agent_conf = dict(agent_conf)
    dic_path = {"PATH_TO_MODEL": str(weights_dir)}
    return dic_traffic_env_conf, dic_agent_conf, dic_path, phase_map


def _git_commit():
    """Current repo commit hash for reproducibility, or None if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _sumocfg_inputs(simulation_config):
    """Pull the net-file and route-files a .sumocfg points at.

    These are the road *network* and the *traffic flow* (vehicle demand) the model
    is trained against. Returns {} if the config can't be read or parsed.
    """
    try:
        root = ET.parse(simulation_config).getroot()
        inputs = {}
        net = root.find("./input/net-file")
        routes = root.find("./input/route-files")
        if net is not None:
            inputs["net_file"] = net.get("value")
        if routes is not None:
            inputs["route_files"] = routes.get("value")
        return inputs
    except Exception:
        return {}


def build_training_metadata(args, order, agent_conf):
    """Assemble a reproducibility record describing how these weights were trained.

    Captures the seeds, the road network + traffic flow, and the training/agent
    hyperparameters so any saved model can be traced back to the setup that
    produced it. Written once per run as a sidecar next to the .h5 weights.
    """
    effective_epochs = args.epochs if args.epochs is not None else agent_conf["EPOCHS"]
    return {
        "controller": args.variant,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "seeded_rngs": ["random", "numpy", "tensorflow"],
        "network": {
            "intersection_config": args.intersection_config,
            "num_intersections": len(order),
            "intersection_ids": list(order),
            "simulation_config": args.simulation_config,
            **_sumocfg_inputs(args.simulation_config),
        },
        "training": {
            "mode": args.mode,
            "num_rounds": args.num_rounds,
            "simulation_steps": args.simulation_steps,
            "epochs": effective_epochs,
            "ablate_attention": args.ablate_attention,
            "train_blockage_scenario": args.train_blockage_scenario,
            "reward_queue_coeff": COLIGHT_REWARD_QUEUE_COEFF,
            "top_k_adjacency": COLIGHT_TOP_K_ADJACENCY,
            "num_lane_features": COLIGHT_NUM_LANE_FEATURES,
        },
        "agent_conf": dict(agent_conf),
        "weights": {
            "filename_pattern": "round_{round}_inter_0.h5",
            # Grows as each round's .h5 is actually written (see train()), so this
            # always reflects the checkpoints on disk -- even if the run crashes.
            # training.num_rounds above records the planned total.
            "rounds_saved": [],
        },
    }


def load_training_metadata(weights_dir):
    """Read a weights dir's training_metadata.json, or None if absent/unreadable."""
    path = Path(weights_dir) / COLIGHT_MODEL_METADATA_FILENAME
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_episode(env, conf, agent, phase_map, order, features, replay, recorder,
                simulation_steps, reward_coeff, collect, ablate_attention=False):
    """Drive one episode with round-synchronized decisions.

    Returns (transitions_per_intersection, total_reward). When ``collect`` is False
    (eval) no transitions are stored and actions are greedy (agent epsilon already 0).
    ``ablate_attention`` forces every intersection's adjacency to self-only (neighbor
    attention contributes nothing) -- the 4x4 attention ablation check.
    """
    id_to_name = {cfg["id"]: name for name, cfg in conf["phases"].items()}

    # Ablation: seed the adjacency cache with self-only rows before any state is built,
    # so the lazy real-adjacency builder is bypassed.
    if ablate_attention:
        env._colight_adjacency = sf.self_only_adjacency_rows(order, COLIGHT_TOP_K_ADJACENCY)

    # Set the starting green explicitly so the RYG string is a known phase from t=0
    # (cur_phase recovers the logical index by parsing the green string).
    for inter_id in order:
        start_green = conf["phases"][agent_start_phase(conf)]["green"]
        env.set_phase(inter_id, start_green)
        replay.record_phase_change(env.get_current_step(), inter_id, start_green,
                                   f"{agent_start_phase(conf)}_GREEN")

    handlers = {
        inter_id: PhaseHandler(env=env, conf=conf, intersection_id=inter_id,
                               start_phase=agent_start_phase(conf), replay_recorder=replay)
        for inter_id in order
    }

    transitions = {inter_id: [] for inter_id in order}
    prev = None  # (live_states, actions, decision_step)
    window_queue = {inter_id: 0.0 for inter_id in order}
    window_steps = 0
    total_reward = 0.0

    for step in range(simulation_steps):
        env.step()
        if recorder is not None:
            recorder.record_step_summary(step)

        for inter_id in order:
            handlers[inter_id].step()
            window_queue[inter_id] += sf.intersection_queue_length(env, inter_id)
        window_steps += 1

        # Decision round: every intersection has finished its green window.
        if all(handlers[inter_id].switch_phase for inter_id in order):
            # One network-wide AWT sample per round; decisions are synchronized,
            # so this matches the per-intersection sampling of the other runners.
            if recorder is not None:
                recorder.record_decision_wait()

            live_states = [sf.build_state(env, inter_id, features) for inter_id in order]

            # Finalize the previous round's transitions (reward = avg queue over window).
            if prev is not None and collect:
                prev_states, prev_actions, prev_step = prev
                for k, inter_id in enumerate(order):
                    reward = reward_coeff * (window_queue[inter_id] / max(window_steps, 1))
                    total_reward += reward
                    transitions[inter_id].append((
                        sf.expand_state_for_memory(prev_states[k], phase_map),
                        int(prev_actions[k]),
                        sf.expand_state_for_memory(live_states[k], phase_map),
                        reward, reward, prev_step, "episode",
                    ))

            actions = agent.choose_action(step, live_states)
            for k, inter_id in enumerate(order):
                handlers[inter_id].activate_phase(id_to_name[int(actions[k])])

            prev = (live_states, actions, step)
            window_queue = {inter_id: 0.0 for inter_id in order}
            window_steps = 0

        # Eval-only early exit, matching the shared control loop (runner_common):
        # once SUMO expects no more vehicles, the remaining steps are all-zero
        # samples that would dilute the step-averaged metrics. Training keeps its
        # fixed horizon so per-round transition counts stay comparable across runs.
        if recorder is not None and traci.simulation.getMinExpectedNumber() <= 0:
            print(f"No more vehicles expected at step {step}, stopping early.")
            break

    # Finalize the LAST decision's transition. Its window ran to the episode end
    # (no following decision to trigger the normal finalize), so we close it here
    # using the terminal observation as next_state. The terminal RYG may be mid
    # yellow/all-red, so cur_phase is taken from the handler's authoritative logical
    # phase rather than parsed from the RYG. Mirrors source construct_sample, which
    # also keeps the last full-window decision.
    if collect and prev is not None and window_steps > 0:
        prev_states, prev_actions, prev_step = prev
        for k, inter_id in enumerate(order):
            # Build the terminal next_state from the variant's features. cur_phase comes
            # from the handler's logical phase (terminal RYG may be mid-transition); all
            # other features are read live from the env.
            terminal_next = {}
            for feat in features:
                if feat == "cur_phase":
                    terminal_next["cur_phase"] = [conf["phases"][handlers[inter_id].current_phase]["id"]]
                else:
                    terminal_next[feat] = sf.FEATURE_REGISTRY[feat](env, inter_id)
            reward = reward_coeff * (window_queue[inter_id] / window_steps)
            total_reward += reward
            transitions[inter_id].append((
                sf.expand_state_for_memory(prev_states[k], phase_map),
                int(prev_actions[k]),
                sf.expand_state_for_memory(terminal_next, phase_map),
                reward, reward, prev_step, "episode",
            ))

    return transitions, total_reward


def agent_start_phase(conf):
    """DEFAULT_START_PHASE if valid for this config, else the config's first phase."""
    return DEFAULT_START_PHASE if DEFAULT_START_PHASE in conf["phases"] else next(iter(conf["phases"]))


def train(args, conf, records_dir, features, agent_conf):
    weights_dir = records_dir / COLIGHT_WEIGHTS_DIR_NAME
    weights_dir.mkdir(parents=True, exist_ok=True)
    progress_path = records_dir / COLIGHT_TRAINING_PROGRESS_FILENAME

    agent = None
    phase_map = None
    memory = None  # {intersection_id: [transition, ...]} accumulated across rounds
    order = None
    metadata = None       # reproducibility sidecar, written/updated as rounds save
    metadata_path = None
    max_memory = agent_conf["MAX_MEMORY_LEN"]
    sample_size = agent_conf["SAMPLE_SIZE"]
    epsilon_init = agent_conf["EPSILON"]
    epsilon_decay = agent_conf["EPSILON_DECAY"]
    min_epsilon = agent_conf["MIN_EPSILON"]
    update_q_bar_freq = agent_conf["UPDATE_Q_BAR_FREQ"]

    _, blockage_manager = build_blockage_manager(args.train_blockage_scenario)

    for r in range(args.num_rounds):
        print(f"\n========== {args.variant} training round {r}/{args.num_rounds - 1} ==========")
        if blockage_manager is not None:
            # Each round is a fresh SUMO session; a manager carrying last
            # round's finished set would silently never re-fire the blockages.
            blockage_manager.reset()
        env = SumoEnv(sumo_config=args.simulation_config, use_gui=args.use_gui,
                      intersection_config=conf, blockage_manager=blockage_manager)
        env.start_simulation()
        order = sorted(env.get_intersections())

        if agent is None:
            dtec, dac, dpath, phase_map = build_agent_confs(conf, len(order), weights_dir,
                                                            features, agent_conf)
            if args.epochs is not None:
                dac["EPOCHS"] = args.epochs
            agent = CoLightAgent(dic_agent_conf=dac, dic_traffic_env_conf=dtec,
                                 dic_path=dpath, cnt_round=0, intersection_id="0")
            memory = {inter_id: [] for inter_id in order}

            # Record how this model is being trained (seeds, network, traffic
            # flow, hyperparameters) up front, so even a run that crashes before
            # round 0 finishes leaves a sidecar of its setup. rounds_saved starts
            # empty and is filled in below as each .h5 is actually written. dac is
            # still the initial config here -- before the per-round epsilon decay.
            metadata = build_training_metadata(args, order, dac)
            metadata_path = weights_dir / COLIGHT_MODEL_METADATA_FILENAME
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"[colight] wrote training metadata -> {metadata_path}")

        # Epsilon decay per round (the source applies this in the agent constructor).
        agent.dic_agent_conf["EPSILON"] = max(epsilon_init * (epsilon_decay ** r), min_epsilon)

        transitions, total_reward = run_episode(
            env, conf, agent, phase_map, order, features,
            replay=_NullReplayRecorder(), recorder=None,
            simulation_steps=args.simulation_steps,
            reward_coeff=COLIGHT_REWARD_QUEUE_COEFF, collect=True,
            ablate_attention=args.ablate_attention)
        env.close()

        # Accumulate + forget (keep last MAX_MEMORY_LEN per intersection).
        for inter_id in order:
            memory[inter_id].extend(transitions[inter_id])
            memory[inter_id] = memory[inter_id][-max_memory:]

        # Sample ONE aligned index set shared across intersections (keeps the
        # multi-agent samples time-aligned, as CoLight's stacking requires).
        buffer_len = len(memory[order[0]])
        num_transitions = sum(len(memory[i]) for i in order)
        if buffer_len > 0:
            n = min(sample_size, buffer_len)
            idx = random.sample(range(buffer_len), n)
            samples_list = [[memory[inter_id][k] for k in idx] for inter_id in order]
            agent.prepare_Xs_Y(samples_list)
            agent.train_network()

        # Target network update.
        if r % update_q_bar_freq == 0:
            agent.q_network_bar = agent.build_network_from_copy(agent.q_network)

        agent.save_network(f"round_{r}_inter_0")

        # Mark this round's checkpoint as saved and rewrite the sidecar, so
        # rounds_saved always matches the .h5 files on disk (the file is tiny;
        # the rewrite cost is negligible next to a full episode + train step).
        metadata["weights"]["rounds_saved"].append(r)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        epsilon = agent.dic_agent_conf["EPSILON"]
        print(f"[round {r}] total_reward={total_reward:.2f} "
              f"transitions_this_round={len(transitions[order[0]])} epsilon={epsilon:.3f}")
        with open(progress_path, "a") as f:
            f.write(json.dumps({
                "round": r,
                "total_reward": round(total_reward, 4),
                "transitions_this_round": len(transitions[order[0]]),
                "buffer_len": buffer_len,
                "epsilon": round(epsilon, 4),
            }) + "\n")

    return weights_dir


def evaluate(args, conf, records_dir, weights_dir, eval_round, features, agent_conf):
    phase_sequence_dir = records_dir / PHASE_SEQUENCES_DIR_NAME
    phase_sequence_dir.mkdir(parents=True, exist_ok=True)

    _, blockage_manager = build_blockage_manager(args.blockage_scenario)
    if args.blockage_scenario:
        shutil.copy(args.blockage_scenario,
                    records_dir / BLOCKAGE_SCENARIO_COPY_FILENAME)

    env = SumoEnv(sumo_config=args.simulation_config, use_gui=args.use_gui,
                  phase_sequence_dir=phase_sequence_dir, intersection_config=conf,
                  output_dir=records_dir, blockage_manager=blockage_manager)
    env.start_simulation()
    order = sorted(env.get_intersections())

    dtec, dac, dpath, phase_map = build_agent_confs(conf, len(order), weights_dir,
                                                    features, agent_conf)
    dac = dict(dac)
    dac["EPSILON"] = 0.0
    dac["MIN_EPSILON"] = 0.0
    agent = CoLightAgent(dic_agent_conf=dac, dic_traffic_env_conf=dtec,
                         dic_path=dpath, cnt_round=0, intersection_id="0")
    agent.load_network(f"round_{eval_round}_inter_0", file_path=str(weights_dir))
    agent.dic_agent_conf["EPSILON"] = 0.0  # greedy

    # Report (and sanity-check) what the loaded model was trained on, so an eval
    # on a different network/flow than training doesn't pass by silently.
    trained_on = load_training_metadata(weights_dir)
    if trained_on is not None:
        trained_net = trained_on.get("network", {})
        print(f"[eval] model trained on intersection_config="
              f"{trained_net.get('intersection_config')} "
              f"simulation_config={trained_net.get('simulation_config')} "
              f"seed={trained_on.get('seed')}")
        if trained_net.get("intersection_config") != args.intersection_config:
            print(f"[eval] WARNING: evaluating intersection_config={args.intersection_config} "
                  f"but model was trained on {trained_net.get('intersection_config')}")
        if trained_net.get("simulation_config") != args.simulation_config:
            print(f"[eval] WARNING: evaluating simulation_config={args.simulation_config} "
                  f"but model was trained on {trained_net.get('simulation_config')}")
    else:
        print(f"[eval] no training metadata found in {weights_dir} "
              f"(model predates metadata logging)")

    recorder = MetricsRecorder(run_dir=records_dir, verbose=False,
                               phase_names=list(conf["phases"].keys()),
                               sumo_config=args.simulation_config,
                               blockage_manager=blockage_manager)
    replay = ReplayRecorder(record_dir=records_dir, meta={
        "controller": args.variant,
        "mode": "eval",
        "eval_round": eval_round,
        "simulation_steps": args.simulation_steps,
        "simulation_config": args.simulation_config,
        "intersection_config": args.intersection_config,
        "blockage_scenario": args.blockage_scenario,
        "training_metadata": trained_on,
    })

    run_episode(env, conf, agent, phase_map, order, features,
                replay=replay, recorder=recorder,
                simulation_steps=args.simulation_steps,
                reward_coeff=COLIGHT_REWARD_QUEUE_COEFF, collect=False,
                ablate_attention=args.ablate_attention)

    # Close SUMO first so it flushes the statistics file, then summarize.
    env.close()
    summary = recorder.save_final_summary()
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="colight")
    parser.add_argument("--variant", type=str, default="colight",
                        choices=list(VARIANTS.keys()),
                        help="colight (lane_num_vehicle) or advanced_colight "
                             "(efficient pressure + near-stopline running).")
    parser.add_argument("--mode", type=str, default="train_eval",
                        choices=["train", "eval", "train_eval"])
    parser.add_argument("--num_rounds", type=int, default=COLIGHT_NUM_ROUNDS)
    parser.add_argument("--simulation_steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--simulation_config", type=str, default=COLIGHT_DEFAULT_SIMULATION_CONFIG)
    parser.add_argument("--intersection_config", type=str,
                        choices=list(INTERSECTION_CONFIGS.keys()),
                        default=COLIGHT_DEFAULT_INTERSECTION_CONFIG_NAME)
    parser.add_argument("--eval_round", type=int, default=-1,
                        help="Round weights to evaluate; -1 = last trained round.")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="For --mode eval: existing weights dir to load from.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training EPOCHS (default uses COLIGHT_AGENT_CONF).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blockage_scenario", type=str, default=None,
                        help="Blockage scenario JSON applied to the EVAL "
                             "episode only (zero-shot robustness arm -- "
                             "matches the LLM's zero-shot setting).")
    parser.add_argument("--train_blockage_scenario", type=str, default=None,
                        help="Blockage scenario JSON applied during TRAINING "
                             "rounds. This is a separate 'trained-on-incident' "
                             "arm: the policy sees the same scheduled incident "
                             "every round. Never pool it with eval-only runs.")
    parser.add_argument("--use_gui", action="store_true", default=False)
    parser.add_argument("--ablate_attention", action="store_true", default=False,
                        help="Force self-only adjacency (no neighbor attention) -- ablation.")
    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
    except Exception:
        pass

    # Use a GPU automatically if this machine has one; log CPU vs GPU either way.
    # Must run before train()/evaluate() build the agent's model.
    configure_tf_devices()

    conf = INTERSECTION_CONFIGS[args.intersection_config]
    features, agent_conf = VARIANTS[args.variant]
    records_dir, _ = create_run_dirs(args.test_name)

    weights_dir = None
    if args.mode in ("train", "train_eval"):
        weights_dir = train(args, conf, records_dir, features, agent_conf)

    if args.mode in ("eval", "train_eval"):
        if weights_dir is None:
            if args.weights_dir is None:
                raise ValueError("--mode eval requires --weights_dir")
            weights_dir = Path(args.weights_dir)
        eval_round = args.eval_round if args.eval_round >= 0 else args.num_rounds - 1
        summary = evaluate(args, conf, records_dir, weights_dir, eval_round, features, agent_conf)
        print(f"\n{args.variant} eval final summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
