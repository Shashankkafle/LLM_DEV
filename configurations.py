"""Single source of truth for constants and intersection configurations.

This module defines the project's tunable constants and the per-network
intersection configs. It imports nothing from the rest of the project, so any
module can import from it without risking a circular import.
"""

# =============================================================================
# Direction / movement reference maps
# =============================================================================

location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}

# These movement maps are identical across every intersection config, so they
# are defined once here and referenced from each config below.
MOVEMENT_DIRECTIONS = {
    "NT": "North", "NL": "North",
    "ST": "South", "SL": "South",
    "ET": "East",  "EL": "East",
    "WT": "West",  "WL": "West",
}

MOVEMENT_TYPES = {
    "NT": "through", "ST": "through", "ET": "through", "WT": "through",
    "NL": "left-turn", "SL": "left-turn", "EL": "left-turn", "WL": "left-turn",
}

# Maps each movement to the road that vehicles are heading toward.
MOVEMENT_OUTGOING_ROAD = {
    "NT": "South", "NL": "East",
    "ST": "North", "SL": "West",
    "ET": "West",  "EL": "South",
    "WT": "East",  "WL": "North",
}


# =============================================================================
# Intersection configurations
#
# One entry per network geometry. The green/yellow/all_red strings are SUMO
# signal states passed straight to setRedYellowGreenState, so their LENGTH must
# equal the network's controlled-link count (sumo_env.set_phase validates this).
# That is why each lane-count layout needs its own config.
#
# Select which one a run uses with: runner.py --intersection_config <name>
# =============================================================================

INTERSECTION_CONFIGS = {
    # simulations/single_intersection/net.xml: one TL "TLS", 8 controlled links
    # (8-character signal states). Link order: 0=NT 1=ST 2=NL 3=SL 4=ET 5=WT
    # 6=EL 7=WL, so each phase lights exactly two indices.
    "single_intersection": {
        "global_settings": {
            "all_red_state": "rrrrrrrr",
            "default_green_duration": 30,
            "yellow_duration": 3,
            "red_duration": 2,
        },
        "phases": {
            "ETWT": {
                "id": 0,
                "green":  "rrrrGGrr",
                "yellow": "rrrryyrr",
                "llm_description": "- ETWT: Eastern and western through lanes.",
            },
            "NTST": {
                "id": 1,
                "green":  "GGrrrrrr",
                "yellow": "yyrrrrrr",
                "llm_description": "- NTST: Northern and southern through lanes.",
            },
            "ELWL": {
                "id": 2,
                "green":  "rrrrrrGG",
                "yellow": "rrrrrryy",
                "llm_description": "- ELWL: Eastern and western left-turn lanes.",
            },
            "NLSL": {
                "id": 3,
                "green":  "rrGGrrrr",
                "yellow": "rryyrrrr",
                "llm_description": "- NLSL: Northern and southern left-turn lanes.",
            },
        },
        "movement_directions": MOVEMENT_DIRECTIONS,
        "movement_types": MOVEMENT_TYPES,
        "movement_outgoing_road": MOVEMENT_OUTGOING_ROAD,
    },

    # 3 lanes per approach (36-character signal states).
    "three_lane": {
        "global_settings": {
            "all_red_state": "gggrrrrrrgggrrrrrrgggrrrrrrgggrrrrrr",
            "default_green_duration": 30,
            "yellow_duration": 3,
            "red_duration": 2,
        },
        "phases": {
            "ETWT": {
                "id": 0,
                "green":  "gggrrrrrrgggGGGrrrgggrrrrrrgggGGGrrr",
                "yellow": "gggrrrrrrgggyyyrrrgggrrrrrrgggyyyrrr",
                "llm_description": "- ETWT: Eastern and western through lanes.",
            },
            "NTST": {
                "id": 1,
                "green":  "gggGGGrrrgggrrrrrrgggGGGrrrgggrrrrrr",
                "yellow": "gggyyyrrrgggrrrrrrgggyyyrrrgggrrrrrr",
                "llm_description": "- NTST: Northern and southern through lanes.",
            },
            "ELWL": {
                "id": 2,
                "green":  "gggrrrrrrgggrrrGGGgggrrrrrrgggrrrGGG",
                "yellow": "gggrrrrrrgggrrryyygggrrrrrrgggrrryyy",
                "llm_description": "- ELWL: Eastern and western left-turn lanes.",
            },
            "NLSL": {
                "id": 3,
                "green":  "gggrrrGGGgggrrrrrrgggrrrGGGgggrrrrrr",
                "yellow": "gggrrryyygggrrrrrrgggrrryyygggrrrrrr",
                "llm_description": "- NLSL: Northern and southern left-turn lanes.",
            },
        },
        "movement_directions": MOVEMENT_DIRECTIONS,
        "movement_types": MOVEMENT_TYPES,
        "movement_outgoing_road": MOVEMENT_OUTGOING_ROAD,
    },

    # 2 lanes per approach, hangzhou 1x1 network (16-character signal states).
    "two_lane_1x1": {
        "global_settings": {
            "intersection_id": "intersection_1_1",
            "all_red_state": "rrrrrrrrrrrrrrrr",
            "default_green_duration": 30,
            "yellow_duration": 3,
            "red_duration": 2,
        },
        "phases": {
            "NTST": {
                "id": 0,
                "green":  "GGrrrrrrGGrrrrrr",
                "yellow": "yyrrrrrryyrrrrrr",
                "llm_description": "- NTST: Northern and southern through lanes.",
            },
            "ETWT": {
                "id": 1,
                "green":  "rrrrGGrrrrrrGGrr",
                "yellow": "rrrryyrrrrrryyrr",
                "llm_description": "- ETWT: Eastern and western through lanes.",
            },
            "NLSL": {
                "id": 2,
                "green":  "rrGGrrrrrrGGrrrr",
                "yellow": "rryyrrrrrryyrrrr",
                "llm_description": "- NLSL: Northern and southern left-turn lanes.",
            },
            "ELWL": {
                "id": 3,
                "green":  "rrrrrrGGrrrrrrGG",
                "yellow": "rrrrrryyrrrrrryy",
                "llm_description": "- ELWL: Eastern and western left-turn lanes.",
            },
        },
        "movement_directions": MOVEMENT_DIRECTIONS,
        "movement_types": MOVEMENT_TYPES,
        "movement_outgoing_road": MOVEMENT_OUTGOING_ROAD,
    },
}

# Config used when no --intersection_config is given. NOTE: this 3-lane config
# does NOT match simulations/single_intersection (8 links). To run that committed
# sim, pass: runner.py --intersection_config single_intersection
DEFAULT_INTERSECTION_CONFIG_NAME = "three_lane"

# Back-compat: modules that `from configurations import INTERSECTION_CONFIG`
# get the default config. runner.py overrides this per-run via the registry.
INTERSECTION_CONFIG = INTERSECTION_CONFIGS[DEFAULT_INTERSECTION_CONFIG_NAME]

# Valid phase names, derived from the default config rather than re-listed.
PHASE_NAMES = list(INTERSECTION_CONFIG["phases"].keys())


# =============================================================================
# Speed thresholds (m/s)
#
# These two are deliberately separate -- do not merge them:
#   MIN_SPEED              : network-wide "stopped" cutoff for queue counting.
#   STOP_SPEED_EARLY_QUEUE : per-lane cutoff for "early queued" at the stopline.
# =============================================================================

MIN_SPEED = 0.1
STOP_SPEED_EARLY_QUEUE = 1.39


# =============================================================================
# Lane segmentation
# =============================================================================

# Each lane is split into this many distance segments from the stopline.
LANE_SEGMENT_COUNT = 3


# =============================================================================
# SUMO binaries
# =============================================================================

SUMO_BINARY = "sumo"
SUMO_GUI_BINARY = "sumo-gui"


# =============================================================================
# LLM inference
# =============================================================================

LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.0
LLM_DO_SAMPLE = False
LLM_SYSTEM_PROMPT = (
    "You are an expert in traffic management. You can use your knowledge of "
    "traffic commonsense to solve this traffic signal control tasks."
)
# Machine-specific local model path; override per-run with --llm_path.
LLM_DEFAULT_PATH = (
    "C:/Users/m6722/Research/LLMTSC_SUMO/models/LLMs/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)


# =============================================================================
# Simulation defaults
# =============================================================================

DEFAULT_SIMULATION_STEPS = 3600
# Use forward slashes: backslashes would be parsed as escape sequences
# (e.g. "\4" -> chr(4), "\a" -> bell). Forward slashes work on Windows too.
DEFAULT_SIMULATION_CONFIG = "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.sumocfg"
DEFAULT_START_PHASE = "ETWT"


# =============================================================================
# LLM signal parsing
# =============================================================================

# Extracts the chosen phase from the LLM output. Must stay in sync with the
# <signal> tag that prompt_builder.getPrompt instructs the model to emit.
SIGNAL_REGEX = r"<signal>(.*?)</signal>"


# =============================================================================
# Output directory and file names
# =============================================================================

LOGS_DIR_NAME = "logs"
PHASE_SEQUENCES_DIR_NAME = "phase_sequences"
RERUNS_DIR_NAME = "reruns"

STEP_SUMMARIES_FILENAME = "step_summaries.jsonl"
FINAL_SUMMARY_FILENAME = "final_summary.json"
DECISIONS_FILENAME = "decisions.jsonl"
PHASE_SEQUENCE_FILENAME_SUFFIX = "_phase_sequence.json"

REPLAY_EVENTS_FILENAME = "replay_record.jsonl"
REPLAY_META_FILENAME = "replay_meta.json"


# =============================================================================
# CoLight RL baseline (additive -- does not affect the LLM path)
#
# CoLight is a graph-attention DQN lifted verbatim from
#   github.com/usail-hkust/LLMTSCS @ d5d4180f34edb843e1d1b462d5846c75d6d4533a
# Hyperparameters below reproduce that repo's utils/config.py exactly. See
# PORTING_PLAN.md for the full contract; utils/state_features.py for the seam.
# =============================================================================

# Canonical movement order (source CoLight's list_lane_order). Both the cur_phase
# one-hot and the lane_num_vehicle 8-vector are indexed by this order, so they are
# co-indexed.
COLIGHT_MOVEMENT_ORDER = ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"]

# Per-movement aggregation -> 8 lane features (vs source's literal 12 raw lanes).
# Read by the lifted agent's _cal_len_feature (defaults to 12 when absent).
COLIGHT_NUM_LANE_FEATURES = 8

# Top-k nearest neighbours used to build each intersection's adjacency row.
COLIGHT_TOP_K_ADJACENCY = 5

# One decision per green window; equals PhaseHandler default_green_duration so the
# RL cadence matches both the source (MIN_ACTION_TIME = MEASURE_TIME = 30) and the LLM.
COLIGHT_NUM_ROUNDS = 100

# Reward = COEFF * (per-intersection stopped-vehicle count), averaged over the window.
COLIGHT_REWARD_QUEUE_COEFF = -0.25

# Default training/eval network for the CoLight runner (hangzhou 1x1, two-lane).
COLIGHT_DEFAULT_SIMULATION_CONFIG = (
    "dataset/sumo_version/hangzhou_1x1_bc-tyc_18041607_1h/roadnet.sumocfg"
)
COLIGHT_DEFAULT_INTERSECTION_CONFIG_NAME = "two_lane_1x1"

# Multi-intersection network for Phase 3 (16 TLs, all 36-link/3-lane, so it reuses the
# existing `three_lane` config -- verified to match this net's link ordering). Run with:
#   runner_colight.py --simulation_config <below> --intersection_config three_lane
COLIGHT_4X4_SIMULATION_CONFIG = (
    "dataset/sumo_version/hangzhou_4x4_gudang_18041610_1h/roadnet.sumocfg"
)
COLIGHT_4X4_INTERSECTION_CONFIG_NAME = "three_lane"

# Subdirectory (under a run's records dir) where .h5 weights are written.
COLIGHT_WEIGHTS_DIR_NAME = "weights"
COLIGHT_TRAINING_PROGRESS_FILENAME = "training_progress.jsonl"

# Agent hyperparameters -- reproduced EXACTLY from source utils/config.py
# (DIC_BASE_AGENT_CONF + the CNN_layers extra from run_colight.py).
COLIGHT_AGENT_CONF = {
    "CNN_layers": [[32, 32]],
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

# Feature sets per variant (mirrors source LIST_STATE_FEATURE). adjacency_matrix MUST
# stay last (the agent drops it with LIST_STATE_FEATURE[:-1]).
COLIGHT_FEATURES = ["cur_phase", "lane_num_vehicle", "adjacency_matrix"]

# Advanced CoLight: SAME CoLightAgent, different features (source run_advanced_colight.py).
ADVANCED_COLIGHT_FEATURES = [
    "cur_phase",
    "traffic_movement_pressure_queue_efficient",
    "lane_enter_running_part",
    "adjacency_matrix",
]
# Source uses the same base agent conf for Advanced CoLight; copy so it can diverge later.
ADVANCED_COLIGHT_AGENT_CONF = dict(COLIGHT_AGENT_CONF)
