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

# Config used when no --intersection_config is given. Preserves prior behavior.
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
DEFAULT_SIMULATION_CONFIG = "simulations/single_intersection/run.sumocfg"
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
