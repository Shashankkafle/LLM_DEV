"""Golden tests pinning the LLM prompt against drift.

THE invariant: with no active blockages, get_prompt returns byte-for-byte what
it returned before the blockage feature existed -- old and new runs stay
comparable, and LightGPT-style fine-tunes keep seeing their training template.
The blockage section may only ever be a pure insertion between the observation
and "Please answer:".

All fixtures are env-shaped (they carry the per-lane "blocked" flags exactly as
SumoEnv.get_state emits them) to prove the flags cannot leak into the text.

Run: PYTHONPATH=<repo root> python tests/test_prompt_builder.py   (or pytest)
"""

import copy
import hashlib

from configurations import LLM_SYSTEM_PROMPT
from utils.prompt_builder import (build_blockage_section,
                                  build_exit_blockage_section, get_prompt)

# sha256 of get_prompt(STATE) for the fixture below, recorded when the blockage
# feature landed. If this changes, the no-blockage prompt drifted -- which
# breaks comparability with every logged run. Change it only deliberately.
EXPECTED_PROMPT_SHA256 = "5844c89b44d97de58ddd9cb4fdaea4b624ec2637e4b4dbf843b02909080bc98d"


def _approach(early, s1, s2, s3):
    segments = {"segment_1": s1, "segment_2": s2, "segment_3": s3}
    return {
        "early_queued": early,
        "segments": dict(segments),
        "lanes": {"L_0": {"early_queued": early, "segments": dict(segments),
                          "blocked": False}},
    }


STATE = {
    "current_phase": 0,
    "lane_states": {},
    "movement_states": {
        "ETWT": {"East": _approach(3, 0, 1, 0), "West": _approach(2, 0, 0, 0)},
        "ELWL": {"East": _approach(0, 1, 0, 0), "West": _approach(1, 0, 0, 2)},
        "NTST": {"North": _approach(4, 0, 0, 0), "South": _approach(0, 0, 3, 0)},
        "NLSL": {"North": _approach(0, 0, 0, 5), "South": _approach(6, 1, 0, 0)},
    },
}

BLOCKAGES = [
    {"lane_id": "W2TLS_0", "approach": "West", "movement": "ETWT",
     "segment": 3, "method": "obstacle_vehicle", "severity": 1.0},
    {"lane_id": "E2TLS_2", "approach": "East", "movement": None,
     "segment": 1, "method": "speed_restriction", "severity": 0.6},
]

EXPECTED_SECTION = (
    "LANE BLOCKAGE CONTEXT\n"
    "The following lane blockages are currently active and restrict vehicle flow:\n"
    "- West approach through lane (signal ETWT), segment 3: "
    "stopped vehicle — full blockage.\n"
    "- East approach right-turn lane (not served by any signal), segment 1: "
    "speed restriction — 60% reduction.\n"
    "On blocked lanes, the queued counts above INCLUDE vehicles trapped behind "
    "the blockage that cannot reach the intersection until it clears."
)

# Shaped exactly like SumoEnv.describe_exit_blockages output.
EXIT_BLOCKAGES = [
    {"lane_id": "road_1_4_3_0", "exit_direction": "South",
     "feeding_movements": ["ELWL", "NTST"], "blocked_lane_index": 0,
     "lane_count": 3, "distance_m": 562.8, "lane_length_m": 572.8,
     "method": "obstacle_vehicle", "severity": 1.0},
    {"lane_id": "road_2_2_0_1", "exit_direction": "East",
     "feeding_movements": [], "blocked_lane_index": 1,
     "lane_count": 3, "distance_m": 100.0, "lane_length_m": 300.0,
     "method": "speed_restriction", "severity": 0.6},
]

# Literal golden (not a sha) so tuning the exit wording is a visible diff here.
EXPECTED_EXIT_SECTION = (
    "DOWNSTREAM BLOCKAGE CONTEXT\n"
    "The following roads LEAVING this intersection are blocked further downstream:\n"
    "- The road exiting toward the South (1 of 3 lanes blocked, 563 m past this "
    "intersection): stopped vehicle — full blockage. Signals ELWL, NTST release "
    "vehicles onto this road.\n"
    "- The road exiting toward the East (1 of 3 lanes blocked, 100 m past this "
    "intersection): speed restriction — 60% reduction. Only right-turning "
    "vehicles enter this road.\n"
    "Extended green for the signals feeding a blocked road can queue vehicles "
    "onto it and cause spillback into this intersection."
)


# --------------------------------------------------------------------------
# The byte-identity invariant
# --------------------------------------------------------------------------

def test_no_blockage_prompt_is_pinned():
    digest = hashlib.sha256(get_prompt(STATE).encode("utf-8")).hexdigest()
    assert digest == EXPECTED_PROMPT_SHA256, (
        f"The no-blockage prompt changed (sha256 {digest}). This breaks "
        f"comparability with every logged run -- if intentional, update the pin.")


def test_empty_blockage_list_changes_nothing():
    base = get_prompt(STATE)
    assert get_prompt(STATE, blockages=None) == base
    assert get_prompt(STATE, blockages=[]) == base


def test_blocked_flags_in_state_cannot_leak():
    flagged = copy.deepcopy(STATE)
    for phase in flagged["movement_states"].values():
        for approach in phase.values():
            for lane in approach["lanes"].values():
                lane["blocked"] = True
    assert get_prompt(flagged) == get_prompt(STATE)


def test_system_prompt_is_pinned():
    assert LLM_SYSTEM_PROMPT == (
        "You are an expert in traffic management. You can use your knowledge of "
        "traffic commonsense to solve this traffic signal control tasks.")


# --------------------------------------------------------------------------
# The blockage section
# --------------------------------------------------------------------------

def test_blockage_section_golden():
    assert build_blockage_section(BLOCKAGES) == EXPECTED_SECTION


def test_left_turn_lane_wording():
    section = build_blockage_section([
        {"lane_id": "W2TLS_1", "approach": "West", "movement": "ELWL",
         "segment": 2, "method": "obstacle_vehicle", "severity": 1.0}])
    assert ("- West approach left-turn lane (signal ELWL), segment 2: "
            "stopped vehicle — full blockage.") in section


def test_blockage_section_is_pure_insertion():
    # An active-blockage prompt must be the base prompt with the section
    # inserted before "Please answer:" -- and NOTHING else changed. This pins
    # the shared text in the active branch too, not just the empty one.
    base = get_prompt(STATE)
    with_blockages = get_prompt(STATE, blockages=BLOCKAGES)
    idx = base.index("Please answer:")
    expected = base[:idx] + "\n" + EXPECTED_SECTION + "\n\n" + base[idx:]
    assert with_blockages == expected


# --------------------------------------------------------------------------
# The exit (upstream-controller) blockage section
# --------------------------------------------------------------------------

def test_exit_blockage_section_golden():
    assert build_exit_blockage_section(EXIT_BLOCKAGES) == EXPECTED_EXIT_SECTION


def test_empty_exit_list_changes_nothing():
    base = get_prompt(STATE)
    assert get_prompt(STATE, exit_blockages=None) == base
    assert get_prompt(STATE, exit_blockages=[]) == base


def test_right_turn_only_exit_wording():
    section = build_exit_blockage_section([EXIT_BLOCKAGES[1]])
    assert "Only right-turning vehicles enter this road." in section


def test_exit_only_is_pure_insertion():
    base = get_prompt(STATE)
    idx = base.index("Please answer:")
    expected = base[:idx] + "\n" + EXPECTED_EXIT_SECTION + "\n\n" + base[idx:]
    assert get_prompt(STATE, exit_blockages=EXIT_BLOCKAGES) == expected


def test_combined_sections_are_pure_insertion():
    # Approach section first, exit section second, blank line between -- and
    # nothing else changed.
    base = get_prompt(STATE)
    idx = base.index("Please answer:")
    expected = (base[:idx] + "\n" + EXPECTED_SECTION + "\n\n"
                + EXPECTED_EXIT_SECTION + "\n\n" + base[idx:])
    assert get_prompt(STATE, blockages=BLOCKAGES,
                      exit_blockages=EXIT_BLOCKAGES) == expected


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
