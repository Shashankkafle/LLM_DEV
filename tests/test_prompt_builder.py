"""Golden tests pinning the LLM prompt against drift.

THE invariant: with no active blockages, get_prompt returns byte-for-byte what
it returned before the blockage feature existed -- old and new runs stay
comparable, and LightGPT-style fine-tunes keep seeing their training template.
The blockage sections may only ever be pure insertions between the observation
and "Please answer:".

The blockage wording implements the event-text prompt templates v2.1
(prompt-review design doc, 2026-07-13). The goldens below reproduce those
templates; changing either side needs a decisions-log entry.

All fixtures are env-shaped (they carry the per-lane "blocked" flags exactly as
SumoEnv.get_state emits them) to prove the flags cannot leak into the text.

Run: PYTHONPATH=<repo root> python tests/test_prompt_builder.py   (or pytest)
"""

import copy
import hashlib

from configurations import LLM_COMMONSENSE_BLOCK, LLM_SYSTEM_PROMPT
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

# Shaped exactly like SumoEnv.describe_blockages output. The second entry sits
# on a lane no signal serves (movement None) and must be suppressed.
FULL_BLOCKAGE = {
    "lane_id": "W2TLS_0", "approach": "West", "movement": "ETWT",
    "segment": 3, "method": "obstacle_vehicle", "severity": 1.0,
    "cause": "stopped vehicle",
}
UNSIGNALIZED_BLOCKAGE = {
    "lane_id": "E2TLS_2", "approach": "East", "movement": None,
    "segment": 1, "method": "speed_restriction", "severity": 0.6,
    "cause": "roadworks",
}
PARTIAL_BLOCKAGE = {
    "lane_id": "W2TLS_1", "approach": "West", "movement": "ELWL",
    "segment": 2, "method": "speed_restriction", "severity": 0.6,
    "cause": "stopped delivery vehicle",
}
BLOCKAGES = [FULL_BLOCKAGE, UNSIGNALIZED_BLOCKAGE]

EXPECTED_SECTION = (
    "LANE BLOCKAGE REPORT (approaches to this intersection)\n"
    "The following blockage is currently active on an approach to this "
    "intersection:\n"
    "- West approach, through lane (served by signal ETWT), segment 3: "
    "stopped vehicle — the lane is fully blocked. Vehicles behind the "
    "blockage in this lane cannot reach the intersection until it clears. "
    "Within signal ETWT, the West queued and approaching counts reported "
    "above include these vehicles; the East counts are unaffected by this "
    "blockage."
)

EXPECTED_PARTIAL_SECTION = (
    "LANE BLOCKAGE REPORT (approaches to this intersection)\n"
    "The following blockage is currently active on an approach to this "
    "intersection:\n"
    "- West approach, left-turn lane (served by signal ELWL), segment 2: "
    "stopped delivery vehicle — the lane is partially blocked. Vehicles can "
    "pass the obstruction slowly, so this lane discharges at a reduced rate. "
    "Within signal ELWL, the West queued and approaching counts reported "
    "above include vehicles delayed behind the obstruction; the East counts "
    "are unaffected by this blockage."
)

# Shaped exactly like SumoEnv.describe_exit_blockages output. The second entry
# is a road only right-turning vehicles enter (no feeding movements) and must
# be suppressed.
EXIT_BLOCKAGE_SOUTH = {
    "lane_id": "road_1_4_3_0", "exit_direction": "South",
    "feeding_movements": ["ELWL", "NTST"],
    "blocked_lane_feeding_movements": ["NTST"], "blocked_lane_index": 0,
    "lane_count": 3, "distance_m": 562.8, "lane_length_m": 572.8,
    "method": "obstacle_vehicle", "severity": 1.0, "cause": "collision",
}
EXIT_BLOCKAGE_RIGHT_TURN_ONLY = {
    "lane_id": "road_2_2_0_1", "exit_direction": "East",
    "feeding_movements": [],
    "blocked_lane_feeding_movements": [], "blocked_lane_index": 1,
    "lane_count": 3, "distance_m": 100.0, "lane_length_m": 300.0,
    "method": "speed_restriction", "severity": 0.6, "cause": "roadworks",
}
EXIT_BLOCKAGES = [EXIT_BLOCKAGE_SOUTH, EXIT_BLOCKAGE_RIGHT_TURN_ONLY]

EXPECTED_EXIT_SECTION = (
    "DOWNSTREAM BLOCKAGE REPORT (roads leaving this intersection)\n"
    "The following road leaving this intersection is blocked further "
    "downstream:\n"
    "- Road exiting toward the South: collision — 1 of 3 lanes fully "
    "blocked, located 563 m past this intersection on a 573 m link. "
    "Movements releasing vehicles onto this road: North→South through (part "
    "of signal NTST) and East→South left turn (part of signal ELWL)."
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


# --------------------------------------------------------------------------
# The commonsense block (event-text templates v2.1, section 0): generic
# domain guidance shared by BOTH LLM arms via the system prompt, so the
# informed arm's advantage can only come from the event facts.
# --------------------------------------------------------------------------

def test_commonsense_block_is_pinned():
    assert LLM_COMMONSENSE_BLOCK == (
        "Incidents may block lanes anywhere in the network. A queue trapped "
        "behind a full lane blockage cannot discharge even when its signal is "
        "green. A road that is blocked further downstream has reduced storage "
        "and discharge capacity, and vehicles released onto it may queue back "
        "into the intersection. When such information is available, consider "
        "whether queued vehicles are actually able to move before allocating "
        "green time to them.")


def test_system_prompt_is_pinned():
    assert LLM_SYSTEM_PROMPT == (
        "You are an expert in traffic management. You can use your knowledge "
        "of traffic commonsense to solve this traffic signal control tasks. "
        + LLM_COMMONSENSE_BLOCK)


# --------------------------------------------------------------------------
# The approach-side blockage section (v2.1 upstream templates)
# --------------------------------------------------------------------------

def test_blockage_section_golden():
    assert build_blockage_section(BLOCKAGES) == EXPECTED_SECTION


def test_partial_blockage_golden():
    assert build_blockage_section([PARTIAL_BLOCKAGE]) == EXPECTED_PARTIAL_SECTION


def test_unsignalized_blockage_is_suppressed():
    # v2.1 section 3: no signal serves the lane, so there is no action the
    # report could inform -- and no section means the prompt stays base.
    assert build_blockage_section([UNSIGNALIZED_BLOCKAGE]) == ""
    assert get_prompt(STATE, blockages=[UNSIGNALIZED_BLOCKAGE]) == get_prompt(STATE)


def test_zero_speed_restriction_counts_as_full():
    # A speed restriction down to zero makes the lane impassable, so it takes
    # the full-blockage wording even though the method is speed_restriction.
    frozen = dict(PARTIAL_BLOCKAGE, severity=1.0, cause="collision")
    section = build_blockage_section([frozen])
    assert "the lane is fully blocked" in section
    assert "partially" not in section


def test_full_approach_emits_one_bullet_per_lane():
    # S3: both lanes of one approach blocked -> two bullets, two signals --
    # never collapsed into one "approach fully blocked" line.
    through = dict(FULL_BLOCKAGE, lane_id="N2TLS_0", approach="North",
                   movement="NTST", segment=2, cause="collision")
    left = dict(FULL_BLOCKAGE, lane_id="N2TLS_1", approach="North",
                movement="NLSL", segment=2, cause="collision")
    section = build_blockage_section([through, left])
    assert ("- North approach, through lane (served by signal NTST), "
            "segment 2: collision") in section
    assert ("- North approach, left-turn lane (served by signal NLSL), "
            "segment 2: collision") in section
    assert section.count("\n- ") == 2
    assert section.count("the South counts are unaffected by this blockage") == 2


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
# The exit (upstream-controller) blockage section (v2.1 downstream template)
# --------------------------------------------------------------------------

def test_exit_blockage_section_golden():
    assert build_exit_blockage_section(EXIT_BLOCKAGES) == EXPECTED_EXIT_SECTION


def test_empty_exit_list_changes_nothing():
    base = get_prompt(STATE)
    assert get_prompt(STATE, exit_blockages=None) == base
    assert get_prompt(STATE, exit_blockages=[]) == base


def test_right_turn_only_exit_road_is_suppressed():
    # No listed movement releases vehicles onto this road, so the report could
    # not inform any action (same principle as the unsignalized approach lane).
    assert build_exit_blockage_section([EXIT_BLOCKAGE_RIGHT_TURN_ONLY]) == ""
    assert (get_prompt(STATE, exit_blockages=[EXIT_BLOCKAGE_RIGHT_TURN_ONLY])
            == get_prompt(STATE))


def test_partial_exit_blockage_wording():
    partial = dict(EXIT_BLOCKAGE_SOUTH, method="speed_restriction",
                   severity=0.6, cause="roadworks",
                   distance_m=100.0, lane_length_m=300.0)
    section = build_exit_blockage_section([partial])
    assert ("- Road exiting toward the South: roadworks — 1 of 3 lanes "
            "partially blocked, located 100 m past this intersection on a "
            "300 m link; vehicles pass the obstruction slowly. Movements "
            "releasing vehicles onto this road: North→South through (part of "
            "signal NTST) and East→South left turn (part of signal ELWL)."
            ) in section


def test_two_blocked_lanes_on_one_exit_road_share_a_bullet():
    # Two lanes of the same exit road -> "2 of 3 lanes fully blocked", quoting
    # the blockage nearest the intersection (it bounds the remaining storage).
    second = dict(EXIT_BLOCKAGE_SOUTH, lane_id="road_1_4_3_1",
                  blocked_lane_index=1, distance_m=400.0)
    section = build_exit_blockage_section([EXIT_BLOCKAGE_SOUTH, second])
    assert section.count("\n- ") == 1
    assert "2 of 3 lanes fully blocked, located 400 m past this intersection" in section


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
