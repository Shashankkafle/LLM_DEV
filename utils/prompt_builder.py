"""Builds the LLM prompt content from the current state of the intersection.

The observation format (block order, wording, spacing, and the quoted direction
names on the Segment lines) is kept byte-identical to what earlier runs used and
what LightGPT-style fine-tunes expect. Do not reformat it casually.

INVARIANT: with no active blockages the returned prompt is byte-identical to
what it was before the blockage feature existed, so old and new runs stay
comparable. tests/test_prompt_builder.py pins this. Blockage sections (approach
side and exit side) are pure insertions before "Please answer:" and vanish
entirely when their lists are empty.
"""

from configurations import MOVEMENT_TYPES, BLOCKAGE_METHOD_OBSTACLE

# Fixed presentation of the phase blocks: block order AND direction order within
# each block, exactly as past runs and the fine-tuned models saw them. Kept as an
# explicit table (not derived from phase names or the config's phase order) so
# the prompt cannot drift if either convention changes.
_PROMPT_PHASES = {
    "ETWT": ("East", "West"),
    "ELWL": ("East", "West"),
    "NTST": ("North", "South"),
    "NLSL": ("North", "South"),
}


def _phase_block(phase_name: str, directions: tuple, approaches: dict) -> str:
    """One "Signal:" observation block for a phase like "ETWT"."""
    dir_a, dir_b = directions
    a = approaches[dir_a]
    b = approaches[dir_b]

    lines = [
        f"Signal: {phase_name}",
        f"- Early queued: {a['early_queued']} ({dir_a}), {b['early_queued']} ({dir_b}), "
        f"{a['early_queued'] + b['early_queued']} (Total)",
    ]
    for i in (1, 2, 3):
        seg = f"segment_{i}"
        count_a = a["segments"][seg]
        count_b = b["segments"][seg]
        lines.append(
            f"- Segment {i}: {count_a} ('{dir_a}'), {count_b} ('{dir_b}'), "
            f"{count_a + count_b} (Total)"
        )
    return "\n".join(lines)


def build_observation(state_dict: dict) -> str:
    """Assembles the observation section for all phases, one block per phase,
    separated by blank lines -- ready to be dropped into the user prompt."""
    movement_states = state_dict["movement_states"]
    return "\n\n".join(
        _phase_block(name, directions, movement_states[name])
        for name, directions in _PROMPT_PHASES.items()
    )


def _cause(blockage: dict) -> str:
    """Shared method wording for both blockage sections. Kept verbatim from the
    source repo -- the approach section's bytes are pinned by golden tests."""
    if blockage["method"] == BLOCKAGE_METHOD_OBSTACLE:
        return "stopped vehicle — full blockage"
    return f"speed restriction — {int(blockage['severity'] * 100)}% reduction"


def _blockage_line(blockage: dict) -> str:
    """One bullet for a blockage description (from SumoEnv.describe_blockages),
    in the prompt's own vocabulary -- approach + lane role + segment, never raw
    lane IDs."""
    approach = blockage["approach"]
    movement = blockage["movement"]
    if movement is not None:
        # "ETWT" + approach "West" -> movement code "WT" -> "through".
        code = next(c for c in (movement[:2], movement[2:])
                    if c[0] == approach[0])
        lane_role = f"{MOVEMENT_TYPES[code]} lane (signal {movement})"
    else:
        lane_role = "right-turn lane (not served by any signal)"
    return f"- {approach} approach {lane_role}, segment {blockage['segment']}: {_cause(blockage)}."


def build_blockage_section(blockages) -> str:
    """Renders active blockage descriptions as a prompt section; '' when there
    are none, which is what keeps no-blockage prompts byte-identical."""
    if not blockages:
        return ""
    lines = [
        "LANE BLOCKAGE CONTEXT",
        "The following lane blockages are currently active and restrict vehicle flow:",
    ]
    lines += [_blockage_line(b) for b in blockages]
    lines.append(
        "On blocked lanes, the queued counts above INCLUDE vehicles trapped "
        "behind the blockage that cannot reach the intersection until it clears."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit-side (upstream controller) blockage wording. Unlike the approach section
# above -- whose bytes are pinned for comparability with logged runs -- this
# wording is NEW and free to tune: edit these constants (format specs included)
# and update EXPECTED_EXIT_SECTION in tests/test_prompt_builder.py to match.
# ---------------------------------------------------------------------------
EXIT_BLOCKAGE_HEADER = "DOWNSTREAM BLOCKAGE CONTEXT"
EXIT_BLOCKAGE_INTRO = ("The following roads LEAVING this intersection are "
                       "blocked further downstream:")
EXIT_BLOCKAGE_LINE = ("- The road exiting toward the {direction} "
                      "({blocked_lanes} of {lane_count} lanes blocked, "
                      "{distance_m:.0f} m past this intersection): {cause}. {served}")
EXIT_SERVED_BY = "Signals {movements} release vehicles onto this road."
EXIT_SERVED_NONE = "Only right-turning vehicles enter this road."
EXIT_BLOCKAGE_FOOTER = ("Extended green for the signals feeding a blocked road "
                        "can queue vehicles onto it and cause spillback into "
                        "this intersection.")


def _exit_blockage_line(blockage: dict) -> str:
    """One bullet for an exit blockage (from SumoEnv.describe_exit_blockages).
    For speed restrictions the distance is nominal -- the restriction covers
    the whole lane (same precedent as the approach section's segment)."""
    movements = blockage["feeding_movements"]
    if movements:
        served = EXIT_SERVED_BY.format(movements=", ".join(movements))
    else:
        served = EXIT_SERVED_NONE
    return EXIT_BLOCKAGE_LINE.format(
        direction=blockage["exit_direction"],
        blocked_lanes=1,
        lane_count=blockage["lane_count"],
        distance_m=blockage["distance_m"],
        cause=_cause(blockage),
        served=served,
    )


def build_exit_blockage_section(exit_blockages) -> str:
    """Renders blockages on roads leaving the intersection as a prompt section;
    '' when there are none, so prompts without exit blockages are unchanged."""
    if not exit_blockages:
        return ""
    lines = [EXIT_BLOCKAGE_HEADER, EXIT_BLOCKAGE_INTRO]
    lines += [_exit_blockage_line(b) for b in exit_blockages]
    lines.append(EXIT_BLOCKAGE_FOOTER)
    return "\n".join(lines)


def get_prompt(state_dict: dict, blockages=None, exit_blockages=None) -> str:
    observation_text = build_observation(state_dict)
    sections = [s for s in (build_blockage_section(blockages),
                            build_exit_blockage_section(exit_blockages)) if s]
    if sections:
        blockage_section = "\n" + "\n\n".join(sections) + "\n\n"
    else:
        blockage_section = ""
    return (
        "A traffic light regulates a four-section intersection with northern, southern, eastern, and western "
        "sections, each containing two lanes: one for through traffic and one for left-turns. Each lane is "
        "further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the "
        "middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching "
        "vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and "
        "await passage permission. Approaching vehicles will arrive at the intersection in the future.\n\n"
        "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
        "specific lanes. The state of the intersection is listed below. It describes:\n"
        "- The group of lanes relieving vehicles' flow under each traffic light phase.\n"
        "- The number of early queued vehicles of the allowed lanes of each signal.\n"
        "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
        f"{observation_text}\n"
        f"{blockage_section}"
        "Please answer:\n"
        "Which is the most effective traffic signal that will most significantly improve the traffic "
        "condition during the next phase?\n\n"
        "Requirements:\n"
        "- Let's think step by step.\n"
        "- You can only choose one of the signals listed above.\n"
        "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
        "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
        "- Your choice can only be given after finishing the analysis.\n"
        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
    )
