"""Builds the LLM prompt content from the current state of the intersection.

The observation format (block order, wording, spacing, and the quoted direction
names on the Segment lines) is kept byte-identical to what earlier runs used and
what LightGPT-style fine-tunes expect. Do not reformat it casually.
"""

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


def get_prompt(state_dict: dict) -> str:
    observation_text = build_observation(state_dict)
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
