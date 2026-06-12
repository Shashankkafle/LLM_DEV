"""
prompt_builder.py
Utility functions to build the LLM prompt content based on the current state of the intersection.
"""

from configurations import INTERSECTION_CONFIG

_MOVEMENT_DIRECTION = INTERSECTION_CONFIG["movement_directions"]
_MOVEMENT_TYPE      = INTERSECTION_CONFIG["movement_types"]

def _phase_observation_block(phase_name: str, movement_states: dict) -> str:
    """
    Builds the observation block for one signal phase, e.g.:

        Signal: NLSL
        Allowed lanes: Northern and southern left-turn lanes
        - Early queued: 4 (North), 3 (South), 7 (Total)
        - Segment 1:    0 (North), 0 (South), 0 (Total)
        - Segment 2:    0 (North), 0 (South), 0 (Total)
        - Segment 3:    1 (North), 2 (South), 3 (Total)

    `movement_states` is the sub-dict from SumoEnv.get_state() keyed by
    movement name (e.g. "NTST", "ELWL", …).  Each value has:
        {
            "early_queued": int,
            "segments": {"segment_1": int, "segment_2": int, "segment_3": int},
            "lanes": { lane_id: {...} }   # per-lane detail, not used here
        }
    """

    # Determine which two movement codes belong to this phase.
    # Convention: phase name is two 2-char movement codes concatenated
    # (e.g. "NTST" → ["NT", "ST"], "ELWL" → ["EL", "WL"]).
    mov_a = phase_name[:2]   # e.g. "NT"
    mov_b = phase_name[2:]   # e.g. "ST"

    dir_a = _MOVEMENT_DIRECTION[mov_a]   # e.g. "North"
    dir_b = _MOVEMENT_DIRECTION[mov_b]   # e.g. "South"
    mov_type = _MOVEMENT_TYPE[mov_a]     # "through" or "left-turn"

    # Fetch aggregated data from movement_states.
    # Fall back to zeroed data if a movement isn't present (e.g. absent from map).
    def _get(mov):
        return movement_states.get(mov, {
            "early_queued": 0,
            "segments": {"segment_1": 0, "segment_2": 0, "segment_3": 0}
        })

    data_a = _get(mov_a)
    data_b = _get(mov_b)

    eq_a  = data_a["early_queued"]
    eq_b  = data_b["early_queued"]
    eq_tot = eq_a + eq_b

    segs_a = data_a["segments"]
    segs_b = data_b["segments"]

    s1_a, s1_b = segs_a.get("segment_1", 0), segs_b.get("segment_1", 0)
    s2_a, s2_b = segs_a.get("segment_2", 0), segs_b.get("segment_2", 0)
    s3_a, s3_b = segs_a.get("segment_3", 0), segs_b.get("segment_3", 0)

    # Allowed-lanes description line (matches paper wording)
    allowed_desc = f"{dir_a}ern and {dir_b.lower()}ern {mov_type} lanes"

    lines = [
        f"Signal: {phase_name}",
        f"Allowed lanes: {allowed_desc}",
        f"- Early queued: {eq_a} ({dir_a}), {eq_b} ({dir_b}), {eq_tot} (Total)",
        f"- Segment 1:    {s1_a} ({dir_a}), {s1_b} ({dir_b}), {s1_a + s1_b} (Total)",
        f"- Segment 2:    {s2_a} ({dir_a}), {s2_b} ({dir_b}), {s2_a + s2_b} (Total)",
        f"- Segment 3:    {s3_a} ({dir_a}), {s3_b} ({dir_b}), {s3_a + s3_b} (Total)",
    ]
    return "\n".join(lines)


def build_observation(state_dict: dict, phases: list) -> str:
    """
    Assembles the full observation section for all phases, separated by
    blank lines — ready to be dropped into the user prompt.
    """
    movement_states = state_dict.get("movement_states", {})
    blocks = []
    for phase_name in phases:
        if phase_name in INTERSECTION_CONFIG["phases"]:
            blocks.append(_phase_observation_block(phase_name, movement_states))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert traffic signal control agent. "
    "Traffic congestion is primarily dictated by early queued vehicles — "
    "they have the most significant impact, so you must pay the most "
    "attention to lanes with long queue lengths. "
    "It is NOT urgent to consider vehicles in distant segments since they "
    "are unlikely to reach the intersection soon."
)


def getPrompt(state_dict: dict, phases: list) -> list[dict]:
    """
    Returns a messages list  [{"role": ..., "content": ...}, ...]
    ready to pass directly to an LLM inference call.

    Parameters
    ----------
    state_dict : dict
        Output of SumoEnv.get_state(intersection_id).
        Must contain a "movement_states" key.
    phases : list[str]
        Ordered list of phase names to present to the LLM,
        e.g. ["NTST", "ETWT", "NLSL", "ELWL"].
    """
    observation_text = build_observation(state_dict, phases)
    num_phases = len(phases)
    phase_list_str = ", ".join(phases)

    user_content = (
        "A traffic light regulates a four-section intersection with northern, "
        "southern, eastern, and western sections. Each section has two incoming "
        "lanes: one for through traffic and one for left-turns. Each lane is "
        "divided into three segments — Segment 1 is closest to the intersection, "
        "Segment 2 is in the middle, and Segment 3 is the farthest.\n\n"
        "Early queued vehicles have arrived at the intersection and are waiting "
        "for a green signal. Approaching vehicles are travelling toward the "
        "intersection and are counted per segment.\n\n"
        f"The traffic light has {num_phases} signal phases: {phase_list_str}. "
        "The state of the intersection is listed below:\n\n"
        f"{observation_text}\n\n"
        "Please answer:\n"
        "Which is the most effective traffic signal that will most significantly "
        "improve the traffic condition during the next phase?\n\n"
        "Requirements:\n"
        "- Let's think step by step.\n"
        "- You can only choose one of the signals listed above.\n"
        "- Follow these steps:\n"
        "  Step 1: Analyze the traffic conditions for each signal — consider "
        "early queued vehicles (highest priority) and approaching vehicles in "
        "nearby segments.\n"
        "  Step 2: State your chosen signal.\n"
        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]