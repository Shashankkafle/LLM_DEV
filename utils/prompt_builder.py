"""
prompt_builder.py
Utility functions to build the LLM prompt content based on the current state of the intersection.
"""

from configurations import INTERSECTION_CONFIG

_MOVEMENT_DIRECTION = INTERSECTION_CONFIG["movement_directions"]
_MOVEMENT_TYPE      = INTERSECTION_CONFIG["movement_types"]

def _phase_observation_block(phase_name: str, movement_states: dict) -> str:
    """
    Builds the observation block for one signal phase exactly as the 
    original LLMLight Commonsense implementation does.
    """

    # Extract movement codes
    mov_a = phase_name[:2]   # e.g. "NT"
    mov_b = phase_name[2:]   # e.g. "ST"

    dir_a = _MOVEMENT_DIRECTION[mov_a]   # e.g. "North"
    dir_b = _MOVEMENT_DIRECTION[mov_b]   # e.g. "South"

    # Fetch aggregated data from movement_states.
    def _get(mov):
        return movement_states.get(mov, {
            "early_queued": 0,
            "segments": {"segment_1": 0, "segment_2": 0, "segment_3": 0}
        })

    data_a = _get(mov_a)
    data_b = _get(mov_b)

    # Cast to int to guarantee exact matching
    eq_a  = int(data_a["early_queued"])
    eq_b  = int(data_b["early_queued"])
    eq_tot = eq_a + eq_b

    segs_a = data_a["segments"]
    segs_b = data_b["segments"]

    s1_a, s1_b = int(segs_a.get("segment_1", 0)), int(segs_b.get("segment_1", 0))
    s2_a, s2_b = int(segs_a.get("segment_2", 0)), int(segs_b.get("segment_2", 0))
    s3_a, s3_b = int(segs_a.get("segment_3", 0)), int(segs_b.get("segment_3", 0))

    # Slice [8:-1] exactly as the original authors did to format the 'Allowed lanes' string
    # E.g., "- NTST: Northern and southern through lanes." -> "Northern and southern through lanes"
    allowed_desc = INTERSECTION_CONFIG["phases"][phase_name]["llm_description"][8:-1]

    # Note: Pay strict attention to the single spaces after colons here. Do not add tabs.
    block = (f"Signal: {phase_name}\n"
             f"Allowed lanes: {allowed_desc}\n"
             f"- Early queued: {eq_a} ({dir_a}), {eq_b} ({dir_b}), {eq_tot} (Total)\n"
             f"- Segment 1: {s1_a} ({dir_a}), {s1_b} ({dir_b}), {s1_a + s1_b} (Total)\n"
             f"- Segment 2: {s2_a} ({dir_a}), {s2_b} ({dir_b}), {s2_a + s2_b} (Total)\n"
             f"- Segment 3: {s3_a} ({dir_a}), {s3_b} ({dir_b}), {s3_a + s3_b} (Total)")
    
    return block
def build_observation(state_dict: dict) -> str:
    """
    Assembles the full observation section for all phases, separated by
    blank lines — ready to be dropped into the user prompt.
    """
    movement_states = state_dict.get("movement_states", {})
    blocks = []
    for phase_name in INTERSECTION_CONFIG["phases"]:
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


def getPrompt(state_dict: dict) -> str:
    observation_text = build_observation(state_dict)
    
    user_content = (
        "A crossroad connects two roads: the north-south and east-west. The traffic light is located at "
        "the intersection of the two roads. The north-south road is divided into two sections by the intersection: "
        "the north and south. Similarly, the east-west road is divided into the east and west. Each section "
        "has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. "
        "Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. "
        "In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. "
        "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
        "vehicles will arrive at the intersection in the future.\n\n"
        "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
        "specific lanes. The state of the intersection is listed below. It describes:\n"
        "- The group of lanes relieving vehicles' flow under each traffic light phase.\n"
        "- The number of early queued vehicles of the allowed lanes of each signal.\n"
        "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
        f"{observation_text}\n"
        "Please answer:\n"
        "Which is the most effective traffic signal that will most significantly improve the traffic "
        "condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?\n\n"
        "Note:\n"
        "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant "
        "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to "
        "consider vehicles in distant segments since they are unlikely to reach the intersection soon.\n\n"
        "Requirements:\n"
        "- Let's think step by step.\n"
        "- You can only choose one of the signals listed above.\n"
        "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
        "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
        "- Your choice can only be given after finishing the analysis.\n"
        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
    )
    
    return user_content