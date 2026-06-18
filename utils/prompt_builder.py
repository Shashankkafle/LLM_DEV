"""
prompt_builder.py
Utility functions to build the LLM prompt content based on the current state of the intersection.
"""

from configurations import INTERSECTION_CONFIG

_MOVEMENT_DIRECTION = INTERSECTION_CONFIG["movement_directions"]
_MOVEMENT_TYPE      = INTERSECTION_CONFIG["movement_types"]

# def _phase_observation_block(phase_name: str, movement_states: dict) -> str:
#     """
#     Builds the observation block for one signal phase exactly as the 
#     original LLMLight Commonsense implementation does.
#     """


#     # Note: Pay strict attention to the single spaces after colons here. Do not add tabs.
#     block = (f"Signal: {phase_name}\n"
#              f"- Early queued: {movement_states[phase_name]['early_queued']} ({dir_a}), {eq_b} ({dir_b}), {eq_tot} (Total)\n"
#              f"- Segment 1: {s1_a} ({dir_a}), {s1_b} ({dir_b}), {s1_a + s1_b} (Total)\n"
#              f"- Segment 2: {s2_a} ({dir_a}), {s2_b} ({dir_b}), {s2_a + s2_b} (Total)\n"
#              f"- Segment 3: {s3_a} ({dir_a}), {s3_b} ({dir_b}), {s3_a + s3_b} (Total)")
    
#     return block
def build_observation(state_dict: dict) -> str:
    """
    Assembles the full observation section for all phases, separated by
    blank lines — ready to be dropped into the user prompt.
    """
    movement_states = state_dict.get("movement_states")

    ETWT_block = (f"Signal: ETWT\n"
             f"- Early queued: {movement_states['ETWT']['East']['early_queued']} (East), {movement_states['ETWT']['West']['early_queued']} (West), {movement_states['ETWT']['West']['early_queued']+movement_states['ETWT']['East']['early_queued']} (Total)\n"
             f"- Segment 1: {movement_states['ETWT']['East']['segments']['segment_1']} ('East'), {movement_states['ETWT']['West']['segments']['segment_1']} ('West'), {movement_states['ETWT']['East']['segments']['segment_1'] + movement_states['ETWT']['West']['segments']['segment_1']} (Total)\n"
             f"- Segment 2: {movement_states['ETWT']['East']['segments']['segment_2']} ('East'), {movement_states['ETWT']['West']['segments']['segment_2']} ('West'), {movement_states['ETWT']['East']['segments']['segment_2'] + movement_states['ETWT']['West']['segments']['segment_2']} (Total)\n"
             f"- Segment 3: {movement_states['ETWT']['East']['segments']['segment_3']} ('East'), {movement_states['ETWT']['West']['segments']['segment_3']} ('West'), {movement_states['ETWT']['East']['segments']['segment_3'] + movement_states['ETWT']['West']['segments']['segment_3']} (Total)")
    

    ELWL_block = (f"Signal: ELWL\n"
             f"- Early queued: {movement_states['ELWL']['East']['early_queued']} (East), {movement_states['ELWL']['West']['early_queued']} (West), {movement_states['ELWL']['West']['early_queued']+movement_states['ELWL']['East']['early_queued']} (Total)\n"
             f"- Segment 1: {movement_states['ELWL']['East']['segments']['segment_1']} ('East'), {movement_states['ELWL']['West']['segments']['segment_1']} ('West'), {movement_states['ELWL']['East']['segments']['segment_1'] + movement_states['ELWL']['West']['segments']['segment_1']} (Total)\n"
             f"- Segment 2: {movement_states['ELWL']['East']['segments']['segment_2']} ('East'), {movement_states['ELWL']['West']['segments']['segment_2']} ('West'), {movement_states['ELWL']['East']['segments']['segment_2'] + movement_states['ELWL']['West']['segments']['segment_2']} (Total)\n"
             f"- Segment 3: {movement_states['ELWL']['East']['segments']['segment_3']} ('East'), {movement_states['ELWL']['West']['segments']['segment_3']} ('West'), {movement_states['ELWL']['East']['segments']['segment_3'] + movement_states['ELWL']['West']['segments']['segment_3']} (Total)")
    
    NTST_block = (f"Signal: NTST\n"
             f"- Early queued: {movement_states['NTST']['North']['early_queued']} (North), {movement_states['NTST']['South']['early_queued']} (South), {movement_states['NTST']['South']['early_queued']+movement_states['NTST']['North']['early_queued']} (Total)\n"
             f"- Segment 1: {movement_states['NTST']['North']['segments']['segment_1']} ('North'), {movement_states['NTST']['South']['segments']['segment_1']} ('South'), {movement_states['NTST']['North']['segments']['segment_1'] + movement_states['NTST']['South']['segments']['segment_1']} (Total)\n"
             f"- Segment 2: {movement_states['NTST']['North']['segments']['segment_2']} ('North'), {movement_states['NTST']['South']['segments']['segment_2']} ('South'), {movement_states['NTST']['North']['segments']['segment_2'] + movement_states['NTST']['South']['segments']['segment_2']} (Total)\n"
             f"- Segment 3: {movement_states['NTST']['North']['segments']['segment_3']} ('North'), {movement_states['NTST']['South']['segments']['segment_3']} ('South'), {movement_states['NTST']['North']['segments']['segment_3'] + movement_states['NTST']['South']['segments']['segment_3']} (Total)")
    NLSL_block = (f"Signal: NLSL\n"
             f"- Early queued: {movement_states['NLSL']['North']['early_queued']} (North), {movement_states['NLSL']['South']['early_queued']} (South), {movement_states['NLSL']['South']['early_queued']+movement_states['NLSL']['North']['early_queued']} (Total)\n"
             f"- Segment 1: {movement_states['NLSL']['North']['segments']['segment_1']} ('North'), {movement_states['NLSL']['South']['segments']['segment_1']} ('South'), {movement_states['NLSL']['North']['segments']['segment_1'] + movement_states['NLSL']['South']['segments']['segment_1']} (Total)\n"
             f"- Segment 2: {movement_states['NLSL']['North']['segments']['segment_2']} ('North'), {movement_states['NLSL']['South']['segments']['segment_2']} ('South'), {movement_states['NLSL']['North']['segments']['segment_2'] + movement_states['NLSL']['South']['segments']['segment_2']} (Total)\n"
             f"- Segment 3: {movement_states['NLSL']['North']['segments']['segment_3']} ('North'), {movement_states['NLSL']['South']['segments']['segment_3']} ('South'), {movement_states['NLSL']['North']['segments']['segment_3'] + movement_states['NLSL']['South']['segments']['segment_3']} (Total)")
    

    blocks = [ETWT_block, ELWL_block, NTST_block, NLSL_block]
    # for phase_name in INTERSECTION_CONFIG["phases"]:
    #     blocks.append(_phase_observation_block(phase_name, movement_states))
    return "\n\n".join(blocks)


def getPrompt(state_dict: dict) -> str:
    observation_text = build_observation(state_dict)
    user_content = (
        "A traffic light regulates a four-section intersection with northern, southern, eastern, and western "
                    "sections, each containing two lanes: one for through traffic and one for left-turns. Each lane is "
                    "further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the "
                    "middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching "
                    "vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and "
                    "await passage permission. Approaching vehicles will arrive at the intersection in the future.\n\n"
                    "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                    "specific lanes. The state of the intersection is listed below. It describes:\n"
                    "- The group of lanes relieving vehicles' flow under each signal phase.\n"
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
    
    return user_content