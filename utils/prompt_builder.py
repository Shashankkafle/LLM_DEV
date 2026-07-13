"""Builds the LLM prompt content from the current state of the intersection.

The observation format (block order, wording, spacing, and the quoted direction
names on the Segment lines) is kept byte-identical to what earlier runs used and
what LightGPT-style fine-tunes expect. Do not reformat it casually.

INVARIANT: with no active blockages the returned prompt is byte-identical to
what it was before the blockage feature existed, so old and new runs stay
comparable. tests/test_prompt_builder.py pins this. Blockage sections (approach
side and exit side) are pure insertions before "Please answer:" and vanish
entirely when there is nothing to report.

The blockage sections implement the event-text prompt templates v2.1
(prompt-review design doc, 2026-07-13): per-event text is factual only, with
movement-level attribution and per-approach scoping; the generic guidance on
what a blockage means for control lives in configurations.LLM_COMMONSENSE_BLOCK,
shared by both LLM arms. Wording changes here must update the goldens in
tests/test_prompt_builder.py and get a decisions-log entry.
"""

from configurations import (
    BLOCKAGE_DEFAULT_CAUSE,
    BLOCKAGE_METHOD_OBSTACLE,
    MOVEMENT_DIRECTIONS,
    MOVEMENT_OUTGOING_ROAD,
    MOVEMENT_TYPES,
)

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


# ---------------------------------------------------------------------------
# Shared helpers for both blockage sections
# ---------------------------------------------------------------------------


def _is_full(blockage: dict) -> bool:
    """A blockage is full when the lane is impassable: a physical obstacle, or
    a speed restriction all the way down to zero."""
    return (blockage["method"] == BLOCKAGE_METHOD_OBSTACLE
            or blockage["severity"] >= 1.0)


def _cause(blockage: dict) -> str:
    """Short factual incident description for the {cause} slot. The env
    describers always provide one; the fallback covers hand-built dicts."""
    return blockage.get("cause") or BLOCKAGE_DEFAULT_CAUSE[blockage["method"]]


# ---------------------------------------------------------------------------
# Approach-side section (v2.1 upstream templates): emitted at the intersection
# whose approach contains the blockage. Only blockages on signal-served lanes
# are reported -- the controller has no action that serves e.g. a right-turn
# lane, and that lane's queue is not in the reported counts, so those are
# suppressed rather than described.
# ---------------------------------------------------------------------------
APPROACH_BLOCKAGE_HEADER = "LANE BLOCKAGE REPORT (approaches to this intersection)"
APPROACH_BLOCKAGE_INTRO = ("The following blockage is currently active on an "
                           "approach to this intersection:")
APPROACH_BLOCKAGE_FULL_LINE = (
    "- {approach} approach, {lane_type} lane (served by signal {signal}), "
    "segment {segment}: {cause} — the lane is fully blocked. Vehicles behind "
    "the blockage in this lane cannot reach the intersection until it clears. "
    "Within signal {signal}, the {approach} queued and approaching counts "
    "reported above include these vehicles; the {opposite} counts are "
    "unaffected by this blockage.")
APPROACH_BLOCKAGE_PARTIAL_LINE = (
    "- {approach} approach, {lane_type} lane (served by signal {signal}), "
    "segment {segment}: {cause} — the lane is partially blocked. Vehicles can "
    "pass the obstruction slowly, so this lane discharges at a reduced rate. "
    "Within signal {signal}, the {approach} queued and approaching counts "
    "reported above include vehicles delayed behind the obstruction; the "
    "{opposite} counts are unaffected by this blockage.")


def _approach_blockage_line(blockage: dict) -> str:
    """One bullet for a signal-served blockage (from SumoEnv.describe_blockages),
    in the prompt's own vocabulary -- approach + lane role + segment, never raw
    lane IDs. Names both the corrupted approach and the clean paired approach of
    the same signal, so the model need not reason on the ambiguous Totals."""
    approach = blockage["approach"]
    signal = blockage["movement"]
    # "ETWT" + approach "West" -> movement code "WT" -> "through".
    code = next(c for c in (signal[:2], signal[2:]) if c[0] == approach[0])
    opposite = next(d for d in _PROMPT_PHASES[signal] if d != approach)
    template = (APPROACH_BLOCKAGE_FULL_LINE if _is_full(blockage)
                else APPROACH_BLOCKAGE_PARTIAL_LINE)
    return template.format(approach=approach, lane_type=MOVEMENT_TYPES[code],
                           signal=signal, segment=blockage["segment"],
                           cause=_cause(blockage), opposite=opposite)


def build_blockage_section(blockages) -> str:
    """Renders active approach blockages as a prompt section; '' when there is
    nothing to report, which is what keeps no-blockage prompts byte-identical.
    Blockages on lanes outside every signal (movement None) are suppressed."""
    signal_served = [b for b in (blockages or []) if b["movement"] is not None]
    if not signal_served:
        return ""
    lines = [APPROACH_BLOCKAGE_HEADER, APPROACH_BLOCKAGE_INTRO]
    lines += [_approach_blockage_line(b) for b in signal_served]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit-side section (v2.1 downstream template): emitted at the intersection
# immediately upstream of the blocked exit link. Roads that no listed signal
# feeds (only right-turning vehicles enter them) are suppressed for the same
# reason as unsignalized approach lanes. The spillback warning that used to
# close this section now lives, generalized, in
# configurations.LLM_COMMONSENSE_BLOCK, shared by both LLM arms.
# ---------------------------------------------------------------------------
EXIT_BLOCKAGE_HEADER = "DOWNSTREAM BLOCKAGE REPORT (roads leaving this intersection)"
EXIT_BLOCKAGE_INTRO = ("The following road leaving this intersection is "
                       "blocked further downstream:")
EXIT_BLOCKAGE_FULL_LINE = (
    "- Road exiting toward the {direction}: {cause} — {blocked} of {total} "
    "lanes fully blocked, located {distance:.0f} m past this intersection on "
    "a {length:.0f} m link. Movements releasing vehicles onto this road: "
    "{movements}.")
EXIT_BLOCKAGE_PARTIAL_LINE = (
    "- Road exiting toward the {direction}: {cause} — {blocked} of {total} "
    "lanes partially blocked, located {distance:.0f} m past this intersection "
    "on a {length:.0f} m link; vehicles pass the obstruction slowly. "
    "Movements releasing vehicles onto this road: {movements}.")
EXIT_MOVEMENT_PHRASE = "{origin}→{direction} {kind} (part of signal {signal})"

# The movement list says "left turn" (a noun phrase), unlike the approach
# section's "left-turn lane" (an adjective).
_EXIT_MOVEMENT_KIND = {"through": "through", "left-turn": "left turn"}


def _movement_phrases(feeding_signals, exit_direction) -> list:
    """Movement-level attribution for a blocked exit road: every signal-served
    movement discharging onto it, e.g. 'North→South through (part of signal
    NTST)'. A signal names two movements (NTST = NT + ST); only the one heading
    toward the exit direction feeds this road. Through movements are listed
    before left turns, matching the template's worked example."""
    phrases = []
    for signal in feeding_signals:
        for code in (signal[:2], signal[2:]):
            if MOVEMENT_OUTGOING_ROAD[code] != exit_direction:
                continue
            phrases.append((MOVEMENT_TYPES[code] != "through",
                            EXIT_MOVEMENT_PHRASE.format(
                                origin=MOVEMENT_DIRECTIONS[code],
                                direction=exit_direction,
                                kind=_EXIT_MOVEMENT_KIND[MOVEMENT_TYPES[code]],
                                signal=signal)))
    return [phrase for _, phrase in sorted(phrases)]


def _join_movement_phrases(phrases) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _exit_blockage_groups(exit_blockages) -> list:
    """One bullet per (exit road, full/partial) rather than one per blocked
    lane, so a two-lane incident reads '2 of 3 lanes fully blocked' instead of
    two contradictory single-lane bullets."""
    groups = {}
    for blockage in exit_blockages:
        edge = blockage["lane_id"].rsplit("_", 1)[0]
        groups.setdefault((edge, _is_full(blockage)), []).append(blockage)
    return list(groups.values())


def _exit_blockage_line(group: list) -> str:
    """One bullet for the blockages of one exit-road group (from
    SumoEnv.describe_exit_blockages). The distance quoted is the blockage
    nearest this intersection: it bounds the storage the road has left. For
    speed restrictions the distance is nominal -- the restriction covers the
    whole lane (same precedent as the approach section's segment)."""
    nearest = min(group, key=lambda b: b["distance_m"])
    template = (EXIT_BLOCKAGE_FULL_LINE if _is_full(nearest)
                else EXIT_BLOCKAGE_PARTIAL_LINE)
    phrases = _movement_phrases(nearest["feeding_movements"],
                                nearest["exit_direction"])
    return template.format(
        direction=nearest["exit_direction"],
        cause=_cause(nearest),
        blocked=len({b["lane_id"] for b in group}),
        total=nearest["lane_count"],
        distance=nearest["distance_m"],
        length=nearest["lane_length_m"],
        movements=_join_movement_phrases(phrases),
    )


def build_exit_blockage_section(exit_blockages) -> str:
    """Renders blockages on roads leaving the intersection as a prompt section;
    '' when there is nothing to report, so prompts without reportable exit
    blockages are unchanged. Roads onto which no listed movement releases
    vehicles are suppressed."""
    served = [b for b in (exit_blockages or [])
              if _movement_phrases(b["feeding_movements"], b["exit_direction"])]
    if not served:
        return ""
    lines = [EXIT_BLOCKAGE_HEADER, EXIT_BLOCKAGE_INTRO]
    lines += [_exit_blockage_line(g) for g in _exit_blockage_groups(served)]

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
