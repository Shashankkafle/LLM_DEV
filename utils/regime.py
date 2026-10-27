"""Regime state machines over movement-flag events (decisions.md 2026/10/26,
2026/10/27).

States N (normal), L (loading), P (processing), R (recovery), recomputed once
per cycle from that cycle's flag events -- no timers, slopes or thresholds of
their own. Two machines: one network-wide (kept for comparison with the v2/v3
batches) and one per extent (utils/extents.py). utils/diagnoser.py drives
both. Observation only: the regime is logged, never acted on.
"""

from dataclasses import dataclass, field

from utils.movement_flags import TOSI_BRANCH

NORMAL = "N"
LOADING = "L"
PROCESSING = "P"
RECOVERY = "R"


def next_regime(state, entries, exits, n_active):
    """One cycle of the network state machine. The order of the checks matters:
    an entry and an exit in the same cycle gives L, and n reaching 0 in the
    exit cycle gives N in that same cycle."""
    if entries > 0:
        state = LOADING
    elif state == LOADING:
        state = PROCESSING
    if exits > 0 and entries == 0:
        state = RECOVERY
    if state == RECOVERY and n_active == 0:
        state = NORMAL
    return state


def next_extent_regime(state, entries, exits, n_conns, newborn):
    """One cycle of one extent's machine (decisions.md 2026/10/26 + 10/27).
    A newborn extent is a new root, so it loads. Every entry into an existing
    extent is chained: it widens the extent and never returns it to L, but in
    R it means queues are held again (-> P). A vanished extent is stepped with
    entries = n_conns = 0."""
    if newborn:
        return LOADING
    if state == LOADING:
        state = PROCESSING
    if entries > 0 and state == RECOVERY:
        state = PROCESSING
    if exits > 0 and entries == 0:
        state = RECOVERY
    if state == RECOVERY and n_conns == 0:
        state = NORMAL
    return state


@dataclass
class CycleEvents:
    """Flag on/off events of one cycle. entries/exits count movements; the lane
    sets feed the extent builder, which works on lanes."""
    entries: int = 0
    exits: int = 0
    tosi_entry_lanes: set = field(default_factory=set)
    exit_lanes: set = field(default_factory=set)

    def add(self, before, after):
        """before/after: active_branches() around one closed green."""
        entered = after.keys() - before.keys()
        exited = before.keys() - after.keys()
        self.entries += len(entered)
        self.exits += len(exited)
        self.tosi_entry_lanes |= {lane for lane, target in entered
                                  if after[(lane, target)] == TOSI_BRANCH}
        self.exit_lanes |= {lane for lane, _ in exited}
