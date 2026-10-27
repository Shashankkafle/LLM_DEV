"""Network regime from movement-flag events (decisions.md 2026/10/26).

States N (normal), L (loading), P (processing), R (recovery), recomputed once
per cycle from that cycle's flag events -- no timers, slopes or thresholds of
its own. Observation only: the regime is logged, never acted on.

The Diagnoser sits in the OversaturationIndices listener slot in front of
MovementFlags. It forwards every callback and reads flag on/off events by
diffing MovementFlags.active_flags() around each closed green, so the events
are exactly the transitions that set `active`.

A green belongs to the cycle containing its close time (t_close // cycle_s),
matching the offline reference regime_ref.py.
"""

import csv
from dataclasses import dataclass, field

from configurations import REGIME_FILENAME
from utils.movement_flags import TOSI_BRANCH

NORMAL = "N"
LOADING = "L"
PROCESSING = "P"
RECOVERY = "R"

REGIME_COLUMNS = ["cycle_start", "n_active", "entries", "exits", "regime"]


def next_regime(state, entries, exits, n_active):
    """One cycle of the regime state machine. The order of the checks matters:
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


def nominal_cycle_s(conf):
    """One full round of every phase at default timing (green + yellow + red)."""
    settings = conf["global_settings"]
    phase_s = (settings["default_green_duration"] + settings["yellow_duration"]
               + settings["red_duration"])
    return len(conf["phases"]) * phase_s


class Diagnoser:
    """Computes and logs the regime from a MovementFlags instance."""

    def __init__(self, movement_flags, cycle_s, records_dir):
        self.movement_flags = movement_flags
        self.cycle_s = cycle_s
        self.regime = NORMAL
        self._cycle_index = 0
        self._last_t = None
        self._events = CycleEvents()
        self._file = open(records_dir / REGIME_FILENAME, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(REGIME_COLUMNS)

    # --- OversaturationIndices listener ---------------------------------------

    def on_green_start(self, tracker):
        self.movement_flags.on_green_start(tracker)

    def on_step(self, tracker, step):
        self.movement_flags.on_step(tracker, step)

    def on_interval_closed(self, tracker, result):
        before = self.movement_flags.active_branches()
        self.movement_flags.on_interval_closed(tracker, result)
        self._events.add(before, self.movement_flags.active_branches())

    # --- cycle clock ------------------------------------------------------------

    def update(self, t):
        """Call once per step, after OversaturationIndices.update(t). Closes
        the current cycle on its last second, so a decision taken from the next
        second on sees that cycle's regime."""
        self._last_t = t
        while (t + 1) // self.cycle_s > self._cycle_index:
            self._close_cycle()

    def close(self):
        """Log the final partial cycle, if the run reached into it, and close
        the file."""
        if self._last_t is not None and self._last_t >= self._cycle_index * self.cycle_s:
            self._close_cycle()
        self._file.close()

    def _close_cycle(self):
        events = self._events
        n_active = len(self.movement_flags.active_flags())
        self.regime = next_regime(self.regime, events.entries, events.exits, n_active)
        self._writer.writerow([self._cycle_index * self.cycle_s, n_active,
                               events.entries, events.exits, self.regime])
        self._cycle_index += 1
        self._events = CycleEvents()
