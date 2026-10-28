"""Per-cycle diagnosis on top of the movement flags: the network regime
(regime.csv) and the extents with their own regimes (extents.csv).

The Diagnoser sits in the OversaturationIndices listener slot in front of
MovementFlags. It forwards every callback and reads flag on/off events by
diffing MovementFlags.active_branches() around each closed green, so the
events are exactly the transitions that set `active`.

A green belongs to the cycle containing its close time (t_close // cycle_s).
Extents are built from the flag state as it stands when the cycle closes,
carried forward from each movement's last green, not from the greens that
closed inside the cycle.

An optional EvaluationLogs gets the same closed greens and cycle closes, and
writes ground_truth.csv and network_metrics.csv.
"""

import csv

from configurations import EXTENTS_FILENAME, REGIME_FILENAME
from utils.extents import ExtentTracker
from utils.objectives import loading_objective, processing_objective, recovery_objective
from utils.oversaturation_indices import format_value
from utils.regime import NORMAL, CycleEvents, next_regime

DEFAULT_CYCLE_S = 140

REGIME_COLUMNS = ["cycle_start", "n_active", "entries", "exits", "regime"]
EXTENT_COLUMNS = ["cycle_start", "extent_id", "members", "n_lanes", "n_conns",
                  "E", "X", "regime", "newborn", "vanished",
                  "root_lane", "oldest_lane", "n_intersections",
                  "obj_loading", "obj_processing", "obj_recovery"]


def extent_csv_row(row, lane_tls):
    """lane_tls: lane -> the signal controlling it, for n_intersections."""
    n_intersections = len({lane_tls[lane] for lane in row.members})
    return [row.cycle_start, row.extent_id, "+".join(sorted(row.members)),
            len(row.members), row.n_conns, row.entries, row.exits, row.regime,
            int(row.newborn), int(row.vanished),
            row.root_lane, row.oldest_lane, n_intersections,
            format_value(loading_objective(row)), format_value(processing_objective(row)),
            format_value(recovery_objective(row))]


def flagged_lanes(active_flags):
    return {lane for lane, _ in active_flags}


class Diagnoser:
    """Computes and logs the network regime and the extents from a
    MovementFlags instance."""

    def __init__(self, movement_flags, cycle_s, records_dir, evaluation_logs=None):
        self.movement_flags = movement_flags
        self.cycle_s = cycle_s
        self.evaluation_logs = evaluation_logs
        self.regime = NORMAL
        self.extents = ExtentTracker()
        self.lane_tls = {}
        # Optional callable(cycle_start, network_row, extent_rows), called after
        # every cycle is logged; lets a watcher print without reading the CSVs.
        self.cycle_listener = None
        self._cycle_index = 0
        self._last_t = None
        self._events = CycleEvents()
        self._regime_file = open(records_dir / REGIME_FILENAME, "w", newline="")
        self._regime_writer = csv.writer(self._regime_file)
        self._regime_writer.writerow(REGIME_COLUMNS)
        self._extent_file = open(records_dir / EXTENTS_FILENAME, "w", newline="")
        self._extent_writer = csv.writer(self._extent_file)
        self._extent_writer.writerow(EXTENT_COLUMNS)

    # --- OversaturationIndices listener ---------------------------------------

    def on_green_start(self, tracker):
        self.movement_flags.on_green_start(tracker)

    def on_step(self, tracker, step):
        self.movement_flags.on_step(tracker, step)

    def on_interval_closed(self, tracker, result):
        self.lane_tls[result.lane] = result.tls
        before = self.movement_flags.active_branches()
        self.movement_flags.on_interval_closed(tracker, result)
        self._events.add(before, self.movement_flags.active_branches())
        if self.evaluation_logs is not None:
            self.evaluation_logs.on_interval_closed(result)

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
        the files."""
        if self._last_t is not None and self._last_t >= self._cycle_index * self.cycle_s:
            self._close_cycle()
        self._regime_file.close()
        self._extent_file.close()
        if self.evaluation_logs is not None:
            self.evaluation_logs.close()

    def _close_cycle(self):
        cycle_start = self._cycle_index * self.cycle_s
        events = self._events
        network_row = self._log_network_regime(cycle_start, events)
        rows = self.extents.close_cycle(cycle_start, self.movement_flags.tosi_flags(), events)
        for row in rows:
            self._extent_writer.writerow(extent_csv_row(row, self.lane_tls))
        if self.evaluation_logs is not None:
            self.evaluation_logs.close_cycle(
                cycle_start, self._last_t, flagged_lanes(self.movement_flags.active_flags()))
        if self.cycle_listener is not None:
            self.cycle_listener(cycle_start, network_row, rows)
        self._cycle_index += 1
        self._events = CycleEvents()

    def _log_network_regime(self, cycle_start, events):
        n_active = len(self.movement_flags.active_flags())
        self.regime = next_regime(self.regime, events.entries, events.exits, n_active)
        network_row = dict(zip(REGIME_COLUMNS, [cycle_start, n_active, events.entries,
                                                events.exits, self.regime]))
        self._regime_writer.writerow(network_row.values())
        return network_row
