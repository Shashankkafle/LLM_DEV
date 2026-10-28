"""Evaluation logs written beside regime.csv and extents.csv, once per cycle
(LOGGING_CHECKLIST.md C and E).

ground_truth.csv: cycle_start, lane, consec_tosi_greens, gt_oversat
    The persistent-queue test the movement flags are scored against: a lane
    is oversaturated when its last PERSISTENT_GREENS greens all had TOSI > 0.
    It reads TOSI only -- never ratio_start or theta_in -- so it does not move
    when the flag thresholds do. A green with TOSI missing (blocked lane) is
    not TOSI > 0, so it ends the run.

network_metrics.csv: cycle_start, vehicles_in_system, total_halted,
                     total_overflow, discharged, side_street_halted
    vehicles_in_system and total_halted are network counts at the step the
    cycle closes. total_overflow sums held_over over every lane's last closed
    green, carried forward. held_over is a lane count shared by all
    connections of the lane, so each lane is counted once, not once per
    connection. discharged sums the greens that closed in the cycle.
    side_street_halted is total_halted minus the halted vehicles on lanes with
    an active flag (either branch) at cycle close.

Both carry each lane's state forward from its last green, like the extent
builder, so a lane with no green ending in a cycle keeps its last value.
"""

import csv
from dataclasses import dataclass

from configurations import GROUND_TRUTH_FILENAME, NETWORK_METRICS_FILENAME
from utils.oversaturation_indices import format_value

PERSISTENT_GREENS = 3

GROUND_TRUTH_COLUMNS = ["cycle_start", "lane", "consec_tosi_greens", "gt_oversat"]
NETWORK_METRICS_COLUMNS = ["cycle_start", "vehicles_in_system", "total_halted",
                           "total_overflow", "discharged", "side_street_halted"]


@dataclass
class NetworkState:
    """Network counts at the step a cycle closes; None when not available."""
    vehicles_in_system: int | None = None
    total_halted: int | None = None
    lane_halting: dict | None = None   # signalized lane -> halted vehicles


# --- pure rules (no I/O) ----------------------------------------------------------

def next_consec_tosi_greens(count, tosi):
    return count + 1 if tosi is not None and tosi > 0 else 0


def is_gt_oversaturated(consec_tosi_greens):
    return consec_tosi_greens >= PERSISTENT_GREENS


def side_street_halted(total_halted, lane_halting, flagged_lanes):
    if total_halted is None or lane_halting is None:
        return None
    return total_halted - sum(lane_halting.get(lane, 0) for lane in flagged_lanes)


class GroundTruth:
    """Consecutive TOSI > 0 greens per lane, for every lane that has closed a
    green so far."""

    def __init__(self):
        self.consec_tosi_greens = {}

    def on_interval_closed(self, result):
        count = self.consec_tosi_greens.get(result.lane, 0)
        self.consec_tosi_greens[result.lane] = next_consec_tosi_greens(count, result.tosi)

    def rows(self, cycle_start):
        return [[cycle_start, lane, count, int(is_gt_oversaturated(count))]
                for lane, count in sorted(self.consec_tosi_greens.items())]


class NetworkMetrics:
    def __init__(self):
        self.held_over = {}         # lane -> held_over of its last closed green
        self.cycle_discharged = 0

    def on_interval_closed(self, result):
        self.held_over[result.lane] = result.held_over
        self.cycle_discharged += result.discharged

    def close_cycle(self, cycle_start, state, flagged_lanes):
        """This cycle's row; starts the next cycle's discharge count."""
        row = [cycle_start, state.vehicles_in_system, state.total_halted,
               sum(self.held_over.values()), self.cycle_discharged,
               side_street_halted(state.total_halted, state.lane_halting, flagged_lanes)]
        self.cycle_discharged = 0
        return [format_value(value) for value in row]


# --- writer -----------------------------------------------------------------------

class EvaluationLogs:
    """Writes ground_truth.csv and network_metrics.csv. Driven by the
    Diagnoser: every closed green, then once per cycle close.

    network_state: callable(cycle_start, t) -> NetworkState at the step t on
    which the cycle closes (live counts in a run, the logged ones in replay).
    None leaves the network count columns empty.
    """

    def __init__(self, records_dir, network_state=None):
        self.network_state = network_state
        self.ground_truth = GroundTruth()
        self.network_metrics = NetworkMetrics()
        self._ground_truth_file = open(records_dir / GROUND_TRUTH_FILENAME, "w", newline="")
        self._ground_truth_writer = csv.writer(self._ground_truth_file)
        self._ground_truth_writer.writerow(GROUND_TRUTH_COLUMNS)
        self._network_file = open(records_dir / NETWORK_METRICS_FILENAME, "w", newline="")
        self._network_writer = csv.writer(self._network_file)
        self._network_writer.writerow(NETWORK_METRICS_COLUMNS)

    def on_interval_closed(self, result):
        self.ground_truth.on_interval_closed(result)
        self.network_metrics.on_interval_closed(result)

    def close_cycle(self, cycle_start, t, flagged_lanes):
        state = (self.network_state(cycle_start, t) if self.network_state is not None
                 else NetworkState())
        self._ground_truth_writer.writerows(self.ground_truth.rows(cycle_start))
        self._network_writer.writerow(
            self.network_metrics.close_cycle(cycle_start, state, flagged_lanes))

    def close(self):
        self._ground_truth_file.close()
        self._network_file.close()
