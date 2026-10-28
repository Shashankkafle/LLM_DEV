"""Lane state the evaluation logs need and no other run file holds, so that
network_metrics.csv and the arrival rates can be rebuilt offline.

lane_halting.csv: cycle_start, lane, halting
    Halted vehicles per signalized lane at the step each cycle closes; lanes
    with none are left out. Feeds side_street_halted (network_metrics.csv).
lane_entries.csv: t, lane, entered
    Vehicles entering each signalized lane per step; rows with none are left
    out. Feeds the realised arrival rates (tools/eval_metrics.py).

LiveNetworkState is the network_state callable EvaluationLogs reads during a
run; LoggedNetworkState rebuilds the same NetworkState from a run's files.
"""

import csv

from configurations import (
    EDGE_TIMESERIES_FILENAME,
    LANE_ENTRIES_FILENAME,
    LANE_HALTING_FILENAME,
)
from utils.evaluation_logs import NetworkState

LANE_HALTING_COLUMNS = ["cycle_start", "lane", "halting"]
LANE_ENTRIES_COLUMNS = ["t", "lane", "entered"]


class LaneStateLog:
    """Writes both files from an OversaturationIndices' lane snapshots."""

    def __init__(self, records_dir, oversaturation):
        self.oversaturation = oversaturation
        self._halting_file = open(records_dir / LANE_HALTING_FILENAME, "w", newline="")
        self._halting_writer = csv.writer(self._halting_file)
        self._halting_writer.writerow(LANE_HALTING_COLUMNS)
        self._entries_file = open(records_dir / LANE_ENTRIES_FILENAME, "w", newline="")
        self._entries_writer = csv.writer(self._entries_file)
        self._entries_writer.writerow(LANE_ENTRIES_COLUMNS)

    def record_step(self, t):
        """Call once per step, after OversaturationIndices.update(t)."""
        for lane, entered in sorted(self.oversaturation.lane_entries().items()):
            if entered > 0:
                self._entries_writer.writerow([t, lane, entered])

    def record_cycle_close(self, cycle_start):
        """Log and return {lane: halted} at the current step."""
        halting = self.oversaturation.lane_halting()
        for lane, count in sorted(halting.items()):
            if count > 0:
                self._halting_writer.writerow([cycle_start, lane, count])
        return halting

    def close(self):
        self._halting_file.close()
        self._entries_file.close()


class LiveNetworkState:
    """network_state for EvaluationLogs in a run: the MetricsRecorder's counts
    for the step just recorded, and the lane halting it logs."""

    def __init__(self, recorder, lane_state_log):
        self.recorder = recorder
        self.lane_state_log = lane_state_log

    def __call__(self, cycle_start, t):
        return NetworkState(self.recorder.last_vehicles_in_network,
                            self.recorder.last_halting_total,
                            self.lane_state_log.record_cycle_close(cycle_start))


# --- reading the logged state back -------------------------------------------------

def read_edge_totals(path):
    """t -> (vehicles, halting) summed over every edge column, the internal
    bucket included, so they equal the network-wide counts of that step."""
    totals = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        vehicle_columns = [i for i, name in enumerate(header) if name.startswith("vehicles:")]
        halting_columns = [i for i, name in enumerate(header) if name.startswith("halting:")]
        for row in reader:
            totals[int(float(row[0]))] = (sum(int(row[i]) for i in vehicle_columns),
                                          sum(int(row[i]) for i in halting_columns))
    return totals


def read_lane_halting(path):
    """cycle_start -> {lane: halted}."""
    by_cycle = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_cycle.setdefault(int(row["cycle_start"]), {})[row["lane"]] = int(row["halting"])
    return by_cycle


class LoggedNetworkState:
    """network_state for EvaluationLogs in replay, from edge_timeseries.csv
    and lane_halting.csv. A file the run did not write leaves its fields None."""

    def __init__(self, run_dir):
        edge_path = run_dir / EDGE_TIMESERIES_FILENAME
        halting_path = run_dir / LANE_HALTING_FILENAME
        self.edge_totals = read_edge_totals(edge_path) if edge_path.exists() else None
        self.lane_halting = read_lane_halting(halting_path) if halting_path.exists() else None

    def last_t(self):
        return max(self.edge_totals) if self.edge_totals else None

    def __call__(self, cycle_start, t):
        vehicles, halting = (self.edge_totals.get(t, (None, None))
                             if self.edge_totals is not None else (None, None))
        lane_halting = (self.lane_halting.get(cycle_start, {})
                        if self.lane_halting is not None else None)
        return NetworkState(vehicles, halting, lane_halting)
