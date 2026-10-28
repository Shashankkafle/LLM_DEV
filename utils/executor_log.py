"""commands.csv and plan_changes.csv: what an executor issued and how the
signal plans changed (decisions.md 2026/09/28; LOGGING_CHECKLIST.md F).

Every run writes both headers, so baseline runs (FixedTime, MaxPressure: no
executor) sit beside mitigation runs with the same files, and the
plan-transition count of a baseline is the row count, 0. An executor calls
log_command once per command issued and log_plan_change once per change of a
signal's cycle or splits.
"""

import csv
import json

from configurations import COMMANDS_FILENAME, PLAN_CHANGES_FILENAME
from utils.oversaturation_indices import format_value

COMMAND_COLUMNS = ["t_issued", "tls", "primitive", "parameters", "watch_condition",
                   "effective_at", "expiry", "extent_id", "regime"]
PLAN_CHANGE_COLUMNS = ["t", "tls", "old_cycle", "new_cycle", "old_splits",
                       "new_splits", "transition_type"]


def json_cell(value):
    """Structured fields (parameters, watch condition, splits) as compact JSON."""
    return "" if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"))


class ExecutorLog:
    def __init__(self, records_dir):
        self._commands_file = open(records_dir / COMMANDS_FILENAME, "w", newline="")
        self._commands_writer = csv.writer(self._commands_file)
        self._commands_writer.writerow(COMMAND_COLUMNS)
        self._plans_file = open(records_dir / PLAN_CHANGES_FILENAME, "w", newline="")
        self._plans_writer = csv.writer(self._plans_file)
        self._plans_writer.writerow(PLAN_CHANGE_COLUMNS)

    def log_command(self, t_issued, tls, primitive, parameters, watch_condition,
                    effective_at, expiry, extent_id, regime):
        """parameters and watch_condition are dicts; expiry is a sim time."""
        self._commands_writer.writerow([
            t_issued, tls, primitive, json_cell(parameters), json_cell(watch_condition),
            effective_at, format_value(expiry), format_value(extent_id), regime])

    def log_plan_change(self, t, tls, old_cycle, new_cycle, old_splits, new_splits,
                        transition_type):
        """splits: {phase name: green seconds}."""
        self._plans_writer.writerow([
            t, tls, format_value(old_cycle), format_value(new_cycle),
            json_cell(old_splits), json_cell(new_splits), transition_type])

    def close(self):
        self._commands_file.close()
        self._plans_file.close()
