"""Per-movement oversaturation flags with hysteresis, on top of TOSI/SOSI.

A movement is one (lane, target_lane) connection. Each closed green of a lane
updates the flag of every movement leaving it:
    entry = (TOSI > 0 or SOSI > 0) and ratio >= theta_in, for n_in greens in a row
    exit  = TOSI == 0 and SOSI == 0 and ratio < theta_out, for k_flag greens in a row
TOSI and the queue/storage ratio are lane-level, so movements of one lane
share them; SOSI is per movement. Clause form per NCHRP pp. 112-114.

A blocked lane (an active blockage on it during the green) forces its flags
off and resets their counters. A green with TOSI, SOSI or the ratio missing is
skipped: no counter or state changes.

Plugged in as the OversaturationIndices listener; only on_interval_closed does
work. Later layers read active_flags() and severity() and nothing else.
"""

import csv
from dataclasses import dataclass

from configurations import MOVEMENT_FLAGS_FILENAME
from utils.oversaturation_indices import format_value

MIN_HYSTERESIS_GAP = 0.10
# Absorbs float error so e.g. theta_in=0.3, theta_out=0.2 is accepted.
GAP_TOLERANCE = 1e-9

FLAG_COLUMNS = [
    "t_close", "tls", "lane", "target_lane", "link_indices", "tosi", "sosi",
    "ratio_end", "halted_count_end", "discharged", "blocked", "skipped",
    "entry_count", "clean_count", "active", "raised_at",
]


@dataclass
class FlagConfig:
    theta_in: float = 0.5
    theta_out: float = 0.35
    n_in: int = 1
    k_flag: int = 3

    def __post_init__(self):
        if self.theta_out > self.theta_in - MIN_HYSTERESIS_GAP + GAP_TOLERANCE:
            raise ValueError(f"theta_out ({self.theta_out}) must be <= theta_in "
                             f"({self.theta_in}) - {MIN_HYSTERESIS_GAP}")


@dataclass
class FlagState:
    active: bool = False
    raised_at: int | None = None      # t_close of the green that raised it
    entry_count: int = 0              # consecutive greens meeting entry condition
    clean_count: int = 0              # consecutive clean greens while active
    last_ratio: float | None = None   # ratio_end of last evaluated green


# --- pure update rule (no I/O) ----------------------------------------------------

def meets_entry(tosi, sosi, ratio, config):
    return (tosi > 0 or sosi > 0) and ratio >= config.theta_in


def is_clean(tosi, sosi, ratio, config):
    return tosi == 0 and sosi == 0 and ratio < config.theta_out


def clear_flag(state):
    state.active = False
    state.raised_at = None
    state.entry_count = 0
    state.clean_count = 0


def update_inactive(state, entering, t_close, config):
    state.entry_count = state.entry_count + 1 if entering else 0
    if state.entry_count >= config.n_in:
        state.active = True
        state.raised_at = t_close
        state.clean_count = 0


def update_active(state, clean, config):
    state.clean_count = state.clean_count + 1 if clean else 0
    if state.clean_count >= config.k_flag:
        clear_flag(state)


def update_flag(state, tosi, sosi, ratio, blocked, t_close, config):
    """Apply one closed green to one movement's flag. Returns True when the
    green was skipped for a missing input."""
    if blocked:
        clear_flag(state)
        return False
    if tosi is None or sosi is None or ratio is None:
        return True
    if state.active:
        update_active(state, is_clean(tosi, sosi, ratio, config), config)
    else:
        update_inactive(state, meets_entry(tosi, sosi, ratio, config), t_close, config)
    state.last_ratio = ratio
    return False


# --- listener ---------------------------------------------------------------------

def flag_row(result, target, link_indices, skipped, state):
    return [result.allred, result.tls, result.lane, target,
            ";".join(str(i) for i in link_indices),
            format_value(result.tosi), format_value(result.sosi[target]),
            format_value(result.ratio_end), result.halted_count_end,
            result.discharged, int(result.blocked_flag), int(skipped),
            state.entry_count, state.clean_count, int(state.active),
            format_value(state.raised_at)]


class MovementFlags:
    """OversaturationIndices listener holding one FlagState per movement.

    A green closes at its all-red onset, so t_close is IntervalResult.allred.
    """

    def __init__(self, config, records_dir):
        self.config = config
        self.flags = {}
        self._file = open(records_dir / MOVEMENT_FLAGS_FILENAME, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FLAG_COLUMNS)

    def on_green_start(self, tracker):
        pass

    def on_step(self, tracker, step):
        pass

    def on_interval_closed(self, tracker, result):
        for target, sosi in result.sosi.items():
            state = self.flags.setdefault((result.lane, target), FlagState())
            skipped = update_flag(state, result.tosi, sosi, result.ratio_end,
                                  result.blocked_flag, result.allred, self.config)
            link_indices = tracker.target_link_indices.get(target, [])
            self._writer.writerow(flag_row(result, target, link_indices, skipped, state))

    def active_flags(self):
        return {movement for movement, state in self.flags.items() if state.active}

    def severity(self, lane, target):
        state = self.flags.get((lane, target))
        return state.last_ratio if state is not None else None

    def close(self):
        self._file.close()
