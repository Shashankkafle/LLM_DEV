"""Per-movement oversaturation flags with hysteresis, on top of TOSI/SOSI.

A movement is one (lane, target_lane) connection. Each closed green of a lane
updates the flag of every movement leaving it (decisions.md 2026/10/26):
    u     = SOSI * green_s, the seconds of green lost to downstream blockage
    entry = (TOSI > 0 and ratio_start >= theta_in)       TOSI branch
            or u >= sosi_loss_s                           SOSI branch
            for n_in greens in a row
    clean = TOSI == 0 and u < sosi_loss_s and ratio_start < theta_out
    exit  = k_flag clean greens in a row; any other green resets the count
TOSI and ratio_start are lane-level, so movements of one lane share them; SOSI
is per movement.

branch is T when the flag entered on the TOSI branch, or met the TOSI-branch
condition on any later green while active; otherwise S. Only T flags feed
the extent builder.

A missing input (None; TOSI is None on a blocked lane) fails every clause, so
such a green neither enters nor counts as clean. The skipped column marks
those greens.

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

TOSI_BRANCH = "T"
SOSI_BRANCH = "S"

FLAG_COLUMNS = [
    "t_close", "tls", "lane", "target_lane", "link_indices", "tosi", "sosi",
    "ratio_end", "halted_count_end", "discharged", "blocked", "skipped",
    "entry_count", "clean_count", "active", "raised_at",
    "held_over", "ratio_start", "sosi_c1", "sosi_c2", "sosi_c3",
    "branch",
]


@dataclass
class FlagConfig:
    theta_in: float = 0.30
    theta_out: float = 0.20
    n_in: int = 1
    k_flag: int = 3
    # L: yellow + all-red, the green a movement can lose without being blocked.
    sosi_loss_s: float = 5.0

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
    branch: str = ""                  # TOSI_BRANCH or SOSI_BRANCH while active
    last_ratio: float | None = None   # ratio_start of last green


# --- pure update rule (no I/O) ----------------------------------------------------

def sosi_loss_s(sosi, green_s):
    return None if sosi is None else sosi * green_s


def has_missing_input(tosi, sosi, ratio_start):
    return tosi is None or sosi is None or ratio_start is None


def meets_tosi_entry(tosi, ratio_start, config):
    return (tosi is not None and ratio_start is not None
            and tosi > 0 and ratio_start >= config.theta_in)


def meets_sosi_entry(loss_s, config):
    return loss_s is not None and loss_s >= config.sosi_loss_s


def is_clean(tosi, loss_s, ratio_start, config):
    return (tosi is not None and loss_s is not None and ratio_start is not None
            and tosi == 0 and loss_s < config.sosi_loss_s
            and ratio_start < config.theta_out)


def entry_branch(tosi_entry, sosi_entry):
    """The branch a green would raise a flag on, or None if it meets neither."""
    if tosi_entry:
        return TOSI_BRANCH
    if sosi_entry:
        return SOSI_BRANCH
    return None


def clear_flag(state):
    state.active = False
    state.raised_at = None
    state.entry_count = 0
    state.clean_count = 0
    state.branch = ""


def update_inactive(state, branch, t_close, config):
    state.entry_count = state.entry_count + 1 if branch else 0
    if state.entry_count >= config.n_in:
        state.active = True
        state.raised_at = t_close
        state.clean_count = 0
        state.branch = branch


def update_active(state, clean, tosi_entry, config):
    state.clean_count = state.clean_count + 1 if clean else 0
    if state.clean_count >= config.k_flag:
        clear_flag(state)
    elif tosi_entry:
        state.branch = TOSI_BRANCH


def update_flag(state, tosi, sosi, green_s, ratio_start, t_close, config):
    """Apply one closed green to one movement's flag."""
    # OPEN: frozen flag on blocked lane (decisions.md)
    # A blocked lane has TOSI None, which fails every clause: an active flag
    # can never exit while its lane stays blocked.
    loss_s = sosi_loss_s(sosi, green_s)
    tosi_entry = meets_tosi_entry(tosi, ratio_start, config)
    if state.active:
        update_active(state, is_clean(tosi, loss_s, ratio_start, config),
                      tosi_entry, config)
    else:
        branch = entry_branch(tosi_entry, meets_sosi_entry(loss_s, config))
        update_inactive(state, branch, t_close, config)
    state.last_ratio = ratio_start


# --- listener ---------------------------------------------------------------------

def flag_row(result, target, link_indices, skipped, state):
    return [result.allred, result.tls, result.lane, target,
            ";".join(str(i) for i in link_indices),
            format_value(result.tosi), format_value(result.sosi[target]),
            format_value(result.ratio_end), result.halted_count_end,
            result.discharged, int(result.blocked_flag), int(skipped),
            state.entry_count, state.clean_count, int(state.active),
            format_value(state.raised_at),
            result.held_over, format_value(result.ratio_start),
            result.stop_halted_steps, result.no_exit_steps,
            result.target_halted_steps[target],
            state.branch]


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
            update_flag(state, result.tosi, sosi, result.green_s,
                        result.ratio_start, result.allred, self.config)
            skipped = has_missing_input(result.tosi, sosi, result.ratio_start)
            link_indices = tracker.target_link_indices.get(target, [])
            self._writer.writerow(flag_row(result, target, link_indices, skipped, state))

    def active_flags(self):
        return {movement for movement, state in self.flags.items() if state.active}

    def severity(self, lane, target):
        state = self.flags.get((lane, target))
        return state.last_ratio if state is not None else None

    def close(self):
        self._file.close()
