"""Per-extent objective values for the extents.csv obj_* columns.

Stubs: each returns None, which extents.csv writes as an empty cell, until the
executor exists and the formulas below are implemented. Each takes the
extent's ExtentRow for the cycle; the inputs the formulas need beyond it
(per-movement overflow, spare storage, queue ratios) are to be passed in when
they are implemented.
"""


def loading_objective(extent_row):
    """decisions.md 2026/09/24, Loading objective:

        sum over the extent's flagged movements of end-of-green overflow,
        weighted by 1 / own spare storage (clamped),
        zeroed for movements whose exit link has less than theta spare storage.

    theta and the clamp epsilon are calibrated with the regime thresholds.
    Not implemented: returns None.
    """
    return None


def processing_objective(extent_row):
    """decisions.md 2026/09/25, Processing objective:

        sum over lanes in the extent of max(0, queue_ratio - theta),

    theta shared with the loading zero-out rule; max-ratio logged as a
    comparison column. Provisional until the shadow log is plotted.
    Not implemented: returns None.
    """
    return None


def recovery_objective(extent_row):
    """decisions.md 2026/09/25, Recovery objective:

        per-cycle drop in total end-of-green overflow on the extent's flagged
        movements (overflow last cycle - overflow this cycle),

    halting-count proxy in the controller; ID-set intersection logged as
    ground truth; discharged-vehicle count logged as comparison column.
    Provisional until the shadow log is plotted. Not implemented: returns None.
    """
    return None
