"""Pure helpers for reading a run's time series and measuring a demand surge
against its same-seed control: throughput inside the window, peak queue on an
approach, an oversaturation flag, recovery time and the deterministic
queueing estimate used to size a horizon. Used by tools/analyze_surge.py and
tests/test_surge_routes.py."""
import csv


def read_timeseries(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_edge_timeseries(path):
    """{t: row} with the numeric columns kept as floats; the row keeps its
    'vehicles:<edge>' / 'halting:<edge>' column names."""
    with open(path, newline="") as f:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return {row["t"]: row for row in rows}


def read_surge_timeseries(path):
    """{surge name: [rows sorted by t]}."""
    by_surge = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {"t": float(row["t"]), "departed_cum": float(row["departed_cum"]),
                     "arrived_cum": float(row["arrived_cum"]), "pending": float(row["pending"])}
            by_surge.setdefault(row["surge"], []).append(entry)
    return by_surge


def relative_rise(value, control):
    if control:
        return round((value - control) / control, 3)
    return None if value == 0 else float("inf")


def trailing_mean(values, window):
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
            for i in range(len(values))]


def excess_over_control(surge_rows, control_rows, smooth_s=300):
    """Surge-minus-control occupancy, smoothed with a trailing mean, so the
    surge's own footprint is visible under the shared background demand:
    where its peak lands and how much of it is left at the end of the run."""
    excess = [s["vehicles_in_network"] - c["vehicles_in_network"]
              for s, c in zip(surge_rows, control_rows)]
    smoothed = trailing_mean(excess, smooth_s)
    peak_index = max(range(len(smoothed)), key=smoothed.__getitem__)
    return {"excess_peak_smoothed": round(smoothed[peak_index], 1),
            "excess_peak_time_s": int(surge_rows[peak_index]["t"]),
            "excess_at_end_smoothed": round(smoothed[-1], 1)}


def series(rows_by_t, column):
    """(sorted times, values) for one column of a {t: row} table."""
    times = sorted(rows_by_t)
    return times, [rows_by_t[t][column] for t in times]


def value_at(times, values, t):
    """Value at the last sample time <= t (0 before the first sample)."""
    index = _last_index_at_or_before(times, t)
    return values[index] if index is not None else 0.0


def _last_index_at_or_before(times, t):
    best = None
    for index, time in enumerate(times):
        if time > t:
            break
        best = index
    return best


def cumulative_delta(rows, column, start, end):
    """Growth of a cumulative column over [start, end]."""
    by_t = {row["t"]: row for row in rows}
    times, values = series(by_t, column)
    return value_at(times, values, end) - value_at(times, values, start)


def throughput_in_window(surge_rows, control_rows, window):
    start, end = window
    surge = cumulative_delta(surge_rows, "arrived_cum", start, end)
    control = cumulative_delta(control_rows, "arrived_cum", start, end)
    return {"arrived_in_window": surge, "control_arrived_in_window": control,
            "extra_arrived_in_window": surge - control}


def peak_in_window(times, values, window):
    start, end = window
    inside = [(v, t) for t, v in zip(times, values) if start <= t <= end]
    if not inside:
        return {"peak": None, "peak_time_s": None}
    peak, at = max(inside)
    return {"peak": peak, "peak_time_s": int(at)}


def oversaturated(times, halting, pending, hold, smooth_s=300, growth_min=2.0):
    """The approach is oversaturated when its smoothed queue, or the surge's
    own insertion backlog, is still growing over the last third of the hold:
    the arrivals exceed what the intersection discharges."""
    hold_start, hold_end = hold
    checkpoint = hold_start + 2.0 * (hold_end - hold_start) / 3.0
    smoothed = trailing_mean(halting, smooth_s)
    queue_growth = value_at(times, smoothed, hold_end) - value_at(times, smoothed, checkpoint)
    pending_growth = value_at(times, pending, hold_end) - value_at(times, pending, checkpoint)
    queue_growing = queue_growth > growth_min
    pending_growing = pending_growth > growth_min
    return {"oversaturated": queue_growing or pending_growing,
            "queue_growing": queue_growing, "pending_growing": pending_growing,
            "queue_growth_last_third": round(queue_growth, 2),
            "pending_growth_last_third": round(pending_growth, 2)}


def recovery_time(times, halting, control_halting, window_end, smooth_s=300,
                  tolerance=0.10, slack=1.0, sustain_s=300):
    """Seconds after the window end until the smoothed approach queue is back
    within tolerance (plus an absolute slack, for near-empty approaches) of the
    control's, and stays there for sustain_s. None when it never recovers."""
    smoothed = trailing_mean(halting, smooth_s)
    control_smoothed = trailing_mean(control_halting, smooth_s)
    recovered_since = None
    for t, value, control in zip(times, smoothed, control_smoothed):
        if t < window_end:
            continue
        if value <= control * (1.0 + tolerance) + slack:
            if recovered_since is None:
                recovered_since = t
            if t - recovered_since >= sustain_s:
                return {"recovery_time_s": recovered_since - window_end,
                        "recovered_at_s": recovered_since}
        else:
            recovered_since = None
    return {"recovery_time_s": None, "recovered_at_s": None}


def max_in_window(rows, column, window):
    start, end = window
    inside = [row[column] for row in rows if start <= row["t"] <= end]
    return max(inside) if inside else None


def queueing_estimate(capacity_veh_h, base_rate_veh_h, surge_rate_veh_h, duration_s):
    """Deterministic queueing: backlog built while the surge exceeds capacity
    and the time to clear it at the spare capacity left after the surge."""
    if capacity_veh_h <= base_rate_veh_h:
        raise ValueError("baseline demand is at or above capacity: no spare capacity to recover")
    backlog = max(surge_rate_veh_h - capacity_veh_h, 0.0) * duration_s / 3600.0
    clear_s = backlog / (capacity_veh_h - base_rate_veh_h) * 3600.0
    return {"backlog_vehicles": round(backlog, 1),
            "clearance_time_s": round(clear_s, 1),
            "suggested_recovery_window_s": round(2 * clear_s, 1)}
