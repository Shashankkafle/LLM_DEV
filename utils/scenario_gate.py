"""Scenario gate (LOGGING_CHECKLIST.md D): does a run load its scenario's
target hard enough to count as a level of the intensity ladder?

passed when any feeder lane of the target has ratio_start >= theta_in on
PERSISTENT_GREENS consecutive greens of that lane. Feeder lanes, from the
net's connections:
    surge     -- the lanes of each surge route's entry edge
    blockage  -- the lanes with a connection into a blocked lane
first_t is the t_close of the green that completes the first such run. A
green with ratio_start missing ends a run. A run with neither a surge nor a
blockage has no target and does not pass.
"""

from utils.evaluation_logs import PERSISTENT_GREENS
from utils.run_logs import blocked_lanes, load_network, read_lane_greens, read_manifest, theta_in


def gate_rule(threshold):
    return (f"feeder lane ratio_start >= {threshold} on >= {PERSISTENT_GREENS} "
            f"consecutive greens")


def target_feeder_lanes(manifest, run_dir, network):
    lanes = set()
    for surge in (manifest.get("surge") or {}).get("surges", []):
        lanes |= set(network.edge_lanes.get(surge["route"][0], []))
    for lane in blocked_lanes(run_dir):
        lanes |= set(network.feeders(lane))
    return lanes


def first_persistent_green(greens, lanes, threshold):
    """t_close of the earliest green completing PERSISTENT_GREENS consecutive
    greens of one lane at ratio_start >= threshold, or None."""
    run_length = {}
    for green in sorted(greens, key=lambda g: (g.t_close, g.lane)):
        if green.lane not in lanes:
            continue
        loaded = green.ratio_start is not None and green.ratio_start >= threshold
        run_length[green.lane] = run_length.get(green.lane, 0) + 1 if loaded else 0
        if run_length[green.lane] >= PERSISTENT_GREENS:
            return green.t_close
    return None


def scenario_gate(run_dir):
    """(passed, first_t) for the run in run_dir. Raises if the run's net
    file cannot be found."""
    manifest = read_manifest(run_dir)
    network = load_network(manifest)
    if network is None:
        raise FileNotFoundError(f"{run_dir}: the run's net file is not on this machine")
    lanes = target_feeder_lanes(manifest, run_dir, network)
    first_t = first_persistent_green(read_lane_greens(run_dir), lanes, theta_in(manifest))
    return first_t is not None, first_t


def scenario_gate_record(run_dir):
    """The manifest's scenario_gate block: {rule, passed, first_t}."""
    passed, first_t = scenario_gate(run_dir)
    return {"rule": gate_rule(theta_in(read_manifest(run_dir))),
            "passed": passed, "first_t": first_t}
