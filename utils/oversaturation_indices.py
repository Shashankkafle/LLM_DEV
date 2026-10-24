"""Per-cycle oversaturation indices: TOSI per lane and SOSI per movement.

TOSI (temporal oversaturation): of the vehicles standing on a lane when its
green came on (H), how many were still on the lane when it went red (R):
    TOSI = |H & R| * headway / green_duration, clamped to [0, 1].
SOSI (spatial oversaturation): the share of a movement's green seconds (the
first SOSI_SKIP_STEPS excluded) lost to downstream spillback, i.e. seconds in
which (a) a vehicle is halted at the stop line, (b) nobody left the lane, and
(c) a vehicle is halted at the entry of the movement's target lane.

A pure observer: it reads TraCI subscriptions and never sets anything, so it
cannot change a control decision. Definitions and the timing convention are
documented in METRICS.md ("Oversaturation indices").

Timing: update(t) runs after simulationStep, so the signal state it reads is
the one that governed the second t-1 -> t. Signal boundaries are therefore
labelled tau = t-1, and the snapshots that define them (H, R, the queue
ratios) are the vehicle snapshot taken at tau -- the previous update's. Taking
H one second late would miss the queue head, which is already moving.
"""

import csv
import math
from dataclasses import dataclass, field

import traci
from traci import constants as tc

from configurations import (
    OBSTACLE_VEHICLE_PREFIX,
    OVERSATURATION_INDICES_FILENAME,
    OVERSATURATION_SHADOW_FILENAME,
)

HEADWAY_S = 2.0
HALT_SPEED_MPS = 0.1
STOP_LINE_WINDOW_M = 15.0
SOSI_SKIP_STEPS = 3
METERS_PER_STORED_VEHICLE = 7.5

RED = "r"
YELLOW = "y"
GREEN_CHARS = ("G", "g")

VEHICLE_VARS = [tc.VAR_SPEED, tc.VAR_LANEPOSITION]

INDICES_COLUMNS = [
    "t_green_start", "t_green_end", "t_allred", "tls", "lane", "target_lane",
    "green_s", "tosi", "discharged", "sosi", "ratio_start", "ratio_end",
    "blocked_flag",
]
SHADOW_COLUMNS = ["t", "lane", "halted_count", "exit_count"]


# --- pure rules (no TraCI) ----------------------------------------------------

def is_obstacle(vehicle_id):
    return str(vehicle_id).startswith(OBSTACLE_VEHICLE_PREFIX)


def clamp_unit(value):
    return min(max(value, 0.0), 1.0)


def compute_tosi(halted_at_start, on_lane_at_allred, green_s, headway=HEADWAY_S):
    """Returns (tosi, discharged) for one green of one lane."""
    held_over = len(set(halted_at_start) & set(on_lane_at_allred))
    discharged = len(set(halted_at_start) - set(on_lane_at_allred))
    tosi = clamp_unit(held_over * headway / green_s) if green_s > 0 else None
    return tosi, discharged


def exited_ids(prev_ids, curr_ids):
    """Vehicles that left the lane this step. Set-based on purpose: {1,2} ->
    {2,3} is one exit even though the count stayed flat."""
    return set(prev_ids) - set(curr_ids)


def has_halted_near_stop(vehicles, lane_length, window=STOP_LINE_WINDOW_M):
    """SOSI rule (a). vehicles: iterable of (id, speed, lane_position)."""
    return any(speed < HALT_SPEED_MPS and position >= lane_length - window
               for _, speed, position in vehicles)


def has_halted_near_entry(vehicles, window=STOP_LINE_WINDOW_M):
    """SOSI rule (c), evaluated on the movement's target lane."""
    return any(speed < HALT_SPEED_MPS and position <= window
               for _, speed, position in vehicles)


def is_spillback_blocked(halted_near_stop, exit_count, target_halted_near_entry):
    return halted_near_stop and exit_count == 0 and target_halted_near_entry


def compute_sosi(blocked_steps, green_s, skip=SOSI_SKIP_STEPS):
    if green_s <= skip:
        return None
    return clamp_unit(blocked_steps / (green_s - skip))


def storage_capacity(lane_length):
    return math.floor(lane_length / METERS_PER_STORED_VEHICLE)


def queue_storage_ratio(halted_count, capacity):
    return halted_count / capacity if capacity > 0 else None


def lane_signal(state, link_indices):
    """Collapse a lane's link characters into one: G/g if any link is green
    (keeping protected vs permissive for display), else y, else r."""
    chars = {state[i] for i in link_indices}
    for char in ("G", "g", YELLOW):
        if char in chars:
            return char
    return RED


def is_green(signal):
    return signal in GREEN_CHARS


@dataclass(frozen=True)
class LaneSnapshot:
    """One lane at one instant, obstacle vehicles already removed."""
    ids: frozenset
    halted_ids: frozenset
    halted_near_stop: bool
    halted_near_entry: bool


EMPTY_SNAPSHOT = LaneSnapshot(frozenset(), frozenset(), False, False)


def build_lane_snapshot(vehicles, lane_length):
    """vehicles: iterable of (id, speed, lane_position). This is the single
    place the obstacle is dropped, so H, R, (a) and (c) all exclude it."""
    traffic = [v for v in vehicles if not is_obstacle(v[0])]
    return LaneSnapshot(
        ids=frozenset(v[0] for v in traffic),
        halted_ids=frozenset(v[0] for v in traffic if v[1] < HALT_SPEED_MPS),
        halted_near_stop=has_halted_near_stop(traffic, lane_length),
        halted_near_entry=has_halted_near_entry(traffic),
    )


# --- per-lane state machine ----------------------------------------------------

@dataclass
class TargetInputs:
    target_lane: str
    target_halted_near_entry: bool
    blocked_step: object  # True/False on a counted green second, else None


@dataclass
class StepInputs:
    """The raw rule inputs for one lane at snapshot time t."""
    t: int
    lane: str
    signal: str
    vehicle_count: int
    halted_count: int
    halted_near_stop: bool
    exit_count: int
    targets: list


@dataclass
class IntervalResult:
    tls: str
    lane: str
    green_start: int
    green_end: int
    allred: int
    halted_at_start: frozenset
    on_lane_at_allred: frozenset
    tosi: object
    discharged: int
    blocked_flag: bool
    capacity: int
    halted_count_start: int
    halted_count_end: int
    ratio_start: object
    ratio_end: object
    blocked_steps: dict
    sosi: dict

    @property
    def green_s(self):
        return self.green_end - self.green_start

    @property
    def held_over(self):
        return len(self.halted_at_start & self.on_lane_at_allred)


@dataclass
class OpenGreen:
    green_start: int
    halted_at_start: frozenset
    blocked_steps: dict
    blocked_flag: bool
    green_end: object = None


@dataclass
class Observation:
    step: StepInputs
    green_started: bool = False
    closed: object = None  # IntervalResult when an interval ended this step


class LaneGreenTracker:
    """Follows one controlled lane through green -> (yellow) -> red and turns
    each completed green into an IntervalResult. Fed synthetic snapshots in
    tests, live ones by OversaturationIndices."""

    def __init__(self, tls, lane, lane_length, link_indices, targets,
                 target_link_indices=None):
        self.tls = tls
        self.lane = lane
        self.lane_length = lane_length
        self.link_indices = link_indices
        self.targets = targets
        # target lane -> every signal link index that reaches it from this lane
        self.target_link_indices = target_link_indices or {}
        self.capacity = storage_capacity(lane_length)
        self.open_green = None
        self.completed_greens = 0

    def observe(self, tau, signal, prev, curr, target_snapshots, lane_blocked):
        """Advance one step. tau labels the second whose signal is `signal`;
        prev is the snapshot at tau, curr the one at tau+1."""
        green = is_green(signal)
        observation = Observation(step=None)
        if self.open_green is not None:
            self.open_green.blocked_flag = self.open_green.blocked_flag or lane_blocked
            if self.open_green.green_end is None and not green:
                self.open_green.green_end = tau
            # Red ends the interval. A green arriving before any red (G -> y
            # -> G, only seen under the net's static program) ends it too.
            if self.open_green.green_end is not None and (green or signal == RED):
                observation.closed = self._close(tau, prev)
        if green and self.open_green is None:
            self.open_green = self._start(tau, prev, lane_blocked)
            observation.green_started = True
        observation.step = self._evaluate_step(tau, signal, prev, curr, target_snapshots)
        return observation

    def _start(self, tau, prev, lane_blocked):
        return OpenGreen(green_start=tau, halted_at_start=prev.halted_ids,
                         blocked_steps={target: 0 for target in self.targets},
                         blocked_flag=lane_blocked)

    def _is_counted_green_second(self, tau):
        green = self.open_green
        return (green is not None and green.green_end is None
                and tau - green.green_start >= SOSI_SKIP_STEPS)

    def _evaluate_step(self, tau, signal, prev, curr, target_snapshots):
        exit_count = len(exited_ids(prev.ids, curr.ids))
        counted = self._is_counted_green_second(tau)
        targets = []
        for target in self.targets:
            near_entry = target_snapshots.get(target, EMPTY_SNAPSHOT).halted_near_entry
            blocked = is_spillback_blocked(curr.halted_near_stop, exit_count, near_entry)
            if counted and blocked:
                self.open_green.blocked_steps[target] += 1
            targets.append(TargetInputs(target, near_entry, blocked if counted else None))
        return StepInputs(t=tau + 1, lane=self.lane, signal=signal,
                          vehicle_count=len(curr.ids), halted_count=len(curr.halted_ids),
                          halted_near_stop=curr.halted_near_stop, exit_count=exit_count,
                          targets=targets)

    def _close(self, allred, prev):
        green = self.open_green
        self.open_green = None
        self.completed_greens += 1
        green_s = green.green_end - green.green_start
        tosi, discharged = compute_tosi(green.halted_at_start, prev.ids, green_s)
        if green.blocked_flag:
            tosi = None
        return IntervalResult(
            tls=self.tls, lane=self.lane,
            green_start=green.green_start, green_end=green.green_end, allred=allred,
            halted_at_start=green.halted_at_start, on_lane_at_allred=prev.ids,
            tosi=tosi, discharged=discharged, blocked_flag=green.blocked_flag,
            capacity=self.capacity,
            halted_count_start=len(green.halted_at_start),
            halted_count_end=len(prev.halted_ids),
            ratio_start=queue_storage_ratio(len(green.halted_at_start), self.capacity),
            ratio_end=queue_storage_ratio(len(prev.halted_ids), self.capacity),
            blocked_steps=dict(green.blocked_steps),
            sosi={target: compute_sosi(steps, green_s)
                  for target, steps in green.blocked_steps.items()},
        )


# --- CSV formatting -------------------------------------------------------------

def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def interval_rows(result):
    """One CSV row per movement (lane -> target lane) of a completed green."""
    for target in result.blocked_steps:
        yield [result.green_start, result.green_end, result.allred, result.tls,
               result.lane, target, result.green_s, format_value(result.tosi),
               result.discharged, format_value(result.sosi[target]),
               format_value(result.ratio_start), format_value(result.ratio_end),
               int(result.blocked_flag)]


# --- TraCI wrapper ---------------------------------------------------------------

def read_controlled_movements():
    """(tls, lane, target_lane, link_index) for every signalized connection.
    Lanes of unsignalized entry/exit junctions never appear here."""
    movements = []
    for tls in traci.trafficlight.getIDList():
        for link_index, links in enumerate(traci.trafficlight.getControlledLinks(tls)):
            for lane, target_lane, _via in links:
                movements.append((tls, lane, target_lane, link_index))
    return movements


def check_against_lane_links(trackers):
    """Cross-check the signal-derived targets against lane.getLinks."""
    for tracker in trackers:
        lane_targets = {link[0] for link in traci.lane.getLinks(tracker.lane)}
        if lane_targets != set(tracker.targets):
            raise RuntimeError(f"{tracker.lane}: controlled-link targets "
                               f"{sorted(tracker.targets)} != lane.getLinks "
                               f"{sorted(lane_targets)}")


def build_trackers(movements):
    by_lane = {}
    for tls, lane, target_lane, link_index in movements:
        entry = by_lane.setdefault(lane, {"tls": tls, "links": [], "targets": [],
                                          "target_links": {}})
        entry["links"].append(link_index)
        if target_lane not in entry["targets"]:
            entry["targets"].append(target_lane)
        entry["target_links"].setdefault(target_lane, []).append(link_index)
    return [LaneGreenTracker(entry["tls"], lane, traci.lane.getLength(lane),
                             entry["links"], entry["targets"], entry["target_links"])
            for lane, entry in by_lane.items()]


def lanes_by_tls(trackers):
    tls_lanes = {}
    for tracker in trackers:
        tls_lanes.setdefault(tracker.tls, []).append(tracker.lane)
    return tls_lanes


class OversaturationIndices:
    """Called once per simulation step via update(t), after env.step().

    listener (optional) gets on_green_start(tracker), on_step(tracker,
    StepInputs) and on_interval_closed(tracker, IntervalResult); the GUI
    inspector uses it so the arithmetic lives only here.
    """

    def __init__(self, records_dir, blockage_manager=None, shadow=False, listener=None):
        self.blockage_manager = blockage_manager
        self.listener = listener
        self.trackers = build_trackers(read_controlled_movements())
        check_against_lane_links(self.trackers)
        self.tls_ids = sorted({tracker.tls for tracker in self.trackers})
        self.tls_lanes = lanes_by_tls(self.trackers)
        self.lane_lengths = self._observed_lane_lengths()
        self._prev = {lane: EMPTY_SNAPSHOT for lane in self.lane_lengths}
        self._subscribe_lanes_and_signals()
        self._indices_file = open(records_dir / OVERSATURATION_INDICES_FILENAME, "w", newline="")
        self._indices_writer = csv.writer(self._indices_file)
        self._indices_writer.writerow(INDICES_COLUMNS)
        self._shadow_file = None
        if shadow:
            self._shadow_file = open(records_dir / OVERSATURATION_SHADOW_FILENAME, "w", newline="")
            self._shadow_writer = csv.writer(self._shadow_file)
            self._shadow_writer.writerow(SHADOW_COLUMNS)

    def _observed_lane_lengths(self):
        """Source lanes plus every target lane (rule (c) reads targets, some
        of which are exit lanes that are not tracked themselves)."""
        lanes = {tracker.lane: tracker.lane_length for tracker in self.trackers}
        for tracker in self.trackers:
            for target in tracker.targets:
                if target not in lanes:
                    lanes[target] = traci.lane.getLength(target)
        return lanes

    def _subscribe_lanes_and_signals(self):
        # Gotcha: a later subscribe() on the same lane or TLS by other code
        # would replace these variable lists. Nothing else subscribes today.
        for lane in self.lane_lengths:
            traci.lane.subscribe(lane, [tc.LAST_STEP_VEHICLE_ID_LIST])
        for tls in self.tls_ids:
            traci.trafficlight.subscribe(tls, [tc.TL_RED_YELLOW_GREEN_STATE])

    def _vehicle_states(self, lane_vehicle_ids):
        """Speed and lane position per vehicle, from vehicle subscriptions.
        A vehicle is subscribed the first time it is seen on an observed
        lane; SUMO drops the subscription when it leaves the network."""
        states = traci.vehicle.getAllSubscriptionResults()
        for ids in lane_vehicle_ids.values():
            for vehicle_id in ids:
                if vehicle_id not in states:
                    traci.vehicle.subscribe(vehicle_id, VEHICLE_VARS)
        return traci.vehicle.getAllSubscriptionResults()

    def _snapshots(self):
        lane_results = traci.lane.getAllSubscriptionResults()
        # Obstacles are dropped before subscribing: BlockageManager removes
        # them with vehicle.remove, after which SUMO prints a "not known"
        # error for any subscription still attached.
        lane_vehicle_ids = {lane: [v for v in lane_results[lane][tc.LAST_STEP_VEHICLE_ID_LIST]
                                   if not is_obstacle(v)]
                            for lane in self.lane_lengths}
        states = self._vehicle_states(lane_vehicle_ids)
        snapshots = {}
        for lane, ids in lane_vehicle_ids.items():
            vehicles = [(v, states[v][tc.VAR_SPEED], states[v][tc.VAR_LANEPOSITION])
                        for v in ids]
            snapshots[lane] = build_lane_snapshot(vehicles, self.lane_lengths[lane])
        return snapshots

    def _blocked_lanes(self):
        if self.blockage_manager is None:
            return set()
        return set(self.blockage_manager.get_blocked_lane_ids())

    def update(self, t):
        tau = t - 1
        curr = self._snapshots()
        signal_results = traci.trafficlight.getAllSubscriptionResults()
        blocked_lanes = self._blocked_lanes()
        for tracker in self.trackers:
            state = signal_results[tracker.tls][tc.TL_RED_YELLOW_GREEN_STATE]
            observation = tracker.observe(
                tau, lane_signal(state, tracker.link_indices),
                self._prev[tracker.lane], curr[tracker.lane], curr,
                tracker.lane in blocked_lanes)
            self._record(tracker, observation)
        self._prev = curr

    def _record(self, tracker, observation):
        if observation.closed is not None:
            self._indices_writer.writerows(interval_rows(observation.closed))
        if self._shadow_file is not None:
            step = observation.step
            self._shadow_writer.writerow([step.t, step.lane, step.halted_count, step.exit_count])
        if self.listener is None:
            return
        if observation.closed is not None:
            self.listener.on_interval_closed(tracker, observation.closed)
        if observation.green_started:
            self.listener.on_green_start(tracker)
        self.listener.on_step(tracker, observation.step)

    def close(self):
        never_closed = [t.lane for t in self.trackers if t.completed_greens == 0]
        if never_closed:
            print(f"Oversaturation indices: {len(never_closed)} lanes never completed "
                  f"a green (always-green right turns get no rows)")
        self._indices_file.close()
        if self._shadow_file is not None:
            self._shadow_file.close()
