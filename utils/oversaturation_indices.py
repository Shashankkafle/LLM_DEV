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

VEHICLE_VARS = [tc.VAR_SPEED, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID]

STOPLINE_EXIT = "stopline"
LANE_CHANGE_EXIT = "lanechange"
OTHER_EXIT = "other"
EXIT_KINDS = (STOPLINE_EXIT, LANE_CHANGE_EXIT, OTHER_EXIT)

INDICES_COLUMNS = [
    "t_green_start", "t_green_end", "t_allred", "tls", "lane", "target_lane",
    "green_s", "tosi", "discharged", "sosi", "ratio_start", "ratio_end",
    "blocked_flag",
    "entered_red", "entered_green", "exited_green", "first_green",
    "lc_out_green", "other_out_green", "stopline_out_red", "lc_out_red", "other_out_red",
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


def entered_ids(prev_ids, curr_ids):
    return set(curr_ids) - set(prev_ids)


def lane_edge(lane_id):
    """SUMO lane ids are always '<edge id>_<lane index>'."""
    return lane_id.rsplit("_", 1)[0]


def classify_exit(lane_now, source_lane, link_targets):
    """Why a vehicle left source_lane, judged only by the lane it is on now
    (None or "" when it is on no lane: arrived, removed, teleporting). Never
    looks ahead: a teleported vehicle reappears far away and would be
    misread. A 1 s step can jump the short junction-internal lane, so landing
    on a link target counts as a stop-line crossing too."""
    if not lane_now:
        return OTHER_EXIT
    if lane_now.startswith(":") or lane_now in link_targets:
        return STOPLINE_EXIT
    if lane_edge(lane_now) == lane_edge(source_lane):
        return LANE_CHANGE_EXIT
    return OTHER_EXIT


def zero_exit_counts():
    return {kind: 0 for kind in EXIT_KINDS}


def subtract_exit_counts(later, earlier):
    return {kind: later[kind] - earlier[kind] for kind in EXIT_KINDS}


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
    entry_count: int = 0


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
    # SOSI's rules counted one at a time over the same green seconds:
    # (a) and (b) are lane-level, (c) is per target lane.
    stop_halted_steps: int
    no_exit_steps: int
    target_halted_steps: dict
    # Lane flow counts. The green window is [green_start, allred), yellow
    # included, the same window discharged covers; the red window runs from
    # the previous green's allred (t=0 on the first green) to green_start.
    # Exits are split by classify_exit; exited_green is stop-line exits only.
    # None when not measured (synthetic or replayed results).
    entered_red: object = None
    entered_green: object = None
    exited_green: object = None
    first_green: object = None
    lc_out_green: object = None
    other_out_green: object = None
    stopline_out_red: object = None
    lc_out_red: object = None
    other_out_red: object = None

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
    target_halted_steps: dict
    blocked_flag: bool
    entries_at_start: int
    exits_at_start: dict
    green_end: object = None
    stop_halted_steps: int = 0
    no_exit_steps: int = 0


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
        # Running totals over the whole run, obstacle excluded (the snapshots
        # already drop it). A lane change into this lane counts as an entry.
        self.entries_total = 0
        self.exits_total = zero_exit_counts()
        self.entries_at_last_close = 0
        self.exits_at_last_close = zero_exit_counts()

    def observe(self, tau, signal, prev, curr, target_snapshots, lane_blocked,
                vehicle_lanes=None):
        """Advance one step. tau labels the second whose signal is `signal`;
        prev is the snapshot at tau, curr the one at tau+1. vehicle_lanes
        maps vehicle id -> lane at tau+1, for classifying exits; a vehicle
        missing from it is on no lane."""
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
        observation.step = self._evaluate_step(tau, signal, prev, curr, target_snapshots,
                                               vehicle_lanes or {})
        return observation

    def _start(self, tau, prev, lane_blocked):
        return OpenGreen(green_start=tau, halted_at_start=prev.halted_ids,
                         blocked_steps={target: 0 for target in self.targets},
                         target_halted_steps={target: 0 for target in self.targets},
                         blocked_flag=lane_blocked,
                         entries_at_start=self.entries_total,
                         exits_at_start=dict(self.exits_total))

    def _is_counted_green_second(self, tau):
        green = self.open_green
        return (green is not None and green.green_end is None
                and tau - green.green_start >= SOSI_SKIP_STEPS)

    def _count_exits(self, exited, vehicle_lanes):
        for vehicle_id in exited:
            kind = classify_exit(vehicle_lanes.get(vehicle_id), self.lane, self.targets)
            self.exits_total[kind] += 1

    def _evaluate_step(self, tau, signal, prev, curr, target_snapshots, vehicle_lanes):
        exited = exited_ids(prev.ids, curr.ids)
        exit_count = len(exited)
        entry_count = len(entered_ids(prev.ids, curr.ids))
        self._count_exits(exited, vehicle_lanes)
        self.entries_total += entry_count
        counted = self._is_counted_green_second(tau)
        if counted and curr.halted_near_stop:
            self.open_green.stop_halted_steps += 1
        if counted and exit_count == 0:
            self.open_green.no_exit_steps += 1
        targets = []
        for target in self.targets:
            near_entry = target_snapshots.get(target, EMPTY_SNAPSHOT).halted_near_entry
            blocked = is_spillback_blocked(curr.halted_near_stop, exit_count, near_entry)
            if counted and near_entry:
                self.open_green.target_halted_steps[target] += 1
            if counted and blocked:
                self.open_green.blocked_steps[target] += 1
            targets.append(TargetInputs(target, near_entry, blocked if counted else None))
        return StepInputs(t=tau + 1, lane=self.lane, signal=signal,
                          vehicle_count=len(curr.ids), halted_count=len(curr.halted_ids),
                          halted_near_stop=curr.halted_near_stop, exit_count=exit_count,
                          targets=targets, entry_count=entry_count)

    def _close(self, allred, prev):
        green = self.open_green
        self.open_green = None
        first_green = self.completed_greens == 0
        self.completed_greens += 1
        entered_red = green.entries_at_start - self.entries_at_last_close
        exits_red = subtract_exit_counts(green.exits_at_start, self.exits_at_last_close)
        exits_green = subtract_exit_counts(self.exits_total, green.exits_at_start)
        self.entries_at_last_close = self.entries_total
        self.exits_at_last_close = dict(self.exits_total)
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
            stop_halted_steps=green.stop_halted_steps,
            no_exit_steps=green.no_exit_steps,
            target_halted_steps=dict(green.target_halted_steps),
            entered_red=entered_red,
            entered_green=self.entries_total - green.entries_at_start,
            exited_green=exits_green[STOPLINE_EXIT],
            first_green=first_green,
            lc_out_green=exits_green[LANE_CHANGE_EXIT],
            other_out_green=exits_green[OTHER_EXIT],
            stopline_out_red=exits_red[STOPLINE_EXIT],
            lc_out_red=exits_red[LANE_CHANGE_EXIT],
            other_out_red=exits_red[OTHER_EXIT],
        )


# --- CSV formatting -------------------------------------------------------------

def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def flow_values(result):
    """The lane flow counts as CSV cells, in INDICES_COLUMNS order."""
    first_green = None if result.first_green is None else int(result.first_green)
    return [format_value(result.entered_red), format_value(result.entered_green),
            format_value(result.exited_green), format_value(first_green),
            format_value(result.lc_out_green), format_value(result.other_out_green),
            format_value(result.stopline_out_red), format_value(result.lc_out_red),
            format_value(result.other_out_red)]


def interval_rows(result):
    """One CSV row per movement (lane -> target lane) of a completed green."""
    for target in result.blocked_steps:
        yield [result.green_start, result.green_end, result.allred, result.tls,
               result.lane, target, result.green_s, format_value(result.tosi),
               result.discharged, format_value(result.sosi[target]),
               format_value(result.ratio_start), format_value(result.ratio_end),
               int(result.blocked_flag), *flow_values(result)]


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
        self._entries = {}
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
        """(snapshot per observed lane, vehicle id -> current lane). The lane
        map covers every vehicle ever seen on an observed lane that is still
        in the network, so it locates vehicles that just left one."""
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
        vehicle_lanes = {v: state[tc.VAR_LANE_ID] for v, state in states.items()}
        return snapshots, vehicle_lanes

    def _blocked_lanes(self):
        if self.blockage_manager is None:
            return set()
        return set(self.blockage_manager.get_blocked_lane_ids())

    def update(self, t):
        tau = t - 1
        curr, vehicle_lanes = self._snapshots()
        signal_results = traci.trafficlight.getAllSubscriptionResults()
        blocked_lanes = self._blocked_lanes()
        entries = {}
        for tracker in self.trackers:
            state = signal_results[tracker.tls][tc.TL_RED_YELLOW_GREEN_STATE]
            observation = tracker.observe(
                tau, lane_signal(state, tracker.link_indices),
                self._prev[tracker.lane], curr[tracker.lane], curr,
                tracker.lane in blocked_lanes, vehicle_lanes)
            self._record(tracker, observation)
            entries[tracker.lane] = observation.step.entry_count
        self._entries = entries
        self._prev = curr

    def lane_halting(self):
        """{signalized lane: halted vehicles} in the snapshot of the last
        update(t), i.e. at t; obstacle excluded."""
        return {tracker.lane: len(self._prev[tracker.lane].halted_ids)
                for tracker in self.trackers}

    def lane_entries(self):
        """{signalized lane: vehicles that entered it between the last two
        updates}. A lane change counts as entering the new lane."""
        return dict(self._entries)

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
