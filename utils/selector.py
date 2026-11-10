"""A2 rule-table selector: turns TOSI flags on a single-intersection extent
into green-split changes for the A1 cyclic plan (RULE_TABLE_SELECTOR ledger).

Pure logic, no SUMO: the inputs are small dataclasses adapted from
IntervalResult (utils/oversaturation_indices.py), MovementFlags.active_branches()
(utils/movement_flags.py), ExtentRow (utils/extents.py) and CyclicPlan
(runner_baselines.py). Feed every closed green through on_green_closed and
call evaluate once per step per intersection, after the observer and the
diagnoser have updated. evaluate returns (new plan or None, rows); nothing is
written here, the next change wires the rows into a CSV and the plan into
CyclicController.set_plan. Rows are never edited once returned, so they can be
written append-only during the run.

Lane roles: lane index 0 is the uncontrolled right turn and never gets a
command; 1 is through, 2 is left. A receiver phase is the one phase whose
lanes carry TOSI-branch flags; the other phases are donors. Decision points
come every k_flag (K03) greens of the receiver, counted from the time the
caller reports the plan applied (mark_plan_applied).
"""

import math
from collections import deque
from dataclasses import dataclass, field

from configurations import DEFAULT_INTERSECTION_CONFIG_NAME, INTERSECTION_CONFIGS
from utils.movement_flags import SOSI_BRANCH, TOSI_BRANCH
from utils.regime import LOADING, PROCESSING, RECOVERY

_SETTINGS = INTERSECTION_CONFIGS[DEFAULT_INTERSECTION_CONFIG_NAME]["global_settings"]

# --- constants (ledger ids in comments) ------------------------------------------

# K03 (receiver greens per decision window) is the run's FlagConfig.k_flag,
# passed to RuleTableSelector; it is not a module constant.
# K07: start-up lost time per green, seconds.
LOST_TIME_S = 3
# K14: shortest green a donor may be cut to, from configurations.py.
MIN_GREEN_S = _SETTINGS["min_green"]
# K16: saturation headway of a through lane, seconds per vehicle.
H_SAT_THROUGH_S = 2.143
# K17: saturation headway of a left lane; placeholder equal to K16.
H_SAT_LEFT_S = 2.143
# K18: a window is shrinking only when last < first - margin (strict at 0).
SHRINK_MARGIN_VEH = 0
# K19: longest green the receiver may grow to. This is only the default; a run
# passes its own max_green to RuleTableSelector.
MAX_GREEN_S = _SETTINGS["max_green"]
# K20: greens per lane averaged for a donor's demand; not an alias of K03.
DONOR_V_WINDOW_GREENS = 3
# Yellow after every green, used by the donor over-capacity check (rule 6).
YELLOW_S = _SETTINGS["yellow_duration"]

RIGHT_TURN_INDEX = 0
THROUGH_INDEX = 1
LEFT_INDEX = 2

INTERSECTION_KIND = "intersection"
ROUTE_KIND = "route"

NEW_COMMAND = "new_command"
ADD = "add"
HOLD = "hold"
EXPIRE = "expire"
AT_FLOOR = "at_floor"
DG_ZERO = "dg_zero"
ROUTE_DEFERRED = "route_deferred_fbp"
OUT_OF_SCOPE = "rule7_out_of_scope"
SOSI_SYMPTOM = "sosi_symptom_no_action"
RECEIVER_AT_MAX_GREEN = "receiver_at_max_green"
RIGHT_LANE_NO_ACTION = "right_lane_no_action"
REENTRY_AFTER_EXPIRY = "reentry_after_expiry"

FLOOR_BOUND_MIN_GREEN = "min_green"
FLOOR_BOUND_VC = "vc"
RECEIVER_BOUND_MAX_GREEN = "max_green"
RECEIVER_BOUND_DONOR_POOL = "donor_pool"
RECEIVER_BOUND_NONE = "none"

ROW_COLUMNS = [
    "t", "intersection", "extent_id", "regime", "phase", "outcome",
    "window_first_t", "window_last_t", "held_first", "held_last",
    "dg_requested", "dg_given", "short_by", "donors",
    "donor_vc_before", "donor_vc_after", "donor_floor_bound", "receiver_bound",
    "add_blocked_recovery", "expiry_t", "donor_over_capacity_greens",
    "plan_before", "plan_after", "ignored_extents", "ignored_extent_cycles",
]


# --- inputs ---------------------------------------------------------------------

def lane_index(lane):
    """SUMO lane ids are always '<edge id>_<lane index>'."""
    return int(lane.rsplit("_", 1)[1])


def is_right_turn(lane):
    return lane_index(lane) == RIGHT_TURN_INDEX


def h_sat(lane):
    return H_SAT_LEFT_S if lane_index(lane) == LEFT_INDEX else H_SAT_THROUGH_S


def lane_branches(active_branches):
    """{lane: branch} from MovementFlags.active_branches(), which is keyed by
    (lane, target_lane). A lane with any TOSI-branch movement is a TOSI lane."""
    branches = {}
    for (lane, _target), branch in active_branches.items():
        if branch == TOSI_BRANCH or lane not in branches:
            branches[lane] = branch
    return branches


@dataclass(frozen=True)
class GreenRecord:
    """One completed green of one lane, adapted from IntervalResult."""
    tls: str
    lane: str
    green_start: int
    green_end: int
    tosi: object          # float, or None when unmeasured (blocked lane)
    held_over: int
    entered_red: object   # int, or None when not measured
    entered_green: object
    exited_green: object

    @property
    def green_s(self):
        return self.green_end - self.green_start

    @property
    def entries(self):
        if self.entered_red is None or self.entered_green is None:
            return None
        return self.entered_red + self.entered_green

    @classmethod
    def from_interval_result(cls, result):
        return cls(tls=result.tls, lane=result.lane, green_start=result.green_start,
                   green_end=result.green_end, tosi=result.tosi,
                   held_over=result.held_over, entered_red=result.entered_red,
                   entered_green=result.entered_green, exited_green=result.exited_green)


@dataclass(frozen=True)
class ExtentView:
    """One extent as the selector sees it. An extent has ended for the
    selector as soon as it is vanished (its component is empty), even though
    the ExtentTracker keeps stepping it until regime N; an extent missing from
    the list given to evaluate has ended too."""
    extent_id: int
    members: frozenset
    intersections: frozenset
    regime: str
    cycle_start: int
    vanished: bool = False

    @property
    def kind(self):
        return INTERSECTION_KIND if len(self.intersections) == 1 else ROUTE_KIND

    @classmethod
    def from_extent_row(cls, row, lane_tls):
        """row: utils.extents.ExtentRow; lane_tls: Diagnoser.lane_tls."""
        return cls(extent_id=row.extent_id, members=frozenset(row.members),
                   intersections=frozenset(lane_tls[lane] for lane in row.members),
                   regime=row.regime, cycle_start=row.cycle_start,
                   vanished=row.vanished)


@dataclass(frozen=True)
class Plan:
    """Mirror of CyclicPlan's order / splits / offset; the selector only ever
    changes splits."""
    order: tuple
    splits: dict
    offset: int = 0

    @property
    def greens_sum(self):
        return sum(self.splits[phase] for phase in self.order)

    def with_splits(self, splits):
        return Plan(order=self.order, splits=dict(splits), offset=self.offset)

    @property
    def text(self):
        return ";".join(f"{phase}:{self.splits[phase]}" for phase in self.order)


@dataclass
class IntersectionLanes:
    """Controlled lanes of one signal grouped by phase; right turns are kept
    apart in right_turn_lanes and take no part in any phase."""
    tls: str
    phase_lanes: dict

    def __post_init__(self):
        self.right_turn_lanes = frozenset(lane for lanes in self.phase_lanes.values()
                                          for lane in lanes if is_right_turn(lane))
        self.phase_lanes = {phase: tuple(lane for lane in lanes if not is_right_turn(lane))
                            for phase, lanes in self.phase_lanes.items()}
        self.lane_phase = {lane: phase for phase, lanes in self.phase_lanes.items()
                           for lane in lanes}

    @property
    def all_lanes(self):
        return frozenset(self.lane_phase)


# --- donor arithmetic (pure) -------------------------------------------------------

def mean_entries(records):
    """v_lane: mean entries per green over the records that measured them."""
    entries = [record.entries for record in records if record.entries is not None]
    if not entries:
        return None
    return sum(entries) / len(entries)


def phase_floor(demand_s):
    """Floor green of a donor and which bound set it."""
    if demand_s > MIN_GREEN_S:
        return math.ceil(demand_s), FLOOR_BOUND_VC
    return math.ceil(MIN_GREEN_S), FLOOR_BOUND_MIN_GREEN


@dataclass
class DonorAssessment:
    phase: str
    demand_s: float      # max over lanes of v_lane * h_sat
    green: int
    floor: int
    floor_bound: str
    given: int = 0

    @property
    def spare(self):
        return max(0, math.floor(self.green - self.floor))

    @property
    def vc_before(self):
        return self.demand_s / self.green

    @property
    def vc_after(self):
        return self.demand_s / (self.green - self.given)


def assess_donor(phase, lanes, history, green):
    """None when no lane of the phase has a measured green in the window."""
    demands = []
    for lane in lanes:
        v_lane = mean_entries(history.get(lane, ()))
        if v_lane is not None:
            demands.append(v_lane * h_sat(lane))
    if not demands:
        return None
    demand_s = max(demands)
    floor, bound = phase_floor(demand_s)
    return DonorAssessment(phase=phase, demand_s=demand_s, green=green,
                           floor=floor, floor_bound=bound)


def take_from_donors(donors, needed):
    """Fill `needed` from donors in ascending v/c, each giving up to its spare.
    Returns the donors in the order they were asked."""
    ordered = sorted(donors, key=lambda donor: donor.vc_before)
    remaining = needed
    for donor in ordered:
        donor.given = min(donor.spare, remaining)
        remaining -= donor.given
    return ordered


def request_dg(flagged_records, green):
    """Rule 2: floor(TOSI_phase * g_phase), TOSI_phase the max over the
    receiver's flagged lanes in their latest green."""
    tosis = [record.tosi for record in flagged_records if record.tosi is not None]
    if not tosis:
        return 0
    return math.floor(max(tosis) * green)


@dataclass
class Allocation:
    dg_requested: int
    dg_capped: int
    donors: list
    given: int
    short_by: int

    @property
    def giving_phases(self):
        return {donor.phase for donor in self.donors if donor.given > 0}

    @property
    def receiver_bound(self):
        if self.short_by > 0:
            return RECEIVER_BOUND_DONOR_POOL
        if self.dg_capped < self.dg_requested:
            return RECEIVER_BOUND_MAX_GREEN
        return RECEIVER_BOUND_NONE


def allocate(dg_requested, receiver_green, donors, max_green):
    dg_capped = max(0, min(dg_requested, max_green - receiver_green))
    ordered = take_from_donors(donors, dg_capped)
    given = sum(donor.given for donor in ordered)
    return Allocation(dg_requested=dg_requested, dg_capped=dg_capped, donors=ordered,
                      given=given, short_by=dg_capped - given)


def apply_allocation(plan, receiver, allocation, max_green):
    """Rule 4: receiver grows by what was given, donors shrink by their share."""
    splits = dict(plan.splits)
    splits[receiver] += allocation.given
    for donor in allocation.donors:
        splits[donor.phase] -= donor.given
    new_plan = plan.with_splits(splits)
    assert new_plan.greens_sum == plan.greens_sum
    assert all(MIN_GREEN_S <= green <= max_green for green in splits.values())
    return new_plan


def is_over_capacity(record):
    """Rule 6: entries above what the green could discharge."""
    if record.entries is None:
        return False
    capacity = (record.green_s + YELLOW_S - LOST_TIME_S) / h_sat(record.lane)
    return record.entries > capacity


# --- state ---------------------------------------------------------------------------

@dataclass
class ActiveCommand:
    extent_id: int
    phase: str
    flagged_lanes: frozenset
    giving_phases: set                                # donors that gave seconds so far
    effective_t: object = None                        # int once the caller applied the plan
    window: dict = field(default_factory=dict)        # receiver green_start -> held_sum
    over_capacity: set = field(default_factory=set)   # (donor phase, green_start)


@dataclass
class ExpiryWatch:
    phase: str
    t_expired: int
    greens_seen: set = field(default_factory=set)


@dataclass
class IntersectionState:
    lanes: IntersectionLanes
    base_plan: object = None
    command: object = None
    expiry_watch: object = None
    history: dict = field(default_factory=dict)       # lane -> deque of GreenRecord
    evaluated: set = field(default_factory=set)       # (extent_id, cycle_start)
    ignored_extent_ids: set = field(default_factory=set)      # since the previous row
    ignored_cycle_starts: set = field(default_factory=set)    # since the previous row


# --- row formatting ---------------------------------------------------------------

def blank_row(t, tls, extent_id, regime, phase, outcome, plan):
    row = {column: "" for column in ROW_COLUMNS}
    row.update(t=t, intersection=tls, extent_id=extent_id, regime=regime, phase=phase,
               outcome=outcome, add_blocked_recovery=0,
               plan_before=plan.text, plan_after=plan.text,
               ignored_extents="", ignored_extent_cycles=0)
    return row


def join_per_phase(donors, value_of):
    return ";".join(f"{donor.phase}:{value_of(donor)}" for donor in donors)


def fill_allocation(row, allocation):
    donors = allocation.donors
    row.update(dg_requested=allocation.dg_requested, dg_given=allocation.given,
               short_by=allocation.short_by, receiver_bound=allocation.receiver_bound,
               donors=join_per_phase(donors, lambda d: d.given),
               donor_vc_before=join_per_phase(donors, lambda d: f"{d.vc_before:.3f}"),
               donor_vc_after=join_per_phase(donors, lambda d: f"{d.vc_after:.3f}"),
               donor_floor_bound=join_per_phase(donors, lambda d: d.floor_bound))


# --- selector ---------------------------------------------------------------------

class RuleTableSelector:
    """One instance per run; keeps per-intersection command state and every
    row it has produced. flag_config is the run's FlagConfig: its k_flag is
    the decision window (K03). max_green is the run's longest green (K19);
    it must match what the PhaseHandlers clamp to."""

    def __init__(self, intersections, flag_config, max_green=MAX_GREEN_S):
        self.decision_window_greens = flag_config.k_flag
        self.max_green = max_green
        self.states = {lanes.tls: IntersectionState(lanes) for lanes in intersections}
        self.rows = []

    def mark_plan_applied(self, tls, effective_t):
        """Call when the executor starts running a plan this selector returned
        for a new command or an add, with the cycle boundary it took effect at.
        Receiver greens are counted toward the decision window only from then;
        a restored base plan (expire) needs no call."""
        command = self.states[tls].command
        if command is not None:
            command.effective_t = effective_t

    # --- per closed green -----------------------------------------------------

    def on_green_closed(self, record):
        state = self.states.get(record.tls)
        if state is None or record.lane not in state.lanes.lane_phase:
            return
        state.history.setdefault(record.lane, deque(maxlen=DONOR_V_WINDOW_GREENS)).append(record)
        phase = state.lanes.lane_phase[record.lane]
        if state.command is not None:
            self._track_command_green(state.command, phase, record)
        watch = state.expiry_watch
        if watch is not None and phase == watch.phase and record.green_start >= watch.t_expired:
            watch.greens_seen.add(record.green_start)

    def _track_command_green(self, command, phase, record):
        if command.effective_t is None or record.green_start < command.effective_t:
            return
        if phase == command.phase:
            held = record.held_over if record.lane in command.flagged_lanes else 0
            command.window[record.green_start] = command.window.get(record.green_start, 0) + held
        elif phase in command.giving_phases and is_over_capacity(record):
            command.over_capacity.add((phase, record.green_start))

    # --- per step ---------------------------------------------------------------

    def evaluate(self, t, tls, plan, extents, branches):
        """extents: live ExtentViews (any intersection); branches: {lane: T|S}
        from lane_branches(). Returns (new Plan or None, rows emitted now)."""
        state = self.states[tls]
        self._note_ignored_extents(state, extents)
        rows = self._check_reentry(t, state, plan, branches)
        if state.command is not None:
            new_plan, command_rows = self._maybe_decision_point(t, state, plan, extents,
                                                                branches)
        else:
            new_plan, command_rows = self._try_new_command(t, state, plan, extents, branches)
        rows += command_rows
        if rows:
            self._stamp_ignored_extents(state, rows[0])
        self.rows.extend(rows)
        return new_plan, rows

    def _note_ignored_extents(self, state, extents):
        """While a command is active, other live extents flagged here get no
        action; remember them for the next row at this intersection."""
        command = state.command
        if command is None:
            return
        for extent in self._touching(state, extents):
            if extent.extent_id == command.extent_id or extent.vanished:
                continue
            if self._right_lanes_only(state, extent):
                continue
            state.ignored_extent_ids.add(extent.extent_id)
            state.ignored_cycle_starts.add(extent.cycle_start)

    def _stamp_ignored_extents(self, state, row):
        row["ignored_extents"] = ";".join(str(i) for i in sorted(state.ignored_extent_ids))
        row["ignored_extent_cycles"] = len(state.ignored_cycle_starts)
        state.ignored_extent_ids = set()
        state.ignored_cycle_starts = set()

    def _check_reentry(self, t, state, plan, branches):
        """A row pointing back at the expiry (expiry_t, phase) when the expired
        phase re-flags TOSI within K03 of its greens."""
        watch = state.expiry_watch
        if watch is None:
            return []
        if len(watch.greens_seen) > self.decision_window_greens:
            state.expiry_watch = None
            return []
        if not self._tosi_lanes(state, branches, state.lanes.phase_lanes[watch.phase]):
            return []
        state.expiry_watch = None
        row = blank_row(t, state.lanes.tls, "", "", watch.phase, REENTRY_AFTER_EXPIRY, plan)
        row["expiry_t"] = watch.t_expired
        return [row]

    # --- new command (rules 1-4) ----------------------------------------------

    def _try_new_command(self, t, state, plan, extents, branches):
        rows = []
        for extent in self._touching(state, extents):
            key = (extent.extent_id, extent.cycle_start)
            if key in state.evaluated:
                continue
            if extent.kind == ROUTE_KIND:
                state.evaluated.add(key)
                rows.append(blank_row(t, state.lanes.tls, extent.extent_id, extent.regime,
                                      "", ROUTE_DEFERRED, plan))
                continue
            if self._right_lanes_only(state, extent):
                state.evaluated.add(key)
                rows.append(blank_row(t, state.lanes.tls, extent.extent_id, extent.regime,
                                      "", RIGHT_LANE_NO_ACTION, plan))
                continue
            if extent.regime not in (LOADING, PROCESSING):
                continue
            state.evaluated.add(key)
            new_plan, row = self._new_command(t, state, plan, extent, branches)
            if row is not None:
                rows.append(row)
            if new_plan is not None:
                return new_plan, rows
        return None, rows

    def _touching(self, state, extents):
        lanes = state.lanes.all_lanes | state.lanes.right_turn_lanes
        return [extent for extent in extents if extent.members & lanes]

    def _right_lanes_only(self, state, extent):
        """For a touching extent: none of its lanes here is a phase lane."""
        return not extent.members & state.lanes.all_lanes

    def _new_command(self, t, state, plan, extent, branches):
        tls = state.lanes.tls
        tosi_phases = [phase for phase, lanes in state.lanes.phase_lanes.items()
                       if self._tosi_lanes(state, branches, lanes)]
        if not tosi_phases:
            if not self._sosi_lanes(state, branches):
                return None, None
            return None, blank_row(t, tls, extent.extent_id, extent.regime, "",
                                   SOSI_SYMPTOM, plan)
        if len(tosi_phases) > 1:
            return None, blank_row(t, tls, extent.extent_id, extent.regime,
                                   "+".join(tosi_phases), OUT_OF_SCOPE, plan)
        phase = tosi_phases[0]
        flagged = self._tosi_lanes(state, branches, state.lanes.phase_lanes[phase])
        row = blank_row(t, tls, extent.extent_id, extent.regime, phase, "", plan)
        allocation = self._allocate(state, plan, phase, flagged)
        if allocation is None:
            row["outcome"] = DG_ZERO
            row["dg_requested"] = 0
            return None, row
        fill_allocation(row, allocation)
        if allocation.dg_capped == 0:
            row["outcome"] = RECEIVER_AT_MAX_GREEN
            return None, row
        if allocation.given == 0:
            row["outcome"] = OUT_OF_SCOPE
            return None, row
        new_plan = apply_allocation(plan, phase, allocation, self.max_green)
        row.update(outcome=NEW_COMMAND, plan_after=new_plan.text)
        if state.base_plan is None:
            state.base_plan = plan
        state.command = ActiveCommand(extent_id=extent.extent_id, phase=phase,
                                      flagged_lanes=flagged,
                                      giving_phases=allocation.giving_phases)
        return new_plan, row

    def _tosi_lanes(self, state, branches, lanes):
        return frozenset(lane for lane in lanes if branches.get(lane) == TOSI_BRANCH)

    def _sosi_lanes(self, state, branches):
        return frozenset(lane for lane in state.lanes.all_lanes
                         if branches.get(lane) == SOSI_BRANCH)

    def _allocate(self, state, plan, receiver, flagged):
        """Rules 2-3 against the current plan; None when nothing is requested."""
        latest = [state.history[lane][-1] for lane in flagged if state.history.get(lane)]
        dg_requested = request_dg(latest, plan.splits[receiver])
        if dg_requested == 0:
            return None
        donors = []
        for phase in plan.order:
            if phase == receiver:
                continue
            donor = assess_donor(phase, state.lanes.phase_lanes[phase], state.history,
                                 plan.splits[phase])
            if donor is not None:
                donors.append(donor)
        return allocate(dg_requested, plan.splits[receiver], donors, self.max_green)

    # --- decision point (rules 5-6) ----------------------------------------------

    def _maybe_decision_point(self, t, state, plan, extents, branches):
        command = state.command
        if len(command.window) < self.decision_window_greens:
            return None, []
        new_plan, row = self._decision_point(t, state, plan, extents, branches)
        command.window = {}
        command.over_capacity = set()
        return new_plan, [row]

    def _decision_point(self, t, state, plan, extents, branches):
        command = state.command
        live = {extent.extent_id: extent for extent in extents}
        extent = live.get(command.extent_id)
        regime = "" if extent is None else extent.regime
        row = blank_row(t, state.lanes.tls, command.extent_id, regime, command.phase, "", plan)
        self._fill_window(row, command)
        shrinking = row["held_last"] < row["held_first"] - SHRINK_MARGIN_VEH
        if extent is None or extent.vanished:
            return self._expire(t, state, row)
        if regime == RECOVERY:
            row.update(outcome=HOLD, add_blocked_recovery=int(not shrinking))
            return None, row
        if shrinking:
            row["outcome"] = HOLD
            return None, row
        return self._add(t, state, plan, command, branches, row)

    def _fill_window(self, row, command):
        starts = sorted(command.window)
        row.update(window_first_t=starts[0], window_last_t=starts[-1],
                   held_first=command.window[starts[0]], held_last=command.window[starts[-1]],
                   donor_over_capacity_greens=len(command.over_capacity))

    def _expire(self, t, state, row):
        base_plan = state.base_plan
        row.update(outcome=EXPIRE, plan_after=base_plan.text)
        state.command = None
        state.base_plan = None
        state.expiry_watch = ExpiryWatch(phase=row["phase"], t_expired=t)
        return base_plan, row

    def _add(self, t, state, plan, command, branches, row):
        flagged = self._tosi_lanes(state, branches, state.lanes.phase_lanes[command.phase])
        if flagged:
            command.flagged_lanes = flagged
        allocation = self._allocate(state, plan, command.phase, command.flagged_lanes)
        if allocation is None:
            row.update(outcome=DG_ZERO, dg_requested=0)
            return None, row
        fill_allocation(row, allocation)
        if allocation.dg_capped == 0:
            row["outcome"] = RECEIVER_AT_MAX_GREEN
            return None, row
        if allocation.given == 0:
            row["outcome"] = AT_FLOOR
            return None, row
        new_plan = apply_allocation(plan, command.phase, allocation, self.max_green)
        row.update(outcome=ADD, plan_after=new_plan.text)
        command.giving_phases |= allocation.giving_phases
        command.effective_t = None
        return new_plan, row
