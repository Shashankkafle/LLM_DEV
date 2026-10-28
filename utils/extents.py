"""Extents: connected groups of TOSI-flagged lanes, rebuilt once per cycle,
each with its own regime (decisions.md 2026/10/01, 10/26, 10/27;
REGIME_EXTENT_SPEC.md sections 2-3).

Nodes are the lanes of the active TOSI-branch flags. A flagged movement links
its lane to its target lane when the target lane is itself flagged: the queue
on one is feeding the queue on the other. Each connected component (links
taken undirected) is one extent. SOSI-branch flags mark a blocking symptom and
take no part.

Identity across cycles: a component inherits the id and regime of the
last-cycle extent holding its oldest lane (first flagged, ties broken by lane
id). Each last-cycle extent is claimed at most once, so components never
merge, and when an extent splits only the piece holding the oldest lane keeps
it; the other piece is newborn and starts in L. A last-cycle extent that no
component claimed has vanished; it keeps being stepped, with no entries and no
connections, until its regime reaches N.
"""

from dataclasses import dataclass

from utils.regime import NORMAL, next_extent_regime


@dataclass
class Extent:
    extent_id: int
    members: frozenset
    regime: str
    root_lane: str = ""  # the oldest lane in the cycle the extent was born


@dataclass
class ExtentRow:
    cycle_start: int
    extent_id: int
    members: frozenset   # empty when vanished
    n_conns: int         # active TOSI-branch movements leaving a member lane
    entries: int         # E: member lanes with a TOSI-branch entry this cycle
    exits: int           # X: last-cycle member lanes with an exit this cycle
    regime: str
    newborn: bool
    vanished: bool
    root_lane: str = ""
    oldest_lane: str = ""  # the lane identity was carried by; empty when vanished


def connected_components(tosi_movements):
    """tosi_movements: set of (lane, target_lane). Returns frozensets of lanes,
    ordered by their smallest lane."""
    lanes = {lane for lane, _ in tosi_movements}
    neighbours = {lane: set() for lane in lanes}
    for lane, target in tosi_movements:
        if target in lanes:
            neighbours[lane].add(target)
            neighbours[target].add(lane)
    components = []
    seen = set()
    for start in sorted(lanes):
        if start in seen:
            continue
        component = reachable_lanes(start, neighbours)
        seen |= component
        components.append(frozenset(component))
    return components


def reachable_lanes(start, neighbours):
    found = {start}
    stack = [start]
    while stack:
        for lane in neighbours[stack.pop()] - found:
            found.add(lane)
            stack.append(lane)
    return found


def count_conns(members, tosi_movements):
    return sum(1 for lane, _ in tosi_movements if lane in members)


class ExtentTracker:
    """Carries extent identity and regime from cycle to cycle."""

    def __init__(self):
        # Last cycle's extents in claim order: components by smallest lane,
        # then the vanished ones not yet back to N.
        self.extents = {}
        self.first_flagged = {}   # lane -> cycle_start it first had a TOSI flag
        self._next_id = 1

    def close_cycle(self, cycle_start, tosi_movements, events):
        """tosi_movements: the active TOSI-branch flags at the end of the
        cycle. events: that cycle's CycleEvents. Returns one ExtentRow per
        extent, current then vanished."""
        for lane, _ in tosi_movements:
            self.first_flagged.setdefault(lane, cycle_start)
        previous = self.extents
        claimed = set()
        current = {}
        rows = []
        for members in connected_components(tosi_movements):
            parent = self._unclaimed_parent(members, previous, claimed)
            row = self._component_row(cycle_start, members, parent,
                                      tosi_movements, events)
            if parent is not None:
                claimed.add(parent.extent_id)
            current[row.extent_id] = Extent(row.extent_id, members, row.regime,
                                            row.root_lane)
            rows.append(row)
        for extent_id, extent in previous.items():
            if extent_id in claimed:
                continue
            row = self._vanished_row(cycle_start, extent, events)
            if row.regime != NORMAL:
                current[extent_id] = Extent(extent_id, extent.members, row.regime,
                                            extent.root_lane)
            rows.append(row)
        self.extents = current
        return rows

    def _component_row(self, cycle_start, members, parent, tosi_movements, events):
        newborn = parent is None
        extent_id = self._take_id() if newborn else parent.extent_id
        entries = len(members & events.tosi_entry_lanes)
        exits = 0 if newborn else len(parent.members & events.exit_lanes)
        n_conns = count_conns(members, tosi_movements)
        state = NORMAL if newborn else parent.regime
        regime = next_extent_regime(state, entries, exits, n_conns, newborn)
        oldest = self._oldest(members)
        root_lane = oldest if newborn else parent.root_lane
        return ExtentRow(cycle_start, extent_id, members, n_conns, entries, exits,
                         regime, newborn=newborn, vanished=False,
                         root_lane=root_lane, oldest_lane=oldest)

    def _vanished_row(self, cycle_start, extent, events):
        exits = len(extent.members & events.exit_lanes)
        regime = next_extent_regime(extent.regime, 0, exits, 0, newborn=False)
        return ExtentRow(cycle_start, extent.extent_id, frozenset(), 0, 0, exits,
                         regime, newborn=False, vanished=True,
                         root_lane=extent.root_lane)

    def _oldest(self, members):
        return min(members, key=lambda lane: (self.first_flagged[lane], lane))

    def _unclaimed_parent(self, members, previous, claimed):
        oldest = self._oldest(members)
        for extent_id, extent in previous.items():
            if oldest in extent.members and extent_id not in claimed:
                return extent
        return None

    def _take_id(self):
        extent_id = self._next_id
        self._next_id += 1
        return extent_id
