"""Extents: connected groups of TOSI-flagged lanes, rebuilt once per cycle
(decisions.md 2026/10/01, 10/26, 10/27; REGIME_EXTENT_SPEC.md section 2).

Nodes are the lanes of the active TOSI-branch flags. A flagged movement links
its lane to its target lane when the target lane is itself flagged: the queue
on one is feeding the queue on the other. Each connected component (links
taken undirected) is one extent. SOSI-branch flags mark a blocking symptom and
take no part.

Identity across cycles: a component inherits the id of the last-cycle extent
holding its oldest lane (first flagged, ties broken by lane id). Each
last-cycle extent is claimed at most once, so components never merge, and when
an extent splits only the piece holding the oldest lane keeps it; the other
piece is newborn. A last-cycle extent that no component claimed has vanished.
"""

from dataclasses import dataclass


@dataclass
class Extent:
    extent_id: int
    members: frozenset


@dataclass
class ExtentRow:
    cycle_start: int
    extent_id: int
    members: frozenset   # empty when vanished
    n_conns: int         # active TOSI-branch movements leaving a member lane
    entries: int         # E: member lanes with a TOSI-branch entry this cycle
    exits: int           # X: last-cycle member lanes with an exit this cycle
    newborn: bool
    vanished: bool


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
    """Carries extent identity from cycle to cycle."""

    def __init__(self):
        # Last cycle's extents in claim order: components first, by smallest lane.
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
            if parent is None:
                extent = Extent(self._take_id(), members)
            else:
                claimed.add(parent.extent_id)
                extent = Extent(parent.extent_id, members)
            current[extent.extent_id] = extent
            rows.append(ExtentRow(
                cycle_start, extent.extent_id, members,
                n_conns=count_conns(members, tosi_movements),
                entries=len(members & events.tosi_entry_lanes),
                exits=len(parent.members & events.exit_lanes) if parent else 0,
                newborn=parent is None, vanished=False))
        for extent_id, extent in previous.items():
            if extent_id not in claimed:
                rows.append(ExtentRow(
                    cycle_start, extent_id, frozenset(), n_conns=0, entries=0,
                    exits=len(extent.members & events.exit_lanes),
                    newborn=False, vanished=True))
        self.extents = current
        return rows

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
