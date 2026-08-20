"""Rank blockage targets for the Hangzhou 4x4 network (Step 0 analysis).

Reads the SUMO network and route file -- no simulation -- and, for every
approach into a central intersection, reports how much traffic uses it and
whether that traffic outgrows the through lane's storage. A high
demand-to-storage ratio is the signal that a stopped-vehicle blockage there
will fill the lane and spill back into the upstream junction, which is what
turns a mild blockage into a severe one.

    python analyze_flow.py
"""
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

NETWORK_FILE = "dataset/llm_light/Hangzhou/4_4/roadnet.net.xml"
ROUTE_FILE = "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.rou.xml"

METERS_PER_QUEUED_VEHICLE = 7.5   # 5.0 m vehicle + 2.5 m minGap (pkw vType)
FREE_FLOW_SPEED = 11.111          # m/s (pkw maxSpeed), for arrival estimates

# Edge ids look like "road_<x>_<y>_<heading>". The heading tells us which
# neighbouring node the car drives into, and which side of it the car enters.
HEADING_STEP = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}  # E, N, W, S
APPROACH_SIDE = {0: "West", 1: "South", 2: "East", 3: "North"}

INTERSECTION_COORDS = range(1, 5)  # nodes 1..4 are signals; 0 and 5 are fringe
CENTRAL_COORDS = range(2, 4)       # 2..3 are the interior (spill-on-grid) nodes


def edge_destination(edge_id):
    """(from_node, heading, dest_node) for a 'road_x_y_h' edge id."""
    _, x, y, heading = edge_id.split("_")
    x, y, heading = int(x), int(y), int(heading)
    step_x, step_y = HEADING_STEP[heading]
    return (x, y), heading, (x + step_x, y + step_y)


def is_intersection(node):
    return node[0] in INTERSECTION_COORDS and node[1] in INTERSECTION_COORDS


def is_central(node):
    return node[0] in CENTRAL_COORDS and node[1] in CENTRAL_COORDS


def read_network(path):
    """From the SUMO net, return three lookups:
        lane_length[lane_id]   -> lane length in metres
        through_lane[edge_id]  -> the lane id that carries the straight movement
        straight_next[edge_id] -> the edge a straight-through car continues onto
    """
    root = ET.parse(path).getroot()

    lane_length = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            lane_length[lane.get("id")] = float(lane.get("length"))

    through_lane = {}
    straight_next = {}
    for conn in root.findall("connection"):
        from_edge = conn.get("from")
        if conn.get("dir") == "s" and from_edge.startswith("road_"):
            through_lane.setdefault(from_edge, f"{from_edge}_{conn.get('fromLane')}")
            straight_next.setdefault(from_edge, conn.get("to"))
    return lane_length, through_lane, straight_next


def edge_travel_seconds(edge_id, lane_length):
    """Free-flow time to cross an edge (ignores signals and congestion)."""
    length = lane_length.get(f"{edge_id}_0")
    return length / FREE_FLOW_SPEED if length else 0.0


def read_demand(path, straight_next, lane_length):
    """From the route file, return per-edge traffic and departure times:
        total_demand[edge]   -> vehicles using the edge on ANY movement
        through_demand[edge] -> vehicles that carry straight on through it
        arrivals[edge]       -> estimated free-flow arrival times at the edge
        departures           -> every vehicle's departure time
    """
    root = ET.parse(path).getroot()
    total_demand = Counter()
    through_demand = Counter()
    arrivals = defaultdict(list)
    departures = []

    for vehicle in root.findall("vehicle"):
        depart = float(vehicle.get("depart"))
        departures.append(depart)
        route = vehicle.find("route").get("edges").split()

        clock = depart
        for position, edge in enumerate(route):
            total_demand[edge] += 1
            arrivals[edge].append(clock)
            next_edge = route[position + 1] if position + 1 < len(route) else None
            if straight_next.get(edge) == next_edge:
                through_demand[edge] += 1
            clock += edge_travel_seconds(edge, lane_length)
    return total_demand, through_demand, arrivals, departures


def build_approaches(total_demand, through_demand, through_lane, lane_length):
    """One record per approach edge that feeds a real intersection."""
    approaches = []
    for edge in total_demand:
        _, heading, dest = edge_destination(edge)
        if not is_intersection(dest):
            continue  # edge leaves the grid; not an approach into a signal
        lane_id = through_lane.get(edge, f"{edge}_0")
        storage = lane_length[lane_id] / METERS_PER_QUEUED_VEHICLE
        approaches.append({
            "edge": edge,
            "intersection": f"{dest[0]}_{dest[1]}",
            "approach": APPROACH_SIDE[heading],
            "central": is_central(dest),
            "total_demand": total_demand[edge],
            "through_demand": through_demand[edge],
            "length_m": lane_length[lane_id],
            "storage": storage,
            "total_ratio": total_demand[edge] / storage,
            "through_ratio": through_demand[edge] / storage,
        })

    return approaches


def print_table(title, columns, rows):
    """Print an auto-aligned table. columns is a list of
    (header, render(row) -> str, align) tuples where align is '<' or '>'."""
    print(f"\n{title}")
    body = [[render(row) for _, render, _ in columns] for row in rows]
    headers = [header for header, _, _ in columns]
    aligns = [align for _, _, align in columns]
    widths = [max(len(headers[i]), *(len(line[i]) for line in body))
              for i in range(len(headers))]

    def format_row(cells):
        return "  ".join(f"{cell:{align}{width}}"
                         for cell, align, width in zip(cells, aligns, widths))

    print(format_row(headers))
    print("  ".join("-" * width for width in widths))
    for line in body:
        print(format_row(line))


APPROACH_COLUMNS = [
    ("edge", lambda a: a["edge"], "<"),
    ("into", lambda a: a["intersection"], "<"),
    ("side", lambda a: a["approach"], "<"),
    ("total", lambda a: str(a["total_demand"]), ">"),
    ("through", lambda a: str(a["through_demand"]), ">"),
    ("length_m", lambda a: f"{a['length_m']:.1f}", ">"),
    ("storage", lambda a: f"{a['storage']:.1f}", ">"),
    ("total/stor", lambda a: f"{a['total_ratio']:.2f}", ">"),
    ("thru/stor", lambda a: f"{a['through_ratio']:.2f}", ">"),
]


def histogram(times, bin_seconds, until_seconds):
    counts = Counter(int(t // bin_seconds) * bin_seconds for t in times)
    return [(start, counts.get(start, 0))
            for start in range(0, until_seconds, bin_seconds)]


def print_ramp(title, times, bin_seconds=120, until_seconds=1800, per_hash=1):
    """Bar chart of when `times` fall. per_hash scales the bar so the busy
    network-wide chart stays readable (one '#' per `per_hash` vehicles)."""
    print(f"\n{title}")
    for start, count in histogram(times, bin_seconds, until_seconds):
        print(f"  t={start:>4}-{start + bin_seconds:<4} {count:>3}  "
              f"{'#' * (count // per_hash)}")


def check_parsing(lane_length):
    """road_0_1_0_1 is 786.4 m per the s1 scenario -- a cross-check that the
    edge-id decoding and length reads line up with a known-good fact."""
    measured = lane_length.get("road_0_1_0_1")
    verdict = "OK" if measured == 786.4 else "MISMATCH"
    print(f"\n[check] road_0_1_0_1 = {measured} m ({verdict}; s1 says 786.4)")


def main():
    lane_length, through_lane, straight_next = read_network(NETWORK_FILE)
    total_demand, through_demand, arrivals, departures = read_demand(
        ROUTE_FILE, straight_next, lane_length)

    approaches = build_approaches(total_demand, through_demand,
                                  through_lane, lane_length)
    central = sorted((a for a in approaches if a["central"]),
                     key=lambda a: a["through_ratio"], reverse=True)

    print(f"{len(departures)} vehicles; "
          f"{len(central)} approaches into central intersections.")
    print_table("Central approaches, worst spillback risk first "
                "(through demand / through-lane storage):",
                APPROACH_COLUMNS, central)

    print_ramp("Network-wide departures (one '#' = 4 vehicles):",
               departures, per_hash=4)
    for approach in central[:3]:
        print_ramp(f"Arrivals onto {approach['edge']} ({approach['approach']} "
                   f"approach of intersection_{approach['intersection']}):",
                   arrivals[approach["edge"]], until_seconds=1200)

    check_parsing(lane_length)


if __name__ == "__main__":
    main()
