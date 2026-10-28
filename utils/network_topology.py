"""Static facts from a run's SUMO inputs, for offline tools: the sumocfg a
run manifest names, the net file's lanes and connections, the route file's
vehicle routes and its surge spec. Internal (junction) lanes are left out
throughout. Stdlib only, so the run manifest can use it.
"""

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Same storage rule as utils/oversaturation_indices.storage_capacity, which
# is not imported because that module needs traci.
METERS_PER_STORED_VEHICLE = 7.5


def resolve_sumocfg(manifest):
    """The run's sumocfg on this machine, or None. The manifest's path is
    repo-relative; its absolute path only helps on the machine that ran it."""
    environment = manifest.get("environment") or {}
    for candidate in (environment.get("simulation_config"),
                      environment.get("simulation_config_abs")):
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = REPO / path
        if path.exists():
            return path
    return None


def sumocfg_input_paths(sumocfg_path):
    """(net file, [route files]) named in a sumocfg, resolved beside it."""
    root = ET.parse(sumocfg_path).getroot()

    def values(tag):
        node = root.find(f"./input/{tag}")
        if node is None:
            return []
        return [sumocfg_path.parent / name.strip()
                for name in node.get("value", "").split(",") if name.strip()]

    nets = values("net-file")
    return (nets[0] if nets else None), values("route-files")


def surge_spec_path(route_path):
    """<stem>.surge.json beside <stem>.rou.xml, the sidecar
    tools/gen_surge_routes.py writes (its sidecar_path_for)."""
    name = route_path.name
    stem = name[:-len(".rou.xml")] if name.endswith(".rou.xml") else route_path.stem
    return route_path.parent / f"{stem}.surge.json"


def load_surge_spec(sumocfg_path):
    """{spec_path, surges: [{name, route, rate_veh_h, window}]} from the
    surge sidecar of the sumocfg's route file, or None when the route file
    has none (not a generated surge scenario). surges is empty for a
    generated file without injected surges (e.g. h2_steady)."""
    _, route_paths = sumocfg_input_paths(Path(sumocfg_path))
    for route_path in route_paths:
        spec_path = surge_spec_path(route_path)
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return {
            "spec_path": str(spec_path),
            "surges": [{key: surge.get(key) for key in ("name", "route", "rate_veh_h", "window")}
                       for surge in spec.get("surges", [])],
        }
    return None


@dataclass
class Network:
    lane_length: dict = field(default_factory=dict)   # lane -> metres
    lane_edge: dict = field(default_factory=dict)     # lane -> edge
    edge_lanes: dict = field(default_factory=dict)    # edge -> [lanes]
    connections: set = field(default_factory=set)     # (from_lane, to_lane)

    def storage(self, lane):
        return math.floor(self.lane_length[lane] / METERS_PER_STORED_VEHICLE)

    def feeders(self, lane):
        """Lanes with a connection into lane."""
        return sorted({source for source, target in self.connections if target == lane})


def read_network(net_path):
    network = Network()
    root = ET.parse(net_path).getroot()
    for edge in root.iter("edge"):
        edge_id = edge.get("id")
        if edge.get("function") == "internal" or edge_id.startswith(":"):
            continue
        for lane in edge.iter("lane"):
            lane_id = lane.get("id")
            network.lane_length[lane_id] = float(lane.get("length"))
            network.lane_edge[lane_id] = edge_id
            network.edge_lanes.setdefault(edge_id, []).append(lane_id)
    for connection in root.iter("connection"):
        source, target = connection.get("from"), connection.get("to")
        if source.startswith(":") or target.startswith(":"):
            continue
        network.connections.add((f"{source}_{connection.get('fromLane')}",
                                 f"{target}_{connection.get('toLane')}"))
    return network


def read_vehicle_routes(route_path):
    """vehicle id -> list of edges, for <vehicle> with an inline <route> or a
    route="id" reference to a top-level <route>."""
    named_routes = {}
    vehicle_routes = {}
    for _, element in ET.iterparse(route_path, events=("end",)):
        if element.tag == "route" and element.get("id"):
            named_routes[element.get("id")] = element.get("edges", "").split()
        elif element.tag == "vehicle":
            inline = element.find("route")
            if inline is not None:
                vehicle_routes[element.get("id")] = inline.get("edges", "").split()
            elif element.get("route") in named_routes:
                vehicle_routes[element.get("id")] = named_routes[element.get("route")]
            element.clear()
    return vehicle_routes
