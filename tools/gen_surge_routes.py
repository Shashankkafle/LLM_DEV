"""Generate a demand-surge SUMO route file offline from a base .rou.xml.

The base file is never modified and its vehicles are copied verbatim (same
attributes, same route, same bytes). The generator only ADDS vehicles:

  * Tiling (any --horizon > 3600): a slice of hour 1 (--tile-source START END,
    default the whole hour) is repeated to fill the horizon at +3600 s per tile
    with +-30 s jitter, ids "<id>_h2", "<id>_h3", ... --hour2-scale thins (< 1)
    or clones (> 1) each tile at random.
  * Clone surge (--mult > 1): inside --window, qualifying base vehicles get
    extra clones following a trapezoidal profile (linear ramp up over --ramp,
    hold, ramp down), each with +-20 s jitter and id "surge_<id>_<k>".
    --surge-type network surges everyone; --surge-type route surges only
    vehicles whose route passes the --via edges in order (default road_2_4_3,
    the C3 approach) or matches the --od pair.
  * Inject surge (scenario JSON only): NEW vehicles on an explicit route (edge
    list, or an OD pair resolved by sumolib) at an absolute rate in veh/h with
    the same trapezoidal profile, id "surge_<name>_<k>". Several surges, of
    either kind, can be combined in one scenario (schema_version 2).

Vehicles are written sorted by depart (SUMO drops out-of-order departs with
only a warning). A sidecar <out>.surge.json next to the output records the
parameters, counts, and the realised per-minute load.

    python tools/gen_surge_routes.py --out dataset/.../surge/x.rou.xml \
        --horizon 7200 --surge-type route --via road_2_4_3 --mult 1.5 \
        --window 600 3600 --gen-seed 20260924
    python tools/gen_surge_routes.py --scenario dataset/.../scenarios/surge_x.json
"""
import argparse
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BASE = "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.rou.xml"
DEFAULT_NET = "dataset/llm_light/Hangzhou/4_4/roadnet.net.xml"
DEFAULT_VIA = ["road_2_4_3"]
DEFAULT_WINDOW = (600.0, 3600.0)
DEFAULT_RAMP = 300.0
HOUR_S = 3600
HOUR2_JITTER_S = 30
SURGE_JITTER_S = 20
SUPPORTED_TOP_LEVEL = {"vType", "vehicle"}
SURGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")
SCHEMA_VERSION = 2

# A <vehicle .../> or <vehicle ...>...</vehicle> block, kept as raw text so base
# vehicles round-trip byte-for-byte.
VEHICLE_BLOCK = re.compile(r"<vehicle\b(?:[^>]*/>|.*?</vehicle>)", re.S)


def fail(message):
    sys.exit(f"gen_surge_routes: {message}")


# --- reading -----------------------------------------------------------------

def parse_vehicle(block):
    try:
        elem = ET.fromstring(block)
    except ET.ParseError as exc:
        fail(f"could not parse vehicle block ({exc}): {block[:120]!r}")
    route = elem.find("route")
    if route is None or not route.get("edges"):
        fail(f"vehicle '{elem.get('id')}' has no inline <route edges=...>; "
             "named routes are not supported")
    try:
        depart = float(elem.get("depart"))
    except (TypeError, ValueError):
        fail(f"vehicle '{elem.get('id')}' has a non-numeric depart "
             f"{elem.get('depart')!r}")
    return {"id": elem.get("id"), "depart": depart, "type": elem.get("type"),
            "edges": route.get("edges").split(), "block": block, "origin": "base",
            "attrib": dict(elem.attrib)}


def read_base(path):
    text = Path(path).read_bytes().decode("utf-8")
    root = ET.fromstring(text)
    unsupported = sorted({child.tag for child in root} - SUPPORTED_TOP_LEVEL)
    if unsupported:
        fail(f"{path}: unsupported top-level elements {unsupported}")
    matches = list(VEHICLE_BLOCK.finditer(text))
    if len(matches) != sum(1 for child in root if child.tag == "vehicle"):
        fail(f"{path}: vehicle block regex disagrees with the XML parser")
    if not matches:
        fail(f"{path}: no vehicles")
    return {
        "vehicles": [parse_vehicle(m.group(0)) for m in matches],
        "vtypes": {c.get("id"): dict(c.attrib) for c in root if c.tag == "vType"},
        "header": text[:matches[0].start()],
        "footer": text[matches[-1].end():],
        "separator": (text[matches[0].end():matches[1].start()]
                      if len(matches) > 1 else "\n"),
    }


# --- cloning -----------------------------------------------------------------

def format_depart(depart):
    depart = float(depart)
    return str(int(depart)) if depart.is_integer() else f"{depart:.2f}"


def clone_vehicle(source, new_id, depart, origin):
    """Copy the source block verbatim, changing only id and depart."""
    tag_end = source["block"].index(">")
    tag = source["block"][:tag_end]
    tag = re.sub(r'(?<=\s)id="[^"]*"', f'id="{new_id}"', tag, count=1)
    tag = re.sub(r'(?<=\s)depart="[^"]*"', f'depart="{format_depart(depart)}"',
                 tag, count=1)
    clone = parse_vehicle(tag + source["block"][tag_end:])
    verbatim = (clone["id"] == new_id
                and clone["depart"] == float(format_depart(depart))
                and clone["type"] == source["type"]
                and clone["edges"] == source["edges"])
    if not verbatim:
        fail(f"clone '{new_id}' of '{source['id']}' is not a verbatim copy "
             f"(type {clone['type']!r} vs {source['type']!r})")
    clone["origin"] = origin
    return clone


def clamp(value, low, high):
    return max(low, min(high, value))


def draw_count(rng, expected):
    """Integer count with the given expectation: floor plus one more with
    probability of the fractional part."""
    whole = math.floor(expected)
    return whole + (1 if rng.random() < expected - whole else 0)


def trapezoid_shape(t, window, ramp):
    """0 outside the window, linear rise to 1 over ramp, hold, linear fall."""
    start, end = window
    if t < start or t > end:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (t - start) / ramp, (end - t) / ramp)


def surge_multiplier(t, window, ramp, mult):
    return 1.0 + (mult - 1.0) * trapezoid_shape(t, window, ramp)


def route_passes_via(edges, via):
    position = 0
    for edge in via:
        if edge not in edges[position:]:
            return False
        position = edges.index(edge, position) + 1
    return True


def make_route_filter(od, via):
    """(description for the sidecar, predicate over a route's edge list)."""
    if od:
        return ({"kind": "od", "edges": list(od)},
                lambda edges: edges[0] == od[0] and edges[-1] == od[1])
    via = list(via or DEFAULT_VIA)
    return {"kind": "via", "edges": via}, lambda edges: route_passes_via(edges, via)


def tile_windows(source, horizon):
    """[(tile_index, tile_start, tile_end)] covering [3600, horizon)."""
    length = source[1] - source[0]
    tiles = []
    k = 1
    while HOUR_S + (k - 1) * length < horizon:
        tile_start = HOUR_S + (k - 1) * length
        tiles.append((k, tile_start, min(tile_start + length, horizon)))
        k += 1
    return tiles


def extend_by_tiling(vehicles, rng, tiling, horizon):
    """Repeat the base vehicles departing inside tiling.source to fill every
    tile after hour 1. Draws happen before the drop check so the RNG sequence
    is a pure function of the seed; the default source [0, 3600) with a 7200 s
    horizon reproduces the original hour-2 copy byte for byte."""
    source_start, source_end = tiling["source"]
    scale = tiling["scale"]
    copies = []
    for k, tile_start, tile_end in tile_windows((source_start, source_end), horizon):
        for v in vehicles:
            if not source_start <= v["depart"] < source_end:
                continue
            for c in range(draw_count(rng, scale)):
                new_id = f"{v['id']}_h{k + 1}" if c == 0 else f"{v['id']}_h{k + 1}_{c}"
                jitter = rng.randint(-HOUR2_JITTER_S, HOUR2_JITTER_S)
                mapped = v["depart"] - source_start + tile_start
                if mapped >= horizon:
                    continue
                depart = clamp(mapped + jitter, tile_start, tile_end - 1)
                copies.append(clone_vehicle(v, new_id, depart, "h2"))
    return copies


def add_surge_clones(vehicles, rng, qualifies, window, ramp, mult, horizon, name=None):
    """Legacy ids (name None) are surge_<baseid>_<k>; a named clone surge
    numbers its clones globally as surge_<name>_<k>."""
    clones = []
    for v in vehicles:
        if not qualifies(v["edges"]):
            continue
        extra = surge_multiplier(v["depart"], window, ramp, mult) - 1.0
        for k in range(draw_count(rng, extra)):
            jitter = rng.randint(-SURGE_JITTER_S, SURGE_JITTER_S)
            depart = clamp(v["depart"] + jitter, 0, horizon - 1)
            new_id = (f"surge_{v['id']}_{k}" if name is None
                      else f"surge_{name}_{len(clones)}")
            clone = clone_vehicle(v, new_id, depart, "surge")
            clone["surge"] = name
            clones.append(clone)
    return clones


# --- injecting new vehicles --------------------------------------------------

def load_net(path):
    try:
        import sumolib
    except ImportError:
        fail("sumolib is required for inject surges and --od routes")
    return sumolib.net.readNet(str(path))


def sumolib_version():
    try:
        import sumolib
        return getattr(sumolib, "__version__", None) or sumolib.__file__
    except ImportError:
        return None


def validate_explicit_route(edges, net):
    for edge_id in edges:
        if not net.hasEdge(edge_id):
            fail(f"route edge '{edge_id}' is not in the network")
    for here, there in zip(edges, edges[1:]):
        outgoing = {e.getID() for e in net.getEdge(here).getOutgoing()}
        if there not in outgoing:
            fail(f"route edge '{there}' does not follow '{here}' "
                 f"(outgoing: {sorted(outgoing)})")


def resolve_route(surge, net):
    if surge.get("route"):
        validate_explicit_route(surge["route"], net)
        return list(surge["route"])
    from_edge, to_edge = surge["od"]
    for edge_id in (from_edge, to_edge):
        if not net.hasEdge(edge_id):
            fail(f"od edge '{edge_id}' is not in the network")
    path, _cost = net.getShortestPath(net.getEdge(from_edge), net.getEdge(to_edge),
                                      vClass="passenger")
    if path is None:
        fail(f"no path from '{from_edge}' to '{to_edge}'")
    return [e.getID() for e in path]


def approach_edge_into(route, intersection, net):
    if not intersection:
        return None
    for edge_id in route:
        if net.getEdge(edge_id).getToNode().getID() == intersection:
            return edge_id
    return None


def inject_rate_per_s(t, surge):
    return surge["rate_veh_h"] / 3600.0 * trapezoid_shape(t, surge["window"], surge["ramp"])


def uniform_departs(surge, horizon):
    """Integer accumulator on the per-second rate: the added load is exactly
    the nominal rate, with no randomness."""
    start, end = surge["window"]
    departs = []
    accumulated = 0.0
    for t in range(int(start), int(end)):
        accumulated += inject_rate_per_s(t, surge)
        while accumulated >= 1.0 - 1e-9:
            if t < horizon:
                departs.append(t)
            accumulated -= 1.0
    return departs


def poisson_departs(surge, rng, horizon):
    """Exponential gaps at the peak rate, thinned by the profile shape."""
    start, end = surge["window"]
    peak_rate = surge["rate_veh_h"] / 3600.0
    departs = []
    t = float(start)
    while True:
        t += rng.expovariate(peak_rate)
        if t >= end:
            return departs
        keep = rng.random() < trapezoid_shape(t, surge["window"], surge["ramp"])
        if keep and int(t) < horizon:
            departs.append(int(t))


def vehicle_attribute_template(base):
    """Ordered attributes of the first base vehicle; id and depart are
    placeholders so injected vehicles mirror the base convention (no type on
    hzreal, type="pkw" plus depart attributes on cfphys)."""
    return list(base["vehicles"][0]["attrib"].items())


def block_layout(base):
    """Whitespace between the opening tag, the <route/> and </vehicle> in the
    base file, so injected blocks are laid out like base blocks."""
    block = base["vehicles"][0]["block"]
    tag_end = block.index(">") + 1
    route_start = block.index("<route")
    route_end = block.index("/>", route_start) + 2
    close = block.index("</vehicle>")
    return {"before_route": block[tag_end:route_start],
            "after_route": block[route_end:close]}


def format_attributes(template, new_id, depart, overrides):
    values = dict(template)
    values.update(overrides or {})
    values["id"] = new_id
    values["depart"] = format_depart(depart)
    template_keys = [key for key, _ in template]
    extra_keys = [key for key in (overrides or {}) if key not in values or key not in template_keys]
    return " ".join(f'{key}="{values[key]}"' for key in template_keys + extra_keys)


def make_injected_vehicle(template, layout, surge, k, depart, route):
    attributes = format_attributes(template, f"surge_{surge['name']}_{k}", depart,
                                   surge.get("depart_attrs"))
    block = (f"<vehicle {attributes}>{layout['before_route']}"
             f'<route edges="{" ".join(route)}"/>{layout["after_route"]}</vehicle>')
    vehicle = parse_vehicle(block)
    vehicle["origin"] = "surge"
    vehicle["surge"] = surge["name"]
    return vehicle


def add_injected_surge(surge, base, horizon, rng):
    template = vehicle_attribute_template(base)
    layout = block_layout(base)
    if surge["arrivals"] == "poisson":
        departs = poisson_departs(surge, rng, horizon)
    else:
        departs = uniform_departs(surge, horizon)
    return [make_injected_vehicle(template, layout, surge, k, depart, surge["route"])
            for k, depart in enumerate(departs)]


# --- output ------------------------------------------------------------------

def write_route_file(out, base, vehicles):
    out = Path(out)
    # out.parent.mkdir(parents=True, exist_ok=True)
    body = base["separator"].join(v["block"] for v in vehicles)
    out.write_bytes((base["header"] + body + base["footer"]).encode("utf-8"))


def sidecar_path_for(out):
    name = Path(out).name
    stem = name[:-len(".rou.xml")] if name.endswith(".rou.xml") else Path(name).stem
    return Path(out).parent / f"{stem}.surge.json"


def default_sumocfg_path(out):
    """<dataset dir>/<stem>.sumocfg for a route file under <dataset dir>/surge/."""
    out = Path(out)
    stem = sidecar_path_for(out).name[:-len(".surge.json")]
    return out.parent.parent / f"{stem}.sumocfg"


def write_sumocfg(path, route_file, horizon, net_name="roadnet.net.xml"):
    path = Path(path)
    route_rel = Path(route_file).resolve().relative_to(path.resolve().parent).as_posix()
    text = ('<?xml version="1.0" encoding="utf-8"?>\n<configuration>\n\t<input>\n'
            f'\t\t<net-file value="{net_name}"/>\n'
            f'\t\t<route-files value="{route_rel}"/>\n\t</input>\n\t<time>\n'
            f'\t\t<begin value="0"/>\n\t\t<end value="{int(horizon)}"/>\n\t</time>\n'
            '</configuration>\n')
    path.write_bytes(text.encode("utf-8"))
    return path


def per_minute_counts(vehicles, horizon, on_route):
    n_minutes = math.ceil(horizon / 60)
    counts = {key: [0] * n_minutes for key in
              ("base_all", "total_all", "base_route", "total_route")}
    for v in vehicles:
        minute = min(int(v["depart"] // 60), n_minutes - 1)
        pre_surge = v["origin"] != "surge"
        counts["total_all"][minute] += 1
        if pre_surge:
            counts["base_all"][minute] += 1
        if on_route(v["edges"]):
            counts["total_route"][minute] += 1
            if pre_surge:
                counts["base_route"][minute] += 1
    counts["minute"] = list(range(n_minutes))
    return counts


def resolved_type_histogram(vehicles):
    histogram = {}
    for v in vehicles:
        resolved = v["type"] or "DEFAULT_VEHTYPE"
        histogram[resolved] = histogram.get(resolved, 0) + 1
    return histogram


def legacy_parameters(spec, clone):
    """The v1 'parameters' block, derived from the first clone surge (or the
    v1 defaults when the scenario has none)."""
    route_filter = clone.get("od") or clone.get("via")
    return {
        "base": spec["base"], "out": spec["out"], "horizon": spec["horizon"],
        "hour2_scale": spec["tiling"]["scale"],
        "surge_type": "route" if clone.get("scope") == "route" else "network",
        "window": list(clone["window"]), "ramp": clone["ramp"], "mult": clone["mult"],
        "od": list(clone["od"]) if clone.get("od") else None,
        "via": list(clone["via"]) if clone.get("via") else None,
        "gen_seed": spec["gen_seed"],
    }


def default_clone_surge():
    return {"name": None, "mode": "clone", "scope": "network", "mult": 1.0,
            "window": list(DEFAULT_WINDOW), "ramp": DEFAULT_RAMP, "od": None, "via": None}


def added_per_minute(vehicles, horizon):
    n_minutes = math.ceil(horizon / 60)
    counts = [0] * n_minutes
    for v in vehicles:
        counts[min(int(v["depart"] // 60), n_minutes - 1)] += 1
    return counts


def surge_sidecar_entry(surge, added, horizon):
    start, end = surge["window"]
    hold = (start + surge["ramp"], end - surge["ramp"])
    in_hold = sum(1 for v in added if hold[0] <= v["depart"] <= hold[1])
    hold_length = max(hold[1] - hold[0], 0)
    clone_keys = ("mult", "scope", "via", "od")
    inject_keys = ("rate_veh_h", "arrivals", "route", "entry_edge", "approach_edge",
                   "intersection", "od", "depart_attrs")
    keys = ("name", "mode", "window", "ramp") + (clone_keys if surge["mode"] == "clone"
                                                  else inject_keys)
    entry = {key: surge.get(key) for key in keys}
    entry["added_count"] = len(added)
    entry["realised_rate_hold_veh_h"] = (round(in_hold / hold_length * 3600, 1)
                                         if hold_length else None)
    entry["per_minute_added"] = added_per_minute(added, horizon)
    return entry


def build_sidecar(spec, base, vehicles, tiles, added_by_surge):
    clone_surges = [s for s in spec["surges"] if s["mode"] == "clone"]
    legacy = clone_surges[0] if clone_surges else default_clone_surge()
    route_filter, on_route = make_route_filter(legacy.get("od"), legacy.get("via"))
    population = "route" if legacy.get("scope") == "route" else "all"
    minutes = per_minute_counts(vehicles, spec["horizon"], on_route)
    minutes["realised_multiplier"] = [
        round(total / count, 3) if count else None
        for count, total in zip(minutes[f"base_{population}"],
                                minutes[f"total_{population}"])]
    minutes["expected_multiplier"] = [
        round(surge_multiplier(m * 60 + 30, tuple(legacy["window"]), legacy["ramp"],
                               legacy["mult"]), 3)
        for m in minutes["minute"]]

    def realised_over(start, end):
        """total / base over the minutes lying fully inside [start, end]."""
        inside = [m for m in minutes["minute"] if start <= m * 60 and (m + 1) * 60 <= end]
        base_count = sum(minutes[f"base_{population}"][m] for m in inside)
        total = sum(minutes[f"total_{population}"][m] for m in inside)
        return round(total / base_count, 3) if base_count else None

    start, end = legacy["window"]
    surge_total = sum(len(added) for added in added_by_surge)
    return {
        "generator": "tools/gen_surge_routes.py",
        "schema_version": SCHEMA_VERSION,
        "parameters": legacy_parameters(spec, legacy),
        "route_filter": route_filter,
        "surge_population": population,
        "base_vehicle_count": len(base["vehicles"]),
        "hour2_added_count": len(tiles),
        "surge_added_count": surge_total,
        "total_vehicle_count": len(vehicles),
        "realised_multiplier_window": realised_over(start, end),
        "realised_multiplier_hold": realised_over(start + legacy["ramp"], end - legacy["ramp"]),
        "vtypes_defined": base["vtypes"],
        "resolved_vehicle_types": {"base": resolved_type_histogram(base["vehicles"]),
                                   "output": resolved_type_histogram(vehicles)},
        "tiling": {"source": list(spec["tiling"]["source"]), "scale": spec["tiling"]["scale"],
                   "tile_count": len(tile_windows(spec["tiling"]["source"], spec["horizon"])),
                   "added_count": len(tiles)},
        "intersection": spec.get("intersection"),
        "surges": [surge_sidecar_entry(surge, added, spec["horizon"])
                   for surge, added in zip(spec["surges"], added_by_surge)],
        "sumolib_version": sumolib_version() if spec.get("net_needed") else None,
        "per_minute": minutes,
    }


# --- scenario specs ------------------------------------------------------------

def legacy_generator_to_spec(generator):
    """A v1 'generator' block (argparse dest names) as a v2 spec with one
    clone surge. Keys missing from the block take the CLI defaults."""
    g = dict(generator)
    horizon = int(g.get("horizon", HOUR_S))
    window = g.get("window") or [DEFAULT_WINDOW[0], min(DEFAULT_WINDOW[1], horizon)]
    clone = {"name": None, "mode": "clone",
             "scope": g.get("surge_type", "network"),
             "mult": float(g.get("mult", 1.0)),
             "window": [float(w) for w in window],
             "ramp": float(g.get("ramp", DEFAULT_RAMP)),
             "od": g.get("od"), "via": g.get("via")}
    return {"base": g.get("base", DEFAULT_BASE), "out": g.get("out"),
            "net": g.get("net", DEFAULT_NET), "horizon": horizon,
            "tiling": {"source": [0, HOUR_S], "scale": float(g.get("hour2_scale", 1.0))},
            "gen_seed": int(g.get("gen_seed", 0)), "intersection": None,
            "surges": [clone], "sumocfg": None}


def spec_from_scenario(scenario, scenario_path):
    if "generator" in scenario:
        spec = legacy_generator_to_spec(scenario["generator"])
        spec["sumocfg"] = scenario.get("sumocfg")
        return spec
    surges = []
    for raw in scenario.get("surges", []):
        surge = dict(raw)
        surge.setdefault("mode", "inject")
        surge.setdefault("ramp", 0.0)
        surge.setdefault("arrivals", "uniform")
        if surge["mode"] == "clone":
            surge.setdefault("scope", "route" if (surge.get("od") or surge.get("via")) else "network")
        surges.append(surge)
    tiling = dict(scenario.get("tiling") or {})
    return {"base": scenario.get("base", DEFAULT_BASE),
            "out": scenario.get("route_file"),
            "net": scenario.get("net", DEFAULT_NET),
            "horizon": int(scenario.get("horizon") or scenario.get("simulation_steps") or HOUR_S),
            "tiling": {"source": list(tiling.get("source", [0, HOUR_S])),
                       "scale": float(tiling.get("scale", 1.0))},
            "gen_seed": int(scenario.get("gen_seed", 0)),
            "intersection": scenario.get("intersection"),
            "surges": surges,
            "sumocfg": scenario.get("sumocfg")}


def load_scenario(path):
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    return spec_from_scenario(scenario, path)


def spec_from_args(args):
    clone = {"name": None, "mode": "clone", "scope": args.surge_type, "mult": args.mult,
             "window": [float(w) for w in args.window], "ramp": args.ramp,
             "od": args.od, "via": args.via}
    return {"base": args.base, "out": args.out, "net": args.net, "horizon": args.horizon,
            "tiling": {"source": [float(s) for s in args.tile_source],
                       "scale": args.hour2_scale},
            "gen_seed": args.gen_seed, "intersection": None, "surges": [clone],
            "sumocfg": None}


def validate_spec(spec):
    if not spec.get("out"):
        fail("--out (or the scenario's route_file) is required")
    if spec["horizon"] < 1:
        fail("--horizon must be >= 1")
    source = spec["tiling"]["source"]
    if not 0 <= source[0] < source[1] <= HOUR_S:
        fail(f"--tile-source must satisfy 0 <= START < END <= {HOUR_S}, got {source}")
    if spec["tiling"]["scale"] < 0:
        fail("--hour2-scale must be >= 0")
    names = [s["name"] for s in spec["surges"] if s["name"] is not None]
    if len(names) != len(set(names)):
        fail(f"surge names must be unique: {names}")
    for surge in spec["surges"]:
        validate_surge(surge, spec["horizon"])


def validate_surge(surge, horizon):
    label = surge["name"] or "clone"
    if surge["name"] is not None and not SURGE_NAME.match(surge["name"]):
        fail(f"surge name '{surge['name']}' must match {SURGE_NAME.pattern}")
    window = surge["window"]
    if not 0 <= window[0] < window[1] <= horizon:
        fail(f"surge '{label}': window must satisfy 0 <= START < END <= horizon, got {window}")
    if surge["ramp"] < 0:
        fail(f"surge '{label}': ramp must be >= 0")
    if window[1] - window[0] < 2 * surge["ramp"]:
        print(f"warning: surge '{label}': window shorter than 2 x ramp, peak is never reached")
    if surge["mode"] == "clone":
        if surge["mult"] < 1.0:
            fail(f"surge '{label}': mult must be >= 1.0: base vehicles are never thinned")
        if surge["scope"] == "network" and (surge.get("od") or surge.get("via")):
            fail(f"surge '{label}': od/via only apply to a route-scoped clone surge")
    elif surge["mode"] == "inject":
        if surge["name"] is None:
            fail("inject surges need a name")
        if bool(surge.get("route")) == bool(surge.get("od")):
            fail(f"surge '{label}': give exactly one of route or od")
        if not surge.get("rate_veh_h") or surge["rate_veh_h"] <= 0:
            fail(f"surge '{label}': rate_veh_h must be > 0")
        if surge["arrivals"] not in ("uniform", "poisson"):
            fail(f"surge '{label}': arrivals must be uniform or poisson")
    else:
        fail(f"surge '{label}': unknown mode {surge['mode']!r}")


# --- CLI ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="surge scenario JSON: a v1 'generator' "
                        "block supplies defaults for the flags below; a v2 file "
                        "(with 'surges') is authoritative")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--net", default=DEFAULT_NET,
                        help="network file, needed for inject surges and --od routes")
    parser.add_argument("--out", help="output .rou.xml path")
    parser.add_argument("--horizon", type=int, default=HOUR_S)
    parser.add_argument("--tile-source", nargs=2, type=float, default=[0, HOUR_S],
                        metavar=("START", "END"),
                        help="slice of hour 1 repeated to fill horizons past 3600 s")
    parser.add_argument("--hour2-scale", type=float, default=1.0,
                        help="expected copies per source vehicle in every tile")
    parser.add_argument("--surge-type", choices=["network", "route"], default="network")
    parser.add_argument("--window", nargs=2, type=float, metavar=("START", "END"),
                        help="surge window in seconds (default 600 3600)")
    parser.add_argument("--ramp", type=float, default=DEFAULT_RAMP)
    parser.add_argument("--mult", type=float, default=1.0,
                        help="peak multiplier inside the window (1.5 = +50%%)")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--od", nargs=2, metavar=("EDGE_FROM", "EDGE_TO"))
    route.add_argument("--via", nargs="+", metavar="EDGE",
                       help=f"route must contain these edges in order (default {DEFAULT_VIA})")
    parser.add_argument("--gen-seed", type=int, default=0)
    parser.add_argument("--write-sumocfg", action="store_true",
                        help="also write the per-scenario sumocfg next to the dataset")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    scenario = None
    if args.scenario:
        scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    if scenario is not None and "generator" not in scenario:
        spec = spec_from_scenario(scenario, args.scenario)
        if args.out:
            spec["out"] = args.out
    else:
        if scenario is not None:
            parser.set_defaults(**scenario["generator"])
            args = parser.parse_args(argv)
        if args.window is None:
            args.window = [DEFAULT_WINDOW[0], min(DEFAULT_WINDOW[1], args.horizon)]
        spec = spec_from_args(args)
        if scenario is not None:
            spec["sumocfg"] = scenario.get("sumocfg")
    spec["write_sumocfg"] = args.write_sumocfg
    validate_spec(spec)
    return spec


def resolve_inject_routes(spec):
    inject = [s for s in spec["surges"] if s["mode"] == "inject"] 
    spec["net_needed"] = bool(inject)
    if not inject:
        return
    net = load_net(spec["net"])
    for surge in inject:
        surge["route"] = resolve_route(surge, net)
        surge["entry_edge"] = surge["route"][0]
        intersection = surge.get("intersection") or spec.get("intersection")
        surge["intersection"] = intersection
        surge["approach_edge"] = approach_edge_into(surge["route"], intersection, net)
        if intersection and surge["approach_edge"] is None:
            fail(f"surge '{surge['name']}': route never enters {intersection}")


def add_surge(surge, vehicles, base, spec, rng):
    if surge["mode"] == "clone":
        _filter, on_route = make_route_filter(surge.get("od"), surge.get("via"))
        qualifies = on_route if surge["scope"] == "route" else (lambda edges: True)
        return add_surge_clones(vehicles, rng, qualifies, tuple(surge["window"]),
                                surge["ramp"], surge["mult"], spec["horizon"],
                                surge["name"])
    own_rng = random.Random(f"{spec['gen_seed']}/{surge['name']}")
    return add_injected_surge(surge, base, spec["horizon"], own_rng)


def describe_surge(surge, added):
    start, end = surge["window"]
    if surge["mode"] == "clone":
        return (f"{surge['scope']} clone surge x{surge['mult']} over "
                f"{start:.0f}-{end:.0f} s (ramp {surge['ramp']:.0f} s): {len(added)} vehicles")
    return (f"inject {surge['name']} {surge['rate_veh_h']:g} veh/h on "
            f"{surge['route'][0]}..{surge['route'][-1]} over {start:.0f}-{end:.0f} s "
            f"(ramp {surge['ramp']:.0f} s): {len(added)} vehicles")


def generate(spec):
    base = read_base(spec["base"])
    resolve_inject_routes(spec)
    rng = random.Random(spec["gen_seed"])

    vehicles = list(base["vehicles"])
    tiles = extend_by_tiling(vehicles, rng, spec["tiling"], spec["horizon"])
    vehicles += tiles
    added_by_surge = []
    for surge in spec["surges"]:
        added = add_surge(surge, vehicles, base, spec, rng)
        added_by_surge.append(added)
    for added in added_by_surge:
        vehicles += added
    vehicles.sort(key=lambda v: v["depart"])  # stable: base order kept on ties

    write_route_file(spec["out"], base, vehicles)
    sidecar = build_sidecar(spec, base, vehicles, tiles, added_by_surge)
    sidecar_path = sidecar_path_for(spec["out"])
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    if spec.get("write_sumocfg"):
        sumocfg = spec.get("sumocfg") or default_sumocfg_path(spec["out"])
        write_sumocfg(sumocfg, spec["out"], spec["horizon"])
        print(f"wrote {sumocfg}")

    surge_total = sum(len(a) for a in added_by_surge)
    print(f"wrote {spec['out']}: {len(base['vehicles'])} base + {len(tiles)} tiled "
          f"+ {surge_total} surge = {len(vehicles)} vehicles | horizon {spec['horizon']} s | "
          f"realised x{sidecar['realised_multiplier_hold']} on the hold, "
          f"x{sidecar['realised_multiplier_window']} over the window | sidecar {sidecar_path}")
    for surge, added in zip(spec["surges"], added_by_surge):
        print("  " + describe_surge(surge, added))
    return sidecar


def main(argv=None):
    return generate(parse_args(argv))


if __name__ == "__main__":
    main()
