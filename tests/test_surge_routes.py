"""Tests for the demand-surge route generator (tools/gen_surge_routes.py) and
the per-second network time series (network_timeseries.csv).

    python -m pytest tests/test_surge_routes.py                 # everything
    python -m pytest tests/test_surge_routes.py -k "not smoke"  # fast subset
    python tests/test_surge_routes.py [--skip-smoke]            # without pytest

Every test writes what it generates to tests/artifacts/surge/<test_name>/
(wiped and rewritten on each run) and prints that path. test_vtype and
test_edge_series and test_smoke need SUMO on PATH; test_smoke runs FixedTime
for 7200 s on the S2 route-surge config and on its control (SUMO only, no LLM,
~20 min). The schema v2 tests regenerate the shipped scenarios, the crossing
template and its steady control, and exercise tiling, injected surges, the
per-edge recorder series and the analysis helpers.
"""
import json
import shutil
import subprocess
import sys
import traceback
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import gen_surge_routes as gen  # noqa: E402
from build_results import config_alias_of  # noqa: E402
from configurations import INTERSECTION_CONFIGS  # noqa: E402
from experiments import CONFIGS, SURGE_SCENARIOS, make_combo  # noqa: E402
from run_matrix import build_command  # noqa: E402
from utils import surge_analysis as sa  # noqa: E402
from utils.surge_analysis import excess_over_control, read_timeseries, relative_rise  # noqa: E402,F401

BASE = REPO / gen.DEFAULT_BASE
CFPHYS_BASE = REPO / "dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_cfphys.rou.xml"
NET = REPO / "dataset/llm_light/Hangzhou/4_4/roadnet.net.xml"
SHIPPED = ["hzreal_h2", "hzreal_s1_net150", "hzreal_s2_route150_2_3",
           "hzreal_h2_steady", "hzreal_cross_2_3"]


def scenario_path(alias):
    return REPO / SURGE_SCENARIOS[alias]["path"]


CROSS_TEMPLATE = scenario_path("hzreal_cross_2_3")
STEADY_CONTROL = scenario_path("hzreal_h2_steady")
CROSS_ROUTES = {
    "we3": ("road_0_3_0", "road_1_3_0", "road_2_3_0", "road_3_3_0", "road_4_3_0"),
    "ns2": ("road_2_5_3", "road_2_4_3", "road_2_3_3", "road_2_2_3", "road_2_1_3"),
}
ARTIFACTS = REPO / "tests/artifacts/surge"
SURGE_ALIAS = "hzreal_s2_route150_2_3"
CONTROL_ALIAS = "hzreal_h2"
VIA_EDGE = "road_2_4_3"
WINDOW = (600, 3600)
HORIZON = 7200
ROUTE_SURGE_FLAGS = ["--horizon", "7200", "--surge-type", "route", "--via", VIA_EDGE,
                     "--mult", "1.5", "--window", "600", "3600", "--gen-seed", "20260924"]


# --- helpers -----------------------------------------------------------------

def artifact_dir(test_name):
    path = ARTIFACTS / test_name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def report(path):
    print(f"[artifacts] {path}")


def generate(out, *flags):
    return gen.main(["--base", str(BASE), "--out", str(out), *flags])


def vehicles_of(path):
    """{id: (depart, route edges, type attribute)} for every <vehicle>."""
    root = ET.parse(path).getroot()
    return {v.get("id"): (float(v.get("depart")),
                          tuple(v.find("route").get("edges").split()),
                          v.get("type"))
            for v in root.iter("vehicle")}


def header_of(path):
    text = path.read_bytes().decode("utf-8")
    return text[:text.index("<vehicle")]


def require_sumo():
    if shutil.which("sumo") is None:
        raise unittest.SkipTest("sumo is not on PATH")


# --- generator ---------------------------------------------------------------

def test_parity():
    out_dir = artifact_dir("test_parity")
    out = out_dir / "parity.rou.xml"
    sidecar = generate(out, "--horizon", "3600", "--mult", "1.0")
    assert vehicles_of(out) == vehicles_of(BASE)
    assert out.read_bytes() == BASE.read_bytes(), "base vehicles must round-trip verbatim"
    assert sidecar["hour2_added_count"] == 0 and sidecar["surge_added_count"] == 0
    report(out_dir)


def test_extension():
    out_dir = artifact_dir("test_extension")
    out = out_dir / "h2.rou.xml"
    sidecar = generate(out, "--horizon", "7200", "--mult", "1.0", "--gen-seed", "1")
    base = vehicles_of(BASE)
    got = vehicles_of(out)
    assert len(got) == 2 * len(base)
    for vid, vehicle in base.items():
        assert got[vid] == vehicle, f"base vehicle {vid} was altered"
    hour2 = {vid: v for vid, v in got.items() if vid.endswith("_h2")}
    assert len(hour2) == len(base)
    for vid, (depart, edges, vtype) in hour2.items():
        source_depart, source_edges, source_type = base[vid[:-len("_h2")]]
        assert 3600 <= depart < 7200, f"{vid} departs at {depart}"
        assert abs(depart - (source_depart + 3600)) <= 30, f"{vid} jitter out of range"
        assert edges == source_edges and vtype == source_type, f"{vid} route/type differs"
    assert sidecar["hour2_added_count"] == len(base) and sidecar["surge_added_count"] == 0
    report(out_dir)


def test_route_filter():
    out_dir = artifact_dir("test_route_filter")
    out = out_dir / "s2.rou.xml"
    sidecar = generate(out, *ROUTE_SURGE_FLAGS)
    base = vehicles_of(BASE)
    got = vehicles_of(out)
    for vid, vehicle in base.items():
        assert got[vid] == vehicle, f"base vehicle {vid} was altered"
    surge = {vid: v for vid, v in got.items() if vid.startswith("surge_")}
    assert surge, "no surge vehicles were generated"
    for vid, (depart, edges, _type) in surge.items():
        assert VIA_EDGE in edges, f"{vid} does not pass {VIA_EDGE}"
        source_id = vid[len("surge_"):].rsplit("_", 1)[0]
        assert got[source_id][1] == edges, f"{vid} route differs from {source_id}"
        assert WINDOW[0] - 20 <= depart <= WINDOW[1] + 20, f"{vid} departs outside the window"
    minutes = sidecar["per_minute"]
    window_minutes = range(WINDOW[0] // 60, WINDOW[1] // 60)
    assert all(minutes["total_route"][m] >= minutes["base_route"][m] for m in window_minutes)
    assert any(minutes["total_route"][m] > minutes["base_route"][m] for m in window_minutes)
    # +-20 s jitter can spill one minute past either edge; beyond that, nothing.
    untouched = list(range(0, WINDOW[0] // 60 - 1)) + list(range(WINDOW[1] // 60 + 1, HORIZON // 60))
    assert all(minutes["total_all"][m] == minutes["base_all"][m] for m in untouched)
    assert sidecar["surge_added_count"] == len(surge)
    report(out_dir)


def test_determinism():
    out_dir = artifact_dir("test_determinism")
    run_a, run_b, run_c = (out_dir / f"run_{k}.rou.xml" for k in "abc")
    generate(run_a, *ROUTE_SURGE_FLAGS)
    generate(run_b, *ROUTE_SURGE_FLAGS)
    assert run_a.read_bytes() == run_b.read_bytes(), "same --gen-seed must give identical bytes"
    generate(run_c, *ROUTE_SURGE_FLAGS[:-1], "1")
    assert run_c.read_bytes() != run_a.read_bytes(), "a different --gen-seed must change the output"
    report(out_dir)


def sample_vtypes_in_sumo(route_file, until_s=3900, every_s=100):
    """Load the generated routes in SUMO and record, per vehicle kind, the
    (typeID, length, minGap) every sampled vehicle actually resolved to."""
    import traci
    seen = {"base": {}, "h2": {}, "surge": {}}
    traci.start(["sumo", "-n", str(NET), "-r", str(route_file), "--time-to-teleport", "-1",
                 "--no-step-log", "true", "--no-warnings", "true"])
    try:
        while traci.simulation.getTime() < until_s:
            traci.simulationStep()
            if int(traci.simulation.getTime()) % every_s:
                continue
            for veh in traci.vehicle.getIDList():
                kind = "surge" if veh.startswith("surge_") else "h2" if "_h2" in veh else "base"
                key = (f"{traci.vehicle.getTypeID(veh)} length={traci.vehicle.getLength(veh)} "
                       f"minGap={traci.vehicle.getMinGap(veh)}")
                seen[kind][key] = seen[kind].get(key, 0) + 1
    finally:
        traci.close()
    return seen


def test_vtype():
    require_sumo()
    out_dir = artifact_dir("test_vtype")
    out = out_dir / "s2.rou.xml"
    sidecar = generate(out, *ROUTE_SURGE_FLAGS)
    assert header_of(out) == header_of(BASE), "vType definitions must be copied verbatim"
    assert (set(sidecar["resolved_vehicle_types"]["output"])
            == set(sidecar["resolved_vehicle_types"]["base"]))
    resolution = sample_vtypes_in_sumo(out)
    (out_dir / "vtype_resolution.json").write_text(json.dumps(resolution, indent=2))
    base_types = set(resolution["base"])
    assert len(base_types) == 1, f"base vehicles resolve to several types: {base_types}"
    assert all("length=5.0 minGap=2.5" in key for key in base_types), base_types
    for kind in ("h2", "surge"):
        assert resolution[kind], f"no {kind} vehicles were observed in SUMO"
        assert set(resolution[kind]) == base_types, (
            f"{kind} clones resolve to {set(resolution[kind])}, base to {base_types}")
    report(out_dir)


# --- schema v2: reproducibility, tiling, injected surges ------------------------

def scenario_spec(path, out, **overrides):
    """A validated spec from a scenario JSON, redirected to the artifact dir."""
    spec = gen.load_scenario(path)
    spec["out"] = str(out)
    spec["write_sumocfg"] = False
    spec.update(overrides)
    gen.validate_spec(spec)
    return spec


def test_shipped_reproducible():
    out_dir = artifact_dir("test_shipped_reproducible")
    for alias in SHIPPED:
        scenario = json.loads(scenario_path(alias).read_text(encoding="utf-8"))
        out = out_dir / Path(scenario["route_file"]).name
        gen.main(["--scenario", str(scenario_path(alias)), "--out", str(out)])
        shipped = (REPO / scenario["route_file"]).read_bytes()
        assert out.read_bytes() == shipped, f"{alias} no longer regenerates byte-for-byte"
    report(out_dir)


def test_tiling():
    out_dir = artifact_dir("test_tiling")
    out = out_dir / "steady.rou.xml"
    sidecar = generate(out, "--horizon", "9000", "--tile-source", "1200", "3600",
                       "--gen-seed", "7")
    base = vehicles_of(BASE)
    got = vehicles_of(out)
    tiles = {vid: v for vid, v in got.items() if "_h" in vid}
    assert sidecar["tiling"]["tile_count"] == 3 and sidecar["hour2_added_count"] == len(tiles)
    suffixes = {vid.rsplit("_h", 1)[1] for vid in tiles}
    assert suffixes == {"2", "3", "4"}, suffixes
    for vid, (depart, edges, _type) in tiles.items():
        source_id, tile = vid.rsplit("_h", 1)
        source_depart, source_edges, _ = base[source_id]
        assert 1200 <= source_depart < 3600, f"{vid} copied from outside the steady slice"
        tile_start = 3600 + (int(tile) - 2) * 2400
        expected = source_depart - 1200 + tile_start
        assert 3600 <= depart < 9000 and abs(depart - expected) <= 30, f"{vid} at {depart}"
        assert edges == source_edges
    # A partial last tile keeps only the departs that fit inside the horizon;
    # like hour 2 at 7199, only the jittered tail (+-30 s) is clamped onto its
    # last second, never whole source vehicles.
    last_tile = [v[0] for vid, v in tiles.items() if vid.endswith("_h4")]
    source_in_tail = sum(1 for d, _e, _t in base.values() if 1800 - 30 <= d < 1800)
    assert max(last_tile) < 9000
    assert sum(1 for d in last_tile if d == 8999) <= source_in_tail
    report(out_dir)


def injected_of(path):
    return {vid: v for vid, v in vehicles_of(path).items()
            if vid.startswith("surge_") and vid[len("surge_")].isalpha()}


def test_multi_surge_inject():
    out_dir = artifact_dir("test_multi_surge_inject")
    out = out_dir / "cross.rou.xml"
    sidecar = gen.generate(scenario_spec(CROSS_TEMPLATE, out))
    control = out_dir / "control.rou.xml"
    gen.generate(scenario_spec(STEADY_CONTROL, control))
    got = vehicles_of(out)
    injected = injected_of(out)
    background = {vid: v for vid, v in got.items() if vid not in injected}
    assert background == vehicles_of(control), "base + tiled vehicles must match the control"

    entries = {e["name"]: e for e in sidecar["surges"]}
    for name, route in CROSS_ROUTES.items():
        entry = entries[name]
        window = entry["window"]
        expected_count = round(entry["rate_veh_h"] * (window[1] - window[0]) / 3600)
        assert tuple(entry["route"]) == route, f"{name}: {entry['route']}"
        mine = {vid: v for vid, v in injected.items() if vid.startswith(f"surge_{name}_")}
        assert len(mine) == expected_count == entry["added_count"], (name, len(mine))
        assert {int(vid.rsplit("_", 1)[1]) for vid in mine} == set(range(expected_count))
        for vid, (depart, edges, vtype) in mine.items():
            assert window[0] <= depart < window[1] and depart < 9000, f"{vid} at {depart}"
            assert edges == route and vtype is None, f"{vid}: {edges} type={vtype}"
    assert entries["ns2"]["approach_edge"] == "road_2_4_3"
    assert entries["we3"]["approach_edge"] == "road_1_3_0"
    assert header_of(out) == header_of(BASE)
    report(out_dir)


def test_inject_mirrors_cfphys_attributes():
    out_dir = artifact_dir("test_inject_mirrors_cfphys_attributes")
    out = out_dir / "cross_cfphys.rou.xml"
    gen.generate(scenario_spec(CROSS_TEMPLATE, out, base=str(CFPHYS_BASE)))
    text = out.read_text(encoding="utf-8")
    block = text[text.index('<vehicle id="surge_we3_0"'):]
    block = block[:block.index("</vehicle>")]
    assert 'type="pkw"' in block and 'departLane="best"' in block and 'departSpeed="max"' in block
    assert all(v[2] == "pkw" for v in injected_of(out).values())
    report(out_dir)


def test_inject_rng_isolation():
    out_dir = artifact_dir("test_inject_rng_isolation")
    outputs = []
    for ns2_rate in (600, 900):
        out = out_dir / f"poisson_ns2_{ns2_rate}.rou.xml"
        spec = scenario_spec(CROSS_TEMPLATE, out)
        for surge in spec["surges"]:
            surge["arrivals"] = "poisson"
            if surge["name"] == "ns2":
                surge["rate_veh_h"] = ns2_rate
        gen.generate(spec)
        outputs.append(vehicles_of(out))
    we3 = [{vid: v for vid, v in got.items() if vid.startswith("surge_we3_")} for got in outputs]
    ns2 = [{vid: v for vid, v in got.items() if vid.startswith("surge_ns2_")} for got in outputs]
    assert we3[0] == we3[1], "changing ns2 must not move we3's Poisson departs"
    assert ns2[0] != ns2[1] and len(ns2[1]) > len(ns2[0])
    report(out_dir)


def test_auto_register():
    for alias, info in SURGE_SCENARIOS.items():
        assert CONFIGS[alias] == info["sumocfg"], alias
        assert (REPO / info["sumocfg"]).exists(), f"{alias}: missing {info['sumocfg']}"
        assert config_alias_of(Path(info["sumocfg"]).name) == alias
    assert SURGE_SCENARIOS["hzreal_cross_2_3"]["control_alias"] == "hzreal_h2_steady"


def test_analysis_synthetic():
    times = list(range(0, 3000))
    control = [2.0] * 3000
    # Queue climbs through the hold (600-1800), then decays linearly to the control.
    halting = [2.0 + max(0.0, min(t - 600, 1200) / 40.0) - max(0.0, (t - 1800) / 20.0)
               for t in times]
    halting = [max(h, 2.0) for h in halting]
    pending = [0.0] * 3000
    over = sa.oversaturated(times, halting, pending, (600, 1800), smooth_s=1)
    assert over["oversaturated"] and over["queue_growing"] and not over["pending_growing"]
    calm = sa.oversaturated(times, control, pending, (600, 1800), smooth_s=1)
    assert not calm["oversaturated"]
    recovery = sa.recovery_time(times, halting, control, 1800, smooth_s=1,
                                tolerance=0.10, slack=1.0, sustain_s=100)
    # 2 + 30 - (t-1800)/20 <= 2*1.1 + 1 -> t >= 1800 + 20*(32-3.2) = 2376
    assert recovery["recovery_time_s"] == 576, recovery
    never = sa.recovery_time(times, [50.0] * 3000, control, 1800, smooth_s=1)
    assert never["recovery_time_s"] is None
    rows = [{"t": float(t), "arrived_cum": float(t // 10)} for t in times]
    assert sa.cumulative_delta(rows, "arrived_cum", 600, 1800) == 120
    estimate = sa.queueing_estimate(900, 300, 1200, 1200)
    assert estimate == {"backlog_vehicles": 100.0, "clearance_time_s": 600.0,
                        "suggested_recovery_window_s": 1200.0}


# --- recorder: edge_timeseries.csv must not disturb the existing outputs ------

def run_recorder(run_dir, steps, edge_series, blockage=None):
    from sumo_env import SumoEnv
    from utils.blockage_manager import BlockageManager, load_scenario
    from utils.metrics_recorder import MetricsRecorder
    manager = BlockageManager(load_scenario(blockage)["blockages"]) if blockage else None
    env = SumoEnv(sumo_config=str(REPO / CONFIGS["hzreal"]),
                  intersection_config=INTERSECTION_CONFIGS["three_lane"],
                  output_dir=run_dir, seed=1, blockage_manager=manager)
    recorder = MetricsRecorder(run_dir=run_dir, verbose=False, blockage_manager=manager,
                               edge_series=edge_series)
    env.start_simulation()
    recorder.record_initial_load()
    for step in range(1, steps + 1):
        env.step()
        recorder.record_step_summary(step)
    env.close()
    return recorder.get_final_summary()


def test_edge_series():
    require_sumo()
    out_dir = artifact_dir("test_edge_series")
    steps = 300
    run_recorder(out_dir / "off", steps, edge_series=False)
    run_recorder(out_dir / "on", steps, edge_series=True)
    for name in ("network_timeseries.csv", "step_summaries.jsonl"):
        assert (out_dir / "on" / name).read_bytes() == (out_dir / "off" / name).read_bytes(), name
    assert not (out_dir / "off" / "edge_timeseries.csv").exists()
    assert not (out_dir / "on" / "surge_timeseries.csv").exists()
    check_edge_invariants(out_dir / "on")

    c3 = REPO / "dataset/llm_light/Hangzhou/4_4/scenarios/c3_through_north_2_3_1200s.json"
    run_recorder(out_dir / "c3", 560, edge_series=True, blockage=str(c3))
    check_edge_invariants(out_dir / "c3")
    report(out_dir)


def check_edge_invariants(run_dir):
    network = read_timeseries(run_dir / "network_timeseries.csv")
    edges = sa.read_edge_timeseries(run_dir / "edge_timeseries.csv")
    assert len(network) == len(edges)
    for row in network:
        edge_row = edges[row["t"]]
        halting = sum(v for k, v in edge_row.items() if k.startswith("halting:"))
        vehicles = sum(v for k, v in edge_row.items() if k.startswith("vehicles:"))
        assert halting == row["halting_total"] and vehicles == row["vehicles_in_network"], row


# --- smoke: FixedTime 7200 s on the shipped S2 config and its control ----------

def run_fixedtime(alias, dest):
    """Launch FixedTime through the run_matrix command builder (so the alias is
    exercised end-to-end) and copy the record files into dest."""
    logs_dir = dest / "runs"
    combo = make_combo("fixedtime", alias, 1, "none", HORIZON, "three_lane", {})
    command = build_command(combo, "smoke", f"ft_{alias}", logs_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "runner_stdout.txt", "w") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    run_dir = sorted((logs_dir / "smoke").glob(f"ft_{alias}_*"))[-1]
    sumocfg = REPO / CONFIGS[alias]
    shutil.copy(sumocfg, dest / sumocfg.name)
    for name in ("network_timeseries.csv", "final_summary.json"):
        shutil.copy(run_dir / name, dest / name)
    return {"run_dir": run_dir,
            "rows": read_timeseries(run_dir / "network_timeseries.csv"),
            "summary": json.loads((run_dir / "final_summary.json").read_text())}


def test_smoke():
    require_sumo()
    out_dir = artifact_dir("test_smoke")
    surge = run_fixedtime(SURGE_ALIAS, out_dir / "surge")
    control = run_fixedtime(CONTROL_ALIAS, out_dir / "control")

    rows = surge["rows"]
    by_t = {int(row["t"]): row for row in rows}
    control_by_t = {int(row["t"]): row for row in control["rows"]}
    peak = surge["summary"]["peak_vehicles_in_network"]
    peak_t = surge["summary"]["peak_vehicles_in_network_time_s"]
    in_window = lambda row: WINDOW[0] <= row["t"] <= WINDOW[1]  # noqa: E731
    report_data = {
        "rows": len(rows),
        "peak_vehicles_in_network": peak,
        "peak_time_s": peak_t,
        "vehicles_in_network_t3600": by_t[3600]["vehicles_in_network"],
        "vehicles_in_network_t7199": by_t[7199]["vehicles_in_network"],
        "control_peak_vehicles_in_network": control["summary"]["peak_vehicles_in_network"],
        "control_peak_time_s": control["summary"]["peak_vehicles_in_network_time_s"],
        "control_vehicles_in_network_t3600": control_by_t[3600]["vehicles_in_network"],
        "control_vehicles_in_network_t7199": control_by_t[7199]["vehicles_in_network"],
        "not_inserted_end_of_run": surge["summary"]["sumo_vehicles_not_inserted"],
        "control_not_inserted_end_of_run": control["summary"]["sumo_vehicles_not_inserted"],
        "max_pending_in_window": max(r["not_inserted_pending"] for r in rows if in_window(r)),
        "control_max_pending_in_window": max(r["not_inserted_pending"]
                                             for r in control["rows"] if in_window(r)),
    }
    report_data.update(excess_over_control(rows, control["rows"]))
    report_data["not_inserted_rise"] = relative_rise(
        report_data["not_inserted_end_of_run"], report_data["control_not_inserted_end_of_run"])
    report_data["max_pending_rise"] = relative_rise(
        report_data["max_pending_in_window"], report_data["control_max_pending_in_window"])
    (out_dir / "smoke_report.json").write_text(json.dumps(report_data, indent=2, default=str))
    print(json.dumps(report_data, indent=2, default=str))
    throttled = any(rise is not None and rise > 0.10 for rise in
                    (report_data["not_inserted_rise"], report_data["max_pending_rise"]))
    if throttled:
        print("WARNING: not-inserted vehicles rose by more than 10% versus the control run: "
              "entry edges are throttling the surge and the multiplier is not realised.")
    report(out_dir)

    assert len(rows) == HORIZON, f"expected {HORIZON} timeseries rows, got {len(rows)}"
    assert peak == max(row["vehicles_in_network"] for row in rows)
    assert WINDOW[0] <= peak_t <= WINDOW[1], (
        f"peak {peak} at t={peak_t} is outside the surge window {WINDOW} "
        f"(control peaks at {report_data['control_peak_vehicles_in_network']} "
        f"at t={report_data['control_peak_time_s']})")
    assert by_t[7199]["vehicles_in_network"] < by_t[3600]["vehicles_in_network"], (
        f"no recovery: occupancy at t=7199 ({by_t[7199]['vehicles_in_network']:.0f}) is not "
        f"below t=3600 ({by_t[3600]['vehicles_in_network']:.0f}); control has "
        f"{report_data['control_vehicles_in_network_t7199']:.0f} vs "
        f"{report_data['control_vehicles_in_network_t3600']:.0f}")


if __name__ == "__main__":
    tests = [test_parity, test_extension, test_route_filter, test_determinism, test_vtype,
             test_shipped_reproducible, test_tiling, test_multi_surge_inject,
             test_inject_mirrors_cfphys_attributes, test_inject_rng_isolation,
             test_auto_register, test_analysis_synthetic, test_edge_series]
    if "--skip-smoke" not in sys.argv:
        tests.append(test_smoke)
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except unittest.SkipTest as exc:
            print(f"SKIP {test.__name__}: {exc}")
        except (Exception, SystemExit):
            failed += 1
            traceback.print_exc()
            print(f"FAIL {test.__name__}")
    sys.exit(1 if failed else 0)
