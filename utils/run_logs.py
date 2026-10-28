"""Readers for a finished run's own files, shared by the offline checks
(utils/scenario_gate.py, tools/eval_metrics.py)."""

import csv
import json
from dataclasses import dataclass

from configurations import (
    BLOCKAGE_SCENARIO_COPY_FILENAME,
    MOVEMENT_FLAGS_FILENAME,
    RUN_MANIFEST_FILENAME,
)
from utils.movement_flags import FlagConfig
from utils.network_topology import read_network, resolve_sumocfg, sumocfg_input_paths


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_float(text):
    return None if text == "" else float(text)


def read_manifest(run_dir):
    path = run_dir / RUN_MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def theta_in(manifest):
    """The run's theta_in, or the FlagConfig default where the manifest
    logged null (the v3 batch)."""
    value = (manifest.get("args") or {}).get("theta_in")
    return FlagConfig.theta_in if value is None else value


def input_paths(manifest):
    """(net file, route file) of the run, or (None, None) if its sumocfg is
    not on this machine."""
    sumocfg = resolve_sumocfg(manifest)
    if sumocfg is None:
        return None, None
    net_path, route_paths = sumocfg_input_paths(sumocfg)
    return net_path, (route_paths[0] if route_paths else None)


def load_network(manifest):
    net_path, _ = input_paths(manifest)
    return read_network(net_path) if net_path is not None and net_path.exists() else None


def blocked_lanes(run_dir):
    """Lanes of the run's blockage scenario (its copy in the run dir)."""
    path = run_dir / BLOCKAGE_SCENARIO_COPY_FILENAME
    if not path.exists():
        return []
    scenario = json.loads(path.read_text(encoding="utf-8"))
    return sorted({blockage["lane_id"] for blockage in scenario.get("blockages", [])})


def blockage_window_ends(run_dir):
    path = run_dir / BLOCKAGE_SCENARIO_COPY_FILENAME
    if not path.exists():
        return []
    scenario = json.loads(path.read_text(encoding="utf-8"))
    return [blockage["end_step"] for blockage in scenario.get("blockages", [])
            if blockage.get("end_step") is not None]


@dataclass
class LaneGreen:
    """One closed green of one lane, collapsed from its movement rows."""
    t_close: int
    tls: str
    lane: str
    tosi: object
    ratio_start: object
    ratio_end: object
    max_sosi: object      # largest SOSI over the lane's movements; None if all missing
    held_over: int
    discharged: int


def lane_greens(flag_rows):
    """LaneGreens in log order (t_close, then lane)."""
    greens = {}
    for row in flag_rows:
        key = (int(row["t_close"]), row["lane"])
        sosi = parse_float(row["sosi"])
        if key not in greens:
            greens[key] = LaneGreen(
                t_close=key[0], tls=row["tls"], lane=row["lane"],
                tosi=parse_float(row["tosi"]), ratio_start=parse_float(row["ratio_start"]),
                ratio_end=parse_float(row["ratio_end"]), max_sosi=sosi,
                held_over=int(row["held_over"]), discharged=int(row["discharged"]))
        elif sosi is not None:
            green = greens[key]
            green.max_sosi = sosi if green.max_sosi is None else max(green.max_sosi, sosi)
    return list(greens.values())


def read_lane_greens(run_dir):
    return lane_greens(read_csv_rows(run_dir / MOVEMENT_FLAGS_FILENAME))
