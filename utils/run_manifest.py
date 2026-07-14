"""Provenance capture for run records.

Every runner writes a run_manifest.json into its run dir describing what ran
(controller, CLI args, code version), on what environment (simulation inputs,
signal timing, seed, SUMO version), and how the run ended. The manifest is
saved at run start so a crash still leaves the record, enriched once SUMO is
up, and finalized (status + wall-clock duration) when the run ends.

Imports only stdlib + configurations, so any module can use it without
circular imports.
"""

import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from importlib import metadata
from pathlib import Path

from configurations import RUN_MANIFEST_FILENAME, RUN_RECORD_SCHEMA_VERSION

# Packages whose versions materially affect results; missing ones are skipped.
TRACKED_PACKAGES = ["torch", "transformers", "tensorflow", "traci", "sumolib", "numpy"]


def git_commit():
    """Current repo commit hash for reproducibility, or None if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_dirty():
    """True when the working tree has uncommitted changes, None if unknown."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def package_versions():
    versions = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            pass
    return versions


def _git_blob_sha1(path):
    # CRLF->LF so a Windows checkout hashes the same as Linux and as
    # git's autocrlf-normalized blobs.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def input_fingerprints(sumo_config):
    """Git-blob SHA-1 of the sumocfg and the input files it references, keyed
    by filename. Matches `git hash-object <file>`, so a run's inputs can be
    checked against a commit directly. Returns None if anything can't be read
    (a run must never fail over provenance).
    """
    try:
        config_path = Path(sumo_config)
        fingerprints = {config_path.name: _git_blob_sha1(config_path)}
        root = ET.parse(config_path).getroot()
        for tag in ("net-file", "route-files", "additional-files"):
            node = root.find(f"./input/{tag}")
            if node is None:
                continue
            for name in node.get("value", "").split(","):
                ref = config_path.parent / name.strip()
                if ref.exists():
                    fingerprints[ref.name] = _git_blob_sha1(ref)
        return fingerprints
    except Exception:
        return None


def file_sha1(path):
    """Plain SHA-1 of a file's raw bytes (for binaries like .h5 weights), or
    None if unreadable."""
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _blockage_block(args):
    scenario_path = getattr(args, "blockage_scenario", None)
    if not scenario_path:
        return None
    try:
        with open(scenario_path) as f:
            scenario_name = json.load(f).get("scenario_name")
    except Exception:
        scenario_name = None
    return {
        "scenario_path": scenario_path,
        "scenario_name": scenario_name,
        "hide_blockage_info": getattr(args, "hide_blockage_info", False),
        "blockage_info_scope": getattr(args, "blockage_info_scope", None),
    }


def build_manifest(controller, args, conf_name, conf, extra=None):
    """Everything knowable before SUMO starts. args is the argparse Namespace;
    conf is the intersection config (None for replay, which has none)."""
    simulation_config = getattr(args, "simulation_config", None)
    manifest = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "controller": controller,
        "test_name": getattr(args, "test_name", None),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "ended_at": None,
        "wall_clock_duration_s": None,
        "status": "running",
        "argv": list(sys.argv),
        "args": vars(args),
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "environment": {
            "simulation_config": simulation_config,
            "simulation_config_abs": (
                str(Path(simulation_config).resolve()) if simulation_config else None
            ),
            "input_files": (
                input_fingerprints(simulation_config) if simulation_config else None
            ),
            "intersection_config": conf_name,
            "signal_timing": dict(conf["global_settings"]) if conf else None,
            "phases": (
                {name: {"id": p["id"], "green": p["green"], "yellow": p["yellow"]}
                 for name, p in conf["phases"].items()}
                if conf else None
            ),
            "seed": getattr(args, "seed", None),
            "simulation_steps": getattr(args, "simulation_steps", None),
        },
        "sumo": None,
        "blockage": _blockage_block(args),
        "llm": None,
        "colight": None,
    }
    if extra:
        manifest.update(extra)
    return manifest


def add_sumo_runtime(manifest, sumo_cmd):
    """Fill the fields only knowable with a live TraCI connection."""
    import traci
    manifest["sumo"] = {
        "version": traci.getVersion()[1],
        "cmd": [str(part) for part in sumo_cmd],
        "step_length_s": traci.simulation.getDeltaT(),
        "intersection_ids": sorted(traci.trafficlight.getIDList()),
    }


def save_manifest(run_dir, manifest):
    path = Path(run_dir) / RUN_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def finalize_manifest(run_dir, status):
    """Stamp end time, duration, and final status ("completed" | "crashed").
    No-op if the manifest was never written."""
    path = Path(run_dir) / RUN_MANIFEST_FILENAME
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    ended = datetime.now()
    manifest["ended_at"] = ended.isoformat(timespec="seconds")
    started = manifest.get("started_at")
    if started:
        manifest["wall_clock_duration_s"] = round(
            (ended - datetime.fromisoformat(started)).total_seconds(), 1)
    manifest["status"] = status
    save_manifest(run_dir, manifest)
