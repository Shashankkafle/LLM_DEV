"""Generate cfphys route/config variants for the dataset/sumo_version networks.

The original converted route files define vehicles with no type and no depart
attributes, so SUMO uses its default vehicle type and inserts every vehicle at
speed 0 -- which makes each vehicle count as "halting" on its entry lane for one
step (a phantom queue of 1, ~290 m from the stop line). The cfphys variant fixes
both, matching the pattern already used under dataset/llm_light (METRICS.md
item 3): CityFlow-matched deterministic drivers plus departLane="best"
departSpeed="max".

Writes roadnet_cfphys.rou.xml and roadnet_cfphys.sumocfg next to each original.
Never modifies the originals.
"""

import re
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset" / "sumo_version"

# Exact vType used by the llm_light cfphys variants (CityFlow-matched physics).
CFPHYS_VTYPE = (
    '<vType id="pkw" length="5.0" width="2.0" minGap="2.5" maxSpeed="11.111" '
    'accel="2.0" decel="4.5" tau="2.0" sigma="0" speedFactor="1" speedDev="0"/>'
)

VTYPE_PATTERN = re.compile(r'<vType id="pkw"[^>]*/>')
VEHICLE_PATTERN = re.compile(r'<vehicle id="([^"]+)" depart="([^"]+)">')
VEHICLE_REPLACEMENT = (
    r'<vehicle id="\1" type="pkw" depart="\2" departLane="best" departSpeed="max">'
)


def transform_routes(text):
    """Return the cfphys version of a roadnet.rou.xml, or raise ValueError."""
    if "departSpeed" in text:
        raise ValueError("already has departSpeed attributes")

    text, n_vtypes = VTYPE_PATTERN.subn(CFPHYS_VTYPE, text)
    if n_vtypes != 1:
        raise ValueError(f"expected exactly 1 pkw vType, found {n_vtypes}")

    total_vehicles = text.count("<vehicle ")
    text, n_vehicles = VEHICLE_PATTERN.subn(VEHICLE_REPLACEMENT, text)
    if n_vehicles != total_vehicles:
        raise ValueError(
            f"transformed {n_vehicles} of {total_vehicles} vehicle elements; "
            "unexpected vehicle attributes present"
        )
    return text, n_vehicles


def transform_sumocfg(text):
    """Point the route-files entry at the cfphys route file."""
    needle = 'route-files value="roadnet.rou.xml"'
    if text.count(needle) != 1:
        raise ValueError("sumocfg does not have the expected route-files line")
    return text.replace(needle, 'route-files value="roadnet_cfphys.rou.xml"')


def main():
    failures = []
    for net_dir in sorted(DATASET_DIR.iterdir()):
        rou_path = net_dir / "roadnet.rou.xml"
        cfg_path = net_dir / "roadnet.sumocfg"
        if not rou_path.is_file() or not cfg_path.is_file():
            continue

        out_rou = net_dir / "roadnet_cfphys.rou.xml"
        out_cfg = net_dir / "roadnet_cfphys.sumocfg"
        if out_rou.exists() or out_cfg.exists():
            print(f"SKIP {net_dir.name}: cfphys files already exist")
            continue

        try:
            new_rou, n_vehicles = transform_routes(rou_path.read_bytes().decode("utf-8"))
            new_cfg = transform_sumocfg(cfg_path.read_bytes().decode("utf-8"))
        except ValueError as e:
            failures.append(net_dir.name)
            print(f"FAIL {net_dir.name}: {e}")
            continue

        # Bytes I/O keeps the original line endings intact.
        out_rou.write_bytes(new_rou.encode("utf-8"))
        out_cfg.write_bytes(new_cfg.encode("utf-8"))
        print(f"OK   {net_dir.name}: {n_vehicles} vehicles")

    if failures:
        sys.exit(f"{len(failures)} network(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
