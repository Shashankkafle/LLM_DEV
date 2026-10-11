"""Derive a family of surge scenarios from a template by sweeping one knob.

    python tools/make_surge_family.py --template dataset/.../scenarios/surge_cross_2_3.json \
        --sweep rate 300 600 900 1200
    python tools/make_surge_family.py --template ... --sweep hold 600 1200 1800 2700 --surges ns2

'rate' sets rate_veh_h on the chosen surges (default: all of them); 'hold'
sets each chosen surge's window end to window start + hold seconds. Every
member gets its own scenario JSON, route file, sidecar and sumocfg, named
<template alias>_r<rate> or _hold<seconds>, and is picked up by
experiments.CONFIGS automatically. The template itself is left untouched.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_surge_routes as gen  # noqa: E402


def member_suffix(knob, value):
    return f"r{value:g}" if knob == "rate" else f"hold{value:g}"


def rename_paths(scenario, suffix):
    """Append the member suffix to the file stem of every path the template
    owns (sumocfg, route file, sidecar)."""
    for key in ("sumocfg", "route_file", "sidecar"):
        path = Path(scenario[key])
        for extension in (".rou.xml", ".surge.json", ".sumocfg"):
            if path.name.endswith(extension):
                stem = path.name[:-len(extension)]
                scenario[key] = path.with_name(f"{stem}_{suffix}{extension}").as_posix()


def apply_knob(scenario, knob, value, chosen):
    for surge in scenario["surges"]:
        if chosen and surge["name"] not in chosen:
            continue
        if knob == "rate":
            surge["rate_veh_h"] = value
        else:
            surge["window"] = [surge["window"][0], surge["window"][0] + value]


def make_member(template, template_path, knob, value, chosen):
    suffix = member_suffix(knob, value)
    alias = f"{template['config_alias']}_{suffix}"
    scenario = copy.deepcopy(template)
    scenario["scenario_name"] = alias
    scenario["config_alias"] = alias
    scenario["description"] = (f"{template['description']} | family member: {knob}={value:g}"
                               f" on {', '.join(chosen) if chosen else 'all surges'}")
    rename_paths(scenario, suffix)
    apply_knob(scenario, knob, value, chosen)
    path = Path(template_path).with_name(f"surge_{alias}.json")
    path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    gen.main(["--scenario", str(path), "--write-sumocfg"])
    return alias


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", required=True, help="v2 scenario JSON to derive from")
    parser.add_argument("--sweep", nargs="+", required=True, metavar=("KNOB", "VALUE"),
                        help="'rate V...' (veh/h) or 'hold V...' (seconds)")
    parser.add_argument("--surges", nargs="*", default=[], help="surge names to change (default all)")
    args = parser.parse_args(argv)
    knob, values = args.sweep[0], args.sweep[1:]
    if knob not in ("rate", "hold") or not values:
        parser.error("--sweep takes 'rate' or 'hold' followed by one or more values")
    args.knob, args.values = knob, [float(v) for v in values]
    return args


def main(argv=None):
    args = parse_args(argv)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    if "surges" not in template:
        sys.exit("make_surge_family: the template must be a schema v2 scenario with 'surges'")
    unknown = set(args.surges) - {s["name"] for s in template["surges"]}
    if unknown:
        sys.exit(f"make_surge_family: unknown surge names {sorted(unknown)}")
    aliases = [make_member(template, args.template, args.knob, value, args.surges)
               for value in args.values]
    print("configs: " + " ".join(aliases))
    return aliases


if __name__ == "__main__":
    main()
