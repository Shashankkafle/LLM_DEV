"""Offline checks that the identity-bearing LLM flags key runs correctly.

Covers the expensive mistake: adding a field to run identity, or changing how
one is normalized, silently invalidates every completed run that predates it and
the grid re-runs work that was already paid for. `_identity_fields` must
therefore gain no field, lose none, and reorder none -- config_key() drops the
seed by position, so the leading offsets are load-bearing.

--quantization no longer changes anything about a request (the server owns its
precision); it survives purely as the operator-asserted label that keeps an
AWQ-served run from pooling with an fp16-served one, which is exactly what these
checks pin.

Run: python tests/smoke_run_identity_flags.py
"""

import sys

sys.path.insert(0, ".")

import experiments

failures = []


def check(cond, msg):
    print(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


BASE = dict(
    controller="llm", simulation_config="hzreal.sumocfg",
    intersection_config="three_lane", steps=3600, seed=1,
    blockage_name="none", hide_info=False, info_scope="both", num_rounds=None,
    llm_path="vllm:qwen2.5_14b",
)

pre_flag = experiments._identity_fields(**BASE)
none = experiments._identity_fields(**BASE, quantization="none")
eight = experiments._identity_fields(**BASE, quantization="8bit")
awq = experiments._identity_fields(**BASE, quantization="awq")

check(pre_flag == none,
      "'none' keys identically to a run from before the flag existed")
check(eight != none and eight != awq,
      "each declared precision is its own result (awq never pools with fp16)")

# A manifest written before the flag has no 'quantization' key at all.
old_manifest = {
    "controller": "llm",
    "environment": {"simulation_config": "hzreal.sumocfg",
                    "intersection_config": "three_lane",
                    "simulation_steps": 3600, "seed": 1},
    "blockage": {"scenario_name": "none"},
    "args": {"llm_path": "vllm:qwen2.5_14b"},
}
check(experiments.identity_from_manifest(old_manifest) == pre_flag,
      "a pre-flag manifest still matches -- completed runs are not re-run")

new_manifest = {**old_manifest,
                "args": {**old_manifest["args"], "quantization": "awq"}}
check(experiments.identity_from_manifest(new_manifest) ==
      experiments._identity_fields(**BASE, quantization="awq"),
      "a declared-precision manifest round-trips, so the run is skipped")

seed2 = experiments._identity_fields(**{**BASE, "seed": 2}, quantization="8bit")
check(experiments.config_key(eight) == experiments.config_key(seed2),
      "config_key still drops the seed (fields were appended, not inserted)")

# --reasoning is normalized the same way, for the same reason.
auto = experiments._identity_fields(**BASE, reasoning="auto")
off = experiments._identity_fields(**BASE, reasoning="off")
check(auto == pre_flag and off != auto,
      "'auto' keys identically to a pre-flag run; 'off' is its own result")

# The run-dir tag and the results' model column must not care about the scheme.
check(experiments.model_token("vllm:qwen2.5_14b") == "qwen2.5_14b"
      and experiments.model_token("openrouter:google/gemma-3-27b-it")
      == "gemma-3-27b-it",
      "model_token strips any scheme, so reports read the same across backends")

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1)
print("ALL CHECKS PASSED")
