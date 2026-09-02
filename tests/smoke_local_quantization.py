"""Offline checks for the local backend's --quantization flag.

Covers what a real GPU run cannot cheaply re-check: that the bitsandbytes
config actually reaches from_pretrained, that an unusable value fails at
construction rather than after the weights load, and -- the expensive mistake --
that adding the flag to run identity did not invalidate the runs that predate it.

Run: python tests/smoke_local_quantization.py
"""

import sys

sys.path.insert(0, ".")

from models_inference.LLM.open_llm import LLM_Inference

failures = []


def check(cond, msg):
    print(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def make_llm(quantization):
    """Skips initialize_llm: the kwargs are built before any weights load."""
    return LLM_Inference("fake/model", max_new_tokens=64,
                         quantization=quantization)


# --- 1. what reaches from_pretrained --------------------------------------

check(make_llm("none")._quantization_kwargs() == {},
      "--quantization none passes no config (existing runs stay identical)")

config = make_llm("8bit")._quantization_kwargs()["quantization_config"]
check(config.load_in_8bit is True and not config.load_in_4bit,
      "--quantization 8bit asks bitsandbytes for 8-bit weights")

config = make_llm("4bit")._quantization_kwargs()["quantization_config"]
check(config.load_in_4bit is True and config.bnb_4bit_quant_type == "nf4",
      "--quantization 4bit asks for nf4")

try:
    make_llm("fp8")
    check(False, "an unknown quantization must be refused at construction")
except ValueError:
    check(True, "an unknown quantization is refused at construction")

# --- 2. run identity ------------------------------------------------------

import experiments  # noqa: E402  (imported late: pulls in SUMO-side config)

BASE = dict(
    controller="llm", simulation_config="hzreal.sumocfg",
    intersection_config="three_lane", steps=3600, seed=1,
    blockage_name="none", hide_info=False, info_scope="both", num_rounds=None,
    llm_path="google/gemma-4-26B-A4B-it",
)

pre_flag = experiments._identity_fields(**BASE)
none = experiments._identity_fields(**BASE, quantization="none")
eight = experiments._identity_fields(**BASE, quantization="8bit")
four = experiments._identity_fields(**BASE, quantization="4bit")

check(pre_flag == none,
      "'none' keys identically to a run from before the flag existed")
check(eight != none and eight != four,
      "each quantization is its own result (8bit never pools with none or 4bit)")

# A manifest written before the flag has no 'quantization' key at all.
old_manifest = {
    "controller": "llm",
    "environment": {"simulation_config": "hzreal.sumocfg",
                    "intersection_config": "three_lane",
                    "simulation_steps": 3600, "seed": 1},
    "blockage": {"scenario_name": "none"},
    "args": {"llm_path": "google/gemma-4-26B-A4B-it"},
}
check(experiments.identity_from_manifest(old_manifest) == pre_flag,
      "a pre-flag manifest still matches -- completed runs are not re-run")

new_manifest = {**old_manifest,
                "args": {**old_manifest["args"], "quantization": "8bit"}}
check(experiments.identity_from_manifest(new_manifest) == eight,
      "an 8bit manifest round-trips, so the run is skipped on a re-run")

seed2 = experiments._identity_fields(**{**BASE, "seed": 2}, quantization="8bit")
check(experiments.config_key(eight) == experiments.config_key(seed2),
      "config_key still drops the seed (the new field was appended, not inserted)")

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1)
print("ALL CHECKS PASSED")
