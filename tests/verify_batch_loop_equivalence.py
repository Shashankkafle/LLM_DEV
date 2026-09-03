"""Verify the batched control loop records exactly what the sequential loop does.

Runs the same short Hangzhou 4x4 simulation three times with a deterministic
stub LLM whose answer is a pure function of the prompt -- once with runner's
sequential driver (--sequential), once with the batched driver, once batched
with a small --max_batch_size cap -- and diffs every intersection's
decisions.jsonl (ignoring wall-clock timestamp and measured latency).

A prompt-dependent stub is the point: if the batched driver ever handed one
intersection's output to another, that intersection's recorded <signal> would
change. Byte-identical logs across all three runs therefore prove the
gather-then-apply restructure preserves inputs, decision cadence, per-decision
recording, AND is invariant to batch size -- with no GPU or real model.

Needs SUMO on PATH. From the repo root:
    PYTHONPATH=. python tests/verify_batch_loop_equivalence.py
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

import runner
from experiments import CONFIGS

PHASES = ["ETWT", "ELWL", "NTST", "NLSL"]
STEPS = 250
SEED = 1
# Fields that legitimately differ run to run (wall clock) -- not part of the
# decision, so excluded from the comparison.
VOLATILE = ("timestamp",)


class DeterministicStubLLM:
    """A stub whose answer depends on the prompt, so a batched mis-mapping would
    surface as a changed decision. No GPU or real model needed."""

    def __init__(self, llm_path, **kwargs):
        pass

    def initialize_llm(self, *args, **kwargs):
        pass

    def describe(self):
        return None

    def _signal(self, prompt):
        digest = int(hashlib.md5((prompt or "").encode()).hexdigest(), 16)
        return PHASES[digest % len(PHASES)]

    def inference(self, prompt):
        return f"analysis <signal>{self._signal(prompt)}</signal>"

    def inference_batch(self, prompts):
        return [self.inference(p) for p in prompts]


def run_once(test_name, config, sequential, max_batch_size):
    args = argparse.Namespace(
        test_name=test_name, run_group=None, simulation_steps=STEPS,
        simulation_config=config, llm_path="stub", use_gui=False, seed=SEED,
        blockage_scenario=None, hide_blockage_info=False,
        blockage_info_scope="both", intersection_config="three_lane",
        sequential=sequential, max_batch_size=max_batch_size,
        # The LLM-backend flags runner.main reads. Defaults only -- the stub
        # never loads a model -- but they must exist or main() raises.
        max_new_tokens=None,
        request_timeout=None,
        reasoning_max_tokens=None,
        reasoning="auto",
        quantization="none",
        logs_dir=None,
        )
    runner.main(args)
    return sorted(Path("logs").glob(f"{test_name}_*"))[-1]


def normalize(event):
    event = json.loads(json.dumps(event))  # deep copy
    for key in VOLATILE:
        event.pop(key, None)
    metrics = event.get("metrics")
    if isinstance(metrics, dict):
        metrics.pop("inference_latency_ms", None)
    return event


def load_decisions(run_dir):
    decisions = {}
    for path in sorted(run_dir.glob("*/decisions.jsonl")):
        iid = path.parent.name
        decisions[iid] = [normalize(json.loads(line)) for line in path.open()]
    return decisions


def compare(label, base, other):
    ok = True
    if set(base) != set(other):
        print(f"[FAIL] {label}: intersection sets differ "
              f"({sorted(set(base) ^ set(other))})")
        return False
    for iid in sorted(base):
        a, b = base[iid], other[iid]
        if a == b:
            continue
        ok = False
        if len(a) != len(b):
            print(f"[FAIL] {label}: {iid} has {len(a)} vs {len(b)} decisions")
            continue
        for i, (ea, eb) in enumerate(zip(a, b)):
            if ea != eb:
                print(f"[FAIL] {label}: {iid} decision #{i} (step {ea.get('step')}) "
                      f"differs")
                print(f"   base: {json.dumps(ea)[:300]}")
                print(f"  other: {json.dumps(eb)[:300]}")
                break
    return ok


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    runner.build_llm = lambda llm_path, **kwargs: DeterministicStubLLM(llm_path)
    config = CONFIGS["hz1"]  # Hangzhou 4x4 -> 16 intersections, real batches

    dir_seq = run_once("batchloop_seq", config, sequential=True, max_batch_size=0)
    dir_bat = run_once("batchloop_bat", config, sequential=False, max_batch_size=0)
    dir_cap = run_once("batchloop_cap", config, sequential=False, max_batch_size=3)

    dec_seq = load_decisions(dir_seq)
    dec_bat = load_decisions(dir_bat)
    dec_cap = load_decisions(dir_cap)

    total = sum(len(v) for v in dec_seq.values())
    queried = sum(1 for v in dec_seq.values() for e in v
                  if e["phase_action"]["decision_type"] != "no_action_empty")
    print(f"\nintersections={len(dec_seq)}  decisions={total}  "
          f"LLM-queried={queried}  (steps={STEPS}, seed={SEED})")

    ok = compare("batched vs sequential", dec_seq, dec_bat)
    ok = compare("max_batch_size=3 vs unlimited", dec_bat, dec_cap) and ok

    if ok and queried == 0:
        print("[FAIL] no LLM-queried decisions -- test exercised nothing")
        ok = False

    if ok:
        for d in (dir_seq, dir_bat, dir_cap):
            shutil.rmtree(d)
        print("\nPASS: batched loop records identically to sequential, and is "
              "invariant to batch size -- run dirs removed")
        return 0
    print("\nFAIL: divergence found -- run dirs kept for inspection")
    return 1


if __name__ == "__main__":
    sys.exit(main())
