"""Verify batched LLM inference is equivalent to sequential inference.

inference_batch() runs many prompts through one generate() with left-padding;
this checks each prompt's output matches what inference() produces alone, so
batching changes throughput, not decisions. Greedy decoding attends within a
sequence only, so the two are logically identical -- the sole gap is GPU
float-point reduction order, which can rarely flip a token on a near-tie. So we
report both the exact raw-text match rate and the extracted <signal> match rate:

  * On CPU/float32 the arithmetic is order-stable, so ANY raw-text divergence is
    a real left-padding/masking/slicing bug -> this config REQUIRES raw-text N/N.
  * On GPU/float16 raw text may differ by a token on a near-tie; what must hold
    is that no extracted <signal> flips (both-empty pairs are NOT counted as a
    match) and a size-1 batch equals the single call exactly.

The multi-length batch is what exercises left-padding; the size-1 check only
covers decode plumbing (no padding). Certify the shipping config on the box with
the REAL model and PRODUCTION token budget -- the 14B reliably emits <signal>, so
signal parity there is discriminating and a long generation can surface late drift:

Local (CPU) mechanism check on the 0.5B default:
    python tests/verify_batch_equivalence.py --dtype float32 --device cpu --max_new_tokens 64
On the GPU box against the 14B (acceptance gate before trusting batched runs):
    python tests/verify_batch_equivalence.py --llm_path ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b --max_new_tokens 1024
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")

from models_inference.LLM.open_llm import LLM_Inference
from configurations import LLM_DEFAULT_PATH
from runner import parse_llm_signal


# Deliberately different lengths, so the batch must left-pad; content is
# irrelevant to equivalence (only whether both paths produce the same tokens).
PROMPTS = [
    "Reply with exactly <signal>ETWT</signal> and nothing else.",
    "You direct a four-way intersection. East has 5 queued cars, west 2, north "
    "0, south 0. Think briefly, then answer with one of ETWT/ELWL/NTST/NLSL "
    "inside <signal></signal>.",
    "Signal ETWT serves east-west through traffic; NTST serves north-south "
    "through traffic. North has a long queue of 12 vehicles that cannot move; "
    "east is clear. Which signal helps most? Answer in <signal></signal>.",
    "Choose a phase. <signal>?</signal>",
    "A blockage stalls the north through lane. Queues: north 9 (stuck), east 3, "
    "west 4, south 1. Reason step by step about which movement can actually "
    "discharge, then give <signal>YOUR_CHOICE</signal>.",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_path", default=LLM_DEFAULT_PATH)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device", default="auto",
                    help="'auto' (device_map=auto) or an explicit device e.g. 'cpu'")
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Lower = faster test; enough to expose any padding bug.")
    return ap.parse_args()


def main():
    args = parse_args()
    llm = LLM_Inference(llm_path=args.llm_path,
                        max_new_tokens=args.max_new_tokens)
    device_map = "auto" if args.device == "auto" else None
    llm.initialize_llm(torch_dtype=getattr(torch, args.dtype), device_map=device_map)
    if args.device != "auto":
        llm.model.to(args.device)

    sequential, seq_usage = [], []
    for prompt in PROMPTS:
        sequential.append(llm.inference(prompt))
        seq_usage.append(dict(llm.last_usage))  # overwritten each call -> copy
    batched = llm.inference_batch(PROMPTS)
    bat_usage = llm.last_usage_batch
    assert len(batched) == len(PROMPTS), \
        f"batch returned {len(batched)} outputs for {len(PROMPTS)} prompts"

    # Token-usage parity: the batched completion_tokens must match the single
    # path (the pad==eos undercount fix). Exact on order-stable configs.
    usage_matches = sum(su == bu for su, bu in zip(seq_usage, bat_usage))

    # Batch-of-1 must equal the single call exactly (decode plumbing); the
    # multi-length batch above is what actually exercises left-padding.
    size1_ok = llm.inference_batch([PROMPTS[0]])[0] == sequential[0]

    n = len(PROMPTS)
    raw_matches = sum(a == b for a, b in zip(sequential, batched))
    sig_pairs = [(parse_llm_signal(a), parse_llm_signal(b))
                 for a, b in zip(sequential, batched)]
    # A pair where BOTH sides produced no <signal> is not evidence of agreement,
    # so it is excluded from the parity check rather than scored as a match.
    signalful = [(sa, sb) for sa, sb in sig_pairs if sa is not None or sb is not None]
    sig_mismatches = [(sa, sb) for sa, sb in signalful if sa != sb]
    both_empty = n - len(signalful)
    order_stable = args.device == "cpu" and args.dtype == "float32"

    print(f"\nprompts={n}  dtype={args.dtype}  device={args.device}  "
          f"max_new_tokens={args.max_new_tokens}  order_stable={order_stable}")
    print(f"raw-text matches:         {raw_matches}/{n}")
    print(f"<signal> pairs compared:  {len(signalful)}/{n}  "
          f"(both-empty, excluded: {both_empty})")
    print(f"<signal> mismatches:      {len(sig_mismatches)}  {sig_mismatches or ''}")
    print(f"token-usage matches:      {usage_matches}/{n}")
    print(f"size-1 batch == single:   {size1_ok}")
    for i, (a, b) in enumerate(zip(sequential, batched)):
        if a != b:
            print(f"\n[raw diff @ prompt {i}] "
                  f"signal seq={parse_llm_signal(a)!r} bat={parse_llm_signal(b)!r}")
            print(f"  seq: {a[:200]!r}")
            print(f"  bat: {b[:200]!r}")

    ok = size1_ok and not sig_mismatches
    if order_stable:
        # Reduction order is stable here, so any raw divergence is a real
        # padding/masking/slicing bug, not a benign float-point tie.
        if raw_matches != n:
            ok = False
            print("\n[FAIL] order-stable config but raw text diverged -- this is "
                  "a left-padding/masking/slicing bug, not a float-point tie.")
        if usage_matches != n:
            ok = False
            print("\n[FAIL] order-stable config but token usage diverged -- "
                  "batched completion_tokens does not match the single path.")
    elif len(signalful) < max(1, n // 2):
        print(f"\n[WARN] only {len(signalful)}/{n} prompts produced a <signal>; "
              "signal-parity evidence is weak. Use a model/prompts that emit "
              "signals (the 14B does) and production --max_new_tokens 1024.")

    print("\n" + ("PASS: batched decisions match sequential" if ok else
                  "FAIL: batched decisions diverge -- inspect the diffs above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
