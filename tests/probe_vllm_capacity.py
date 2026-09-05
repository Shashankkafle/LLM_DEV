"""Find how many concurrent requests the vLLM server takes before throughput
stops rising -- in a couple of minutes, without a SUMO run.

    python tests/probe_vllm_capacity.py --llm_path vllm:qwen2.5_14b
    python tests/probe_vllm_capacity.py --llm_path vllm:qwen2.5_14b --levels 8,16,32

For each level N it fires N identical requests at once through the real
client (same payload, timeout and retries a run uses), repeats, and reads the
server's /metrics around the burst. The prompt is a real decision prompt from
the newest run under logs/, because prompt length drives prefill cost.

Reading the table: the knee is where tok/s stops growing; the cliff is the
first level with waiting > 0 or a preemption. One SUMO run averages ~3
concurrent requests, so the safe number of parallel runs is roughly knee / 3.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import monitor_llm  # noqa: E402
from models_inference.LLM import http_llm  # noqa: E402


def newest_prompt(logs_dir):
    files = sorted(Path(logs_dir).glob("**/decisions.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                prompt = (json.loads(line).get("llm_input") or {}).get("user_prompt")
                if prompt:
                    return prompt
    sys.exit(f"No decision prompt found under {logs_dir}; pass --prompt-file")


def wait_until_idle(poller, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not poller.poll()["running"]:
            return
        time.sleep(1)


def run_level(llm, prompt, n):
    """N concurrent requests, exactly as a run's batched step issues them.
    Returns (wall_s, [latency_ms...], completion_tokens)."""
    started = time.perf_counter()
    llm.inference_batch([prompt] * n)
    wall_s = time.perf_counter() - started
    usages = llm.last_usage_batch
    return wall_s, [u["latency_ms"] for u in usages], sum(u["completion_tokens"] or 0 for u in usages)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm_path", required=True)
    parser.add_argument("--levels", default="1,2,4,8,16,32,64")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--force", action="store_true",
                        help="run even if the server is already busy")
    args = parser.parse_args()

    prompt = (Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file
              else newest_prompt(args.logs_dir))
    llm = http_llm.build(args.llm_path, max_new_tokens=args.max_new_tokens)
    llm.initialize_llm()
    poller = monitor_llm.ServerPoller(monitor_llm.metrics_url(llm.base_url))

    if poller.poll()["running"] and not args.force:
        sys.exit("Server is busy (requests running); refusing to probe. --force to override.")

    print(f"prompt: {len(prompt)} chars   max_tokens: {args.max_new_tokens}\n")
    print(f"{'N':>4} {'wall s':>7} {'tok/s':>7} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}"
          f" {'waiting':>7} {'kv %':>6} {'preempt':>7}")
    for n in (int(x) for x in args.levels.split(",")):
        wait_until_idle(poller)
        run_level(llm, prompt, n)                       # warm-up, discarded
        wait_until_idle(poller)
        poller.poll()
        wall_s, latencies, tokens = run_level(llm, prompt, n)
        after = poller.poll()
        latencies.sort()
        print(f"{n:>4} {wall_s:>7.1f} {tokens / wall_s:>7.0f}"
              f" {statistics.median(latencies):>8.0f}"
              f" {latencies[int(0.95 * (len(latencies) - 1))]:>8.0f} {latencies[-1]:>8.0f}"
              f" {monitor_llm.fmt(after['waiting']):>7} {monitor_llm.fmt(after['kv_pct'], '{:.1f}'):>6}"
              f" {monitor_llm.fmt(after['preemptions_delta']):>7}")


if __name__ == "__main__":
    main()
