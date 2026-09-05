"""Live view of the vLLM server and of the runs using it.

    python monitor_llm.py                # poll every 2 s until Ctrl-C
    python monitor_llm.py --once
    python monitor_llm.py --jsonl logs/monitor.jsonl   # also append every poll

Reads the server's Prometheus /metrics (stdlib only, no prometheus_client),
nvidia-smi when available, and the per-run heartbeats that utils/llm_activity
writes under <logs>/_active. The questions it answers, in order of weight:

  waiting      requests the scheduler could not admit. Sustained > 0 means
               the server is past capacity.
  kv cache     KV-pool occupancy. Near 100 % is the other saturation mode.
  preemptions  a sequence was evicted and will be recomputed. Any increase
               means back off.
  ttft / e2e   latency percentiles over the last poll interval, to compare
               against the client's request timeout.

vLLM's V1 engine renamed some families relative to V0, so every lookup takes
alternatives and a missing family renders as n/a rather than failing. Before
trusting the names below on a new server: curl .../metrics | grep '# HELP'.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

from configurations import LLM_REQUEST_TIMEOUT_S
from utils import llm_activity

DEFAULT_BASE_URL = "http://localhost:8000/v1"
INF = float("inf")

RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREEMPTIONS = ("vllm:num_preemptions_total",)
PROMPT_TOKENS = ("vllm:prompt_tokens_total",)
GENERATION_TOKENS = ("vllm:generation_tokens_total",)
REQUESTS_DONE = ("vllm:request_success_total",)
TTFT_HISTOGRAM = "vllm:time_to_first_token_seconds"
E2E_HISTOGRAM = "vllm:e2e_request_latency_seconds"


# --- Prometheus text format --------------------------------------------------

def parse_metrics(text):
    """Prometheus exposition text -> {name: [(labels, value), ...]}.

    Lines that do not parse are skipped: an unfamiliar family must never take
    the monitor down mid-sweep."""
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value_text = line.rpartition(" ")
        try:
            value = float(value_text)
        except ValueError:
            continue
        name, _, label_text = name_part.partition("{")
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', label_text))
        metrics.setdefault(name, []).append((labels, value))
    return metrics


def metric_value(metrics, names):
    """Sum of the first present family (the server labels everything by
    model_name; one model is served). None when no name is present."""
    for name in names:
        if name in metrics:
            return sum(value for _, value in metrics[name])
    return None


def histogram_buckets(metrics, name):
    """{upper_bound: cumulative_count} for one histogram family."""
    buckets = {}
    for labels, value in metrics.get(f"{name}_bucket", []):
        bound = float(labels.get("le", "inf").replace("+Inf", "inf"))
        buckets[bound] = buckets.get(bound, 0.0) + value
    return buckets


def quantile(buckets, q):
    """Interpolated quantile from cumulative bucket counts; None when empty.
    A quantile that lands in the +Inf bucket returns the last finite bound --
    an "at least" value, which is the honest answer."""
    total = buckets.get(INF, 0.0)
    if total <= 0:
        return None
    target = q * total
    prev_bound, prev_count = 0.0, 0.0
    for bound in sorted(buckets):
        count = buckets[bound]
        if count >= target:
            if math.isinf(bound):
                return prev_bound
            fraction = (target - prev_count) / max(count - prev_count, 1e-9)
            return prev_bound + (bound - prev_bound) * fraction
        prev_bound, prev_count = bound, count
    return None


# --- polling -------------------------------------------------------------------

def metrics_url(base_url):
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-len("/v1")]
    return f"{root}/metrics"


def fetch_text(url, timeout_s=5):
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8", "replace")


class ServerPoller:
    """Turns two consecutive scrapes into rates and windowed percentiles.

    Counters and histograms are cumulative since server start, so on their
    own they describe the server's whole life. Differencing against the
    previous poll is what makes the display say what is happening now."""

    def __init__(self, url):
        self.url = url
        self.prev = None
        self.prev_time = None

    def poll(self):
        metrics = parse_metrics(fetch_text(self.url))
        now = time.time()
        snapshot = {
            "time": now,
            "running": metric_value(metrics, RUNNING),
            "waiting": metric_value(metrics, WAITING),
            "kv_pct": _percent(metric_value(metrics, KV_USAGE)),
            "preemptions": metric_value(metrics, PREEMPTIONS),
            "preemptions_delta": self._delta(metrics, PREEMPTIONS),
            "prefill_tps": self._rate(metrics, PROMPT_TOKENS, now),
            "decode_tps": self._rate(metrics, GENERATION_TOKENS, now),
            "requests_ps": self._rate(metrics, REQUESTS_DONE, now),
        }
        for label, family in (("ttft", TTFT_HISTOGRAM), ("e2e", E2E_HISTOGRAM)):
            window = self._bucket_window(metrics, family)
            snapshot[f"{label}_p50"] = quantile(window, 0.50)
            snapshot[f"{label}_p99"] = quantile(window, 0.99)
        self.prev, self.prev_time = metrics, now
        return snapshot

    def _delta(self, metrics, names):
        now_value = metric_value(metrics, names)
        prev_value = metric_value(self.prev, names) if self.prev else None
        if now_value is None or prev_value is None or now_value < prev_value:
            return None
        return now_value - prev_value

    def _rate(self, metrics, names, now):
        delta = self._delta(metrics, names)
        if delta is None or not self.prev_time:
            return None
        return delta / max(now - self.prev_time, 1e-9)

    def _bucket_window(self, metrics, family):
        now_buckets = histogram_buckets(metrics, family)
        prev_buckets = histogram_buckets(self.prev, family) if self.prev else {}
        return {bound: count - prev_buckets.get(bound, 0.0)
                for bound, count in now_buckets.items()}


def _percent(fraction):
    return None if fraction is None else fraction * 100


def read_gpu():
    """utilisation %, memory used/total in GB from nvidia-smi; None without it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        util, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"gpu_util": util, "gpu_mem_used_gb": used / 1024, "gpu_mem_total_gb": total / 1024}


# --- rendering -------------------------------------------------------------------

def fmt(value, spec="{:.0f}", missing="n/a"):
    return missing if value is None else spec.format(value)


def render(snapshot, gpu, runs, timeout_s=LLM_REQUEST_TIMEOUT_S):
    lines = [
        f"vLLM  {datetime.fromtimestamp(snapshot['time']):%H:%M:%S}",
        f"  running {fmt(snapshot['running']):>4}   waiting {fmt(snapshot['waiting']):>4}"
        f"   kv {fmt(snapshot['kv_pct'], '{:.1f}'):>5} %"
        f"   preemptions {fmt(snapshot['preemptions'])} (+{fmt(snapshot['preemptions_delta'])})",
        f"  prefill {fmt(snapshot['prefill_tps']):>6} tok/s   decode {fmt(snapshot['decode_tps']):>6} tok/s"
        f"   requests {fmt(snapshot['requests_ps'], '{:.2f}')} /s",
        f"  ttft p50 {fmt(snapshot['ttft_p50'], '{:.2f}s')}  p99 {fmt(snapshot['ttft_p99'], '{:.2f}s')}"
        f"   e2e p50 {fmt(snapshot['e2e_p50'], '{:.1f}s')}  p99 {fmt(snapshot['e2e_p99'], '{:.1f}s')}"
        f"   (client timeout {timeout_s}s)",
    ]
    if gpu:
        lines.append(f"  gpu util {gpu['gpu_util']:.0f} %   mem "
                     f"{gpu['gpu_mem_used_gb']:.1f}/{gpu['gpu_mem_total_gb']:.1f} GB")
    else:
        lines.append("  gpu n/a")
    lines.append("")
    lines.append(f"  {'pid':>7}  {'experiment':<22}{'slug':<28}{'step':>11}"
                 f"  {'inflight':>8}  {'reqs':>5}  {'err':>4}  {'last ms':>8}")
    if not runs:
        lines.append("  (no active LLM runs)")
    for run in runs:
        step = f"{run.get('sim_step', 0)}/{run.get('sim_steps_total') or '?'}"
        lines.append(
            f"  {run.get('pid', ''):>7}  {str(run.get('experiment') or '-'):<22.22}"
            f"{str(run.get('slug') or '-'):<28.28}{step:>11}"
            f"  {run.get('inflight', 0):>8}  {run.get('requests_total', 0):>5}"
            f"  {run.get('errors_total', 0):>4}  {fmt(run.get('latency_ms_last')):>8}")
    if runs:
        lines.append(f"  {'':>7}  {'total':<50}{'':>11}"
                     f"  {sum(r.get('inflight', 0) for r in runs):>8}"
                     f"  {sum(r.get('requests_total', 0) for r in runs):>5}"
                     f"  {sum(r.get('errors_total', 0) for r in runs):>4}")
    return "\n".join(lines)


# --- entry point -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url",
                        default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL),
                        help="the client's base URL; /metrics is derived from it")
    parser.add_argument("--logs-dir", default=None,
                        help="logs root the runs were started with (default logs/)")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="print one poll and exit")
    parser.add_argument("--jsonl", default=None, help="append every poll as one JSON line")
    return parser.parse_args()


def main():
    args = parse_args()
    poller = ServerPoller(metrics_url(args.base_url))
    active = llm_activity.active_dir(args.logs_dir)
    clear_screen = sys.stdout.isatty() and not args.once
    while True:
        try:
            snapshot = poller.poll()
        except (OSError, ValueError) as exc:
            print(f"[{datetime.now():%H:%M:%S}] cannot read {poller.url}: {exc}")
            if args.once:
                sys.exit(1)
            time.sleep(args.interval)
            continue
        gpu = read_gpu()
        runs = llm_activity.read_all(active)
        if clear_screen:
            print("\x1b[H\x1b[2J", end="")
        print(render(snapshot, gpu, runs))
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(snapshot, **(gpu or {}), runs=runs)) + "\n")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
