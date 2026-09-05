"""Offline checks for monitor_llm.py and utils/llm_activity.py, plus the
per-request latency the vLLM client now reports. No GPU or server needed:
everything runs against tests/fake_vllm.FakeVLLM.

Run: python tests/smoke_monitor_llm.py
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from tests.fake_vllm import FakeVLLM, SERVED  # noqa: E402

fake = FakeVLLM()
os.environ["VLLM_BASE_URL"] = fake.base_url

import monitor_llm  # noqa: E402
from utils import llm_activity  # noqa: E402
from models_inference.LLM.http_llm import VLLMLLM  # noqa: E402

failures = []


def check(cond, msg):
    print(f"[{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


# --- 1. Prometheus parsing ----------------------------------------------------

metrics = monitor_llm.parse_metrics(fake.metrics_text())
check(monitor_llm.metric_value(metrics, monitor_llm.RUNNING) == 0.0,
      "a labelled gauge parses to its value")
check(monitor_llm.metric_value(metrics, ("vllm:missing", "vllm:num_requests_waiting")) == 0.0,
      "metric_value falls through to the first present name")
check(monitor_llm.metric_value(metrics, ("vllm:missing",)) is None,
      "a family that is not exposed reads as None, not an error")
check("this line is deliberately malformed" not in metrics,
      "a malformed line is skipped")
check(metrics["vllm:something_nan"][0][1] != metrics["vllm:something_nan"][0][1],
      "NaN parses as NaN rather than breaking the page")
check(metrics["vllm:request_success_total"][0][0] == {"finished_reason": "stop", "model_name": SERVED},
      "labels with several pairs parse into a dict")

buckets = {0.1: 10.0, 0.5: 50.0, 1.0: 90.0, monitor_llm.INF: 100.0}
check(abs(monitor_llm.quantile(buckets, 0.5) - 0.5) < 1e-9,
      "quantile interpolates within the bucket holding the target count")
check(monitor_llm.quantile(buckets, 0.99) == 1.0,
      "a quantile landing in +Inf reports the last finite bound")
check(monitor_llm.quantile({}, 0.5) is None, "an empty histogram has no quantile")

# --- 2. the poller and the rendered page --------------------------------------

poller = monitor_llm.ServerPoller(monitor_llm.metrics_url(fake.base_url))
first = poller.poll()
check(first["decode_tps"] is None, "rates are unknown on the first poll")
check(first["kv_pct"] == 6.1, "kv usage is shown as a percentage")
text = monitor_llm.render(first, None, [])
check("running    0" in text and "kv   6.1 %" in text and "gpu n/a" in text,
      "the page renders running / kv, and gpu as n/a when nvidia-smi is absent")

llm = VLLMLLM(f"vllm:{SERVED}")
llm.initialize_llm()
llm.inference_batch(["a", "b", "c"])
time.sleep(0.05)
second = poller.poll()
check(second["decode_tps"] is not None and second["decode_tps"] > 0,
      "the second poll reports a positive token rate from the counter delta")
check(second["ttft_p50"] is not None,
      "windowed percentiles come from the bucket delta between polls")

fake.metrics_kv = False
degraded = poller.poll()
check(degraded["kv_pct"] is None and "kv   n/a %" in monitor_llm.render(degraded, None, []),
      "a server without the kv family still renders, with n/a")
fake.metrics_kv = True

# --- 3. heartbeat --------------------------------------------------------------

tmp = Path(tempfile.mkdtemp())
active = llm_activity.active_dir(tmp)

with llm_activity.track():
    pass
check(not active.exists(), "track() is a no-op before start()")

llm_activity.start(tmp / "run", tmp, {"experiment": "e", "slug": "s",
                                      "model": SERVED, "base_url": fake.base_url,
                                      "sim_steps_total": 100})
beat = active / f"{os.getpid()}.json"
check(beat.exists(), "start() writes <logs>/_active/<pid>.json")


def hammer():
    for _ in range(100):
        with llm_activity.track():
            pass


threads = [threading.Thread(target=hammer) for _ in range(32)]
partial_reads = 0
stop_reading = threading.Event()


def reader():
    """Reads the heartbeat continuously. A half-written file would fail to
    parse; an OSError is only Windows refusing a read mid-replace, which
    read_all() skips."""
    global partial_reads
    while not stop_reading.is_set():
        try:
            json.loads(beat.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            partial_reads += 1
        except OSError:
            pass


reader_thread = threading.Thread(target=reader)
reader_thread.start()
for t in threads:
    t.start()
for t in threads:
    t.join()
stop_reading.set()
reader_thread.join()

# A write refused while the reader held the file is dropped; the next
# set_step (at most a second later in a run) rewrites the file.
time.sleep(llm_activity.STEP_WRITE_INTERVAL_S)
llm_activity.set_step(1)
record = json.loads(beat.read_text(encoding="utf-8"))
check(record["inflight"] == 0 and record["requests_total"] == 3200,
      "32 threads x 100 tracked requests count exactly, ending with nothing in flight")
check(partial_reads == 0, "a concurrent reader never sees a partial file")

try:
    with llm_activity.track():
        raise RuntimeError("boom")
except RuntimeError:
    pass
record = json.loads(beat.read_text(encoding="utf-8"))
check(record["inflight"] == 0 and record["errors_total"] == 1,
      "a request that raises is released and counted as an error")

llm_activity.set_step(42)
check(json.loads(beat.read_text(encoding="utf-8"))["sim_step"] == 1,
      "set_step is rate-limited to one write per second")

stale = active / "99999.json"
stale.write_text(json.dumps({"pid": 99999, "updated_at": time.time() - 120}))
live = llm_activity.read_all(active)
check([r["pid"] for r in live] == [os.getpid()] and not stale.exists(),
      "read_all keeps the fresh heartbeat and removes the stale one")
check("e" in monitor_llm.render(first, None, live) and "total" in monitor_llm.render(first, None, live),
      "the page lists the live run and a totals line")

llm_activity.close()
check(not beat.exists(), "close() removes the heartbeat")
llm_activity.close()
check(True, "a second close() is harmless")

# --- 4. per-request latency ----------------------------------------------------

fake.delay_s = 0.2
llm.inference_batch(["a", "b", "c", "d"])
latencies = [u["latency_ms"] for u in llm.last_usage_batch]
check(all(l >= 190 for l in latencies), "each request's latency covers the server delay")
check(len(set(latencies)) > 1,
      "batched requests report their own latency, not one shared wall clock")
fake.delay_s = 0.0

fake.shutdown()
print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
