"""Per-run heartbeat so a monitor beside the vLLM server can tell which run is
making requests.

Several SUMO runs share one server, and the server's /metrics are server-wide:
they can say three requests are running but never whose. So each LLM run
writes one small JSON file -- <logs>/_active/<pid>.json -- with its identity
and live request counts, and monitor_llm.py reads them all.

Nothing here is on a hot path worth worrying about: a request starts or ends a
few times per second at most, and each write is one tiny file replaced
atomically. When no run has started (tests, other controllers) every call is a
no-op.
"""

import contextlib
import json
import os
import threading
import time
from pathlib import Path

from configurations import LOGS_DIR_NAME

ACTIVE_DIR_NAME = "_active"
STEP_WRITE_INTERVAL_S = 1.0

_beacon = None


class _Beacon:
    def __init__(self, path, info):
        self.path = path
        self.info = info
        self.lock = threading.Lock()
        self.inflight = 0
        self.requests_total = 0
        self.errors_total = 0
        self.latency_ms_last = None
        self.sim_step = 0
        self._last_write = 0.0

    def write(self):
        record = dict(self.info, pid=os.getpid(), updated_at=time.time(),
                      sim_step=self.sim_step, inflight=self.inflight,
                      requests_total=self.requests_total,
                      errors_total=self.errors_total,
                      latency_ms_last=self.latency_ms_last)
        # Write-then-rename so the monitor never reads a half-written file.
        # Best-effort: a failed write (e.g. Windows refusing to replace a file
        # the monitor has open) must never fail the run; the next write
        # catches up.
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(record), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            return
        self._last_write = time.time()


def active_dir(logs_dir=None):
    return Path(logs_dir or LOGS_DIR_NAME) / ACTIVE_DIR_NAME


def start(run_dir, logs_dir, info):
    """Begin announcing this process. info holds the static fields shown by
    the monitor (experiment, slug, model, base_url, sim_steps_total)."""
    global _beacon
    directory = active_dir(logs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _beacon = _Beacon(directory / f"{os.getpid()}.json",
                      dict(info, run_dir=str(run_dir)))
    _beacon.write()


@contextlib.contextmanager
def track():
    """Wrap one completion request. Counts it as in flight while it runs and
    as an error if it raises."""
    if _beacon is None:
        yield
        return
    started = time.perf_counter()
    with _beacon.lock:
        _beacon.inflight += 1
        _beacon.write()
    ok = False
    try:
        yield
        ok = True
    finally:
        with _beacon.lock:
            _beacon.inflight -= 1
            _beacon.requests_total += 1
            if ok:
                _beacon.latency_ms_last = round(
                    (time.perf_counter() - started) * 1000, 1)
            else:
                _beacon.errors_total += 1
            _beacon.write()


def set_step(step):
    """Record simulation progress, writing at most once a second so the file
    also stays fresh while no request is running."""
    if _beacon is None:
        return
    with _beacon.lock:
        _beacon.sim_step = step
        if time.time() - _beacon._last_write >= STEP_WRITE_INTERVAL_S:
            _beacon.write()


def close():
    global _beacon
    if _beacon is None:
        return
    with _beacon.lock:
        _beacon.path.unlink(missing_ok=True)
    _beacon = None


def read_all(directory, stale_after_s=60.0):
    """Every live heartbeat in the directory. A file not updated within
    stale_after_s belongs to a run that died without close(); it is removed."""
    directory = Path(directory)
    if not directory.exists():
        return []
    live = []
    for path in directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if time.time() - record.get("updated_at", 0) > stale_after_s:
            path.unlink(missing_ok=True)
            continue
        live.append(record)
    return sorted(live, key=lambda r: r.get("pid", 0))
