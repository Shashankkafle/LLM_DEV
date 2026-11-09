"""PhaseHandler.green_source: a green's length is read when the green starts
(end of all-red), not at the decision that chose the phase.

    uv run --with pytest python -m pytest tests/test_phase_handler_green_source.py
    python tests/test_phase_handler_green_source.py     # without pytest

Fake env; no SUMO.
"""
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from utils.phase_handler import PhaseHandler  # noqa: E402

CONF = INTERSECTION_CONFIGS["three_lane"]
TRANSITION_STEPS = (CONF["global_settings"]["yellow_duration"]
                    + CONF["global_settings"]["red_duration"])


def make_handler():
    env = SimpleNamespace(set_phase=lambda **kwargs: None, get_current_step=lambda: 0)
    recorder = SimpleNamespace(record_phase_change=lambda **kwargs: None)
    handler = PhaseHandler(env=env, conf=CONF, intersection_id="tls",
                           start_phase="ETWT", replay_recorder=recorder)
    handler.switch_phase = True
    return handler


def run_transition(handler):
    for _ in range(TRANSITION_STEPS):
        handler.step()


def test_without_source_the_decision_length_is_used():
    handler = make_handler()
    handler.activate_phase("NTST", green=20)
    run_transition(handler)
    assert handler.current_phase_type == "GREEN" and handler.duration == 20


def test_source_is_asked_at_green_start_and_wins():
    handler = make_handler()
    calls = []
    handler.green_source = lambda tls, phase: calls.append((tls, phase)) or 45
    handler.activate_phase("NTST", green=20)
    assert calls == []
    for _ in range(TRANSITION_STEPS - 1):
        handler.step()
    assert calls == []
    handler.step()
    assert calls == [("tls", "NTST")]
    assert handler.duration == 45 and handler.green_duration == 45


def test_source_returning_none_keeps_the_decision_length():
    handler = make_handler()
    handler.green_source = lambda tls, phase: None
    handler.activate_phase("NTST", green=20)
    run_transition(handler)
    assert handler.duration == 20


def test_source_length_is_clamped():
    handler = make_handler()
    handler.green_source = lambda tls, phase: 999
    handler.activate_phase("NTST", green=20)
    run_transition(handler)
    assert handler.duration == CONF["global_settings"]["max_green"]


if __name__ == "__main__":
    failed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
