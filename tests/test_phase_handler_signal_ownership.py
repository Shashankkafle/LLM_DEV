"""PhaseHandler owns the signal: take_control sets the start green at t=0, a
hold re-sends the green if SUMO shows something else, and
assert_signal_is_ours fails when SUMO shows a state the handler did not set.

    uv run --with pytest python -m pytest tests/test_phase_handler_signal_ownership.py
    python tests/test_phase_handler_signal_ownership.py     # without pytest

Fake env; no SUMO.
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configurations import INTERSECTION_CONFIGS  # noqa: E402
from utils.phase_handler import PhaseHandler  # noqa: E402

CONF = INTERSECTION_CONFIGS["three_lane"]
ETWT_GREEN = CONF["phases"]["ETWT"]["green"]
ETWT_YELLOW = CONF["phases"]["ETWT"]["yellow"]
NET_FILE_STATE = "GGGrrrrrrGGGGGGrrrGGGrrrrrrGGGGGGrrr"
GREEN_STEPS = CONF["global_settings"]["default_green_duration"]


class FakeEnv:
    """Starts on the net file's state, like SUMO before any set_phase."""

    def __init__(self):
        self.state = NET_FILE_STATE
        self.set_calls = []

    def set_phase(self, intersection_id, phase_config, initial=False):
        self.set_calls.append((phase_config, initial))
        self.state = phase_config

    def get_current_phase(self, intersection_id):
        return self.state

    def get_current_step(self):
        return 0


class FakeReplayRecorder:
    def __init__(self):
        self.events = []

    def record_phase_change(self, step, intersection_id, phase, phase_name):
        self.events.append((step, phase, phase_name))


def make_handler():
    env = FakeEnv()
    recorder = FakeReplayRecorder()
    handler = PhaseHandler(env=env, conf=CONF, intersection_id="tls",
                           start_phase="ETWT", replay_recorder=recorder)
    return handler, env, recorder


def run_green(handler):
    for _ in range(GREEN_STEPS):
        handler.step()
    assert handler.switch_phase


def expect_raises(action):
    try:
        action()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_take_control_sets_the_start_green_as_the_initial_state():
    handler, env, recorder = make_handler()
    handler.take_control()
    assert env.set_calls == [(ETWT_GREEN, True)]
    assert recorder.events == [(0, ETWT_GREEN, "ETWT_GREEN")]
    assert handler.last_state_set == ETWT_GREEN


def test_hold_after_take_control_sends_nothing():
    handler, env, _ = make_handler()
    handler.take_control()
    run_green(handler)
    handler.activate_phase("ETWT")
    assert env.set_calls == [(ETWT_GREEN, True)]


def test_hold_resends_the_green_when_sumo_shows_another_state():
    handler, env, _ = make_handler()
    run_green(handler)
    handler.activate_phase("ETWT")
    assert env.set_calls == [(ETWT_GREEN, False)]
    assert env.state == ETWT_GREEN


def test_assertion_passes_while_sumo_shows_the_state_the_handler_set():
    handler, _, _ = make_handler()
    handler.take_control()
    run_green(handler)
    handler.assert_signal_is_ours()
    handler.activate_phase("NTST")
    assert handler.last_state_set == ETWT_YELLOW
    handler.assert_signal_is_ours()


def test_assertion_fails_when_sumo_shows_a_state_the_handler_did_not_set():
    handler, env, _ = make_handler()
    handler.take_control()
    env.state = NET_FILE_STATE
    expect_raises(handler.assert_signal_is_ours)


def test_assertion_checks_nothing_before_the_handler_sets_a_state():
    handler, _, _ = make_handler()
    handler.assert_signal_is_ours()


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
