from configurations import DEFAULT_START_PHASE

PHASE_TYPES = ["ALL_RED", "GREEN", "YELLOW"]

# handler to keep track of how long the current phase has been active and when to switch to the next phase
class PhaseHandler:
    def __init__(self,env, conf, intersection_id,start_phase=DEFAULT_START_PHASE,replay_recorder=None):

        self.switch_phase = False
        self.current_phase_type = "GREEN"
        self.next_phase = None
        self.conf = conf
        self.replay_recorder = replay_recorder
        self.phases = self.conf["phases"].keys()
        self.duration = self.conf["global_settings"]["default_green_duration"]
        self.green_duration = self.conf["global_settings"]["default_green_duration"]
        self.env = env
        self.intersection_id = intersection_id
        # The last signal state this handler sent to SUMO. None until it sends
        # one: until then SUMO runs the net file's own program, whatever
        # current_phase says.
        self.last_state_set = None
        if start_phase in self.phases:
            self.current_phase = start_phase
        else:

            self.current_phase = list(self.phases)[0]  # default to the first phase if start_phase is not valid

    def take_control(self):
        """Send the start phase's green to SUMO. Call once, before the first
        simulation step, so the net file's program never runs."""
        self._set_state(self.conf["phases"][self.current_phase]["green"], initial=True)

    def assert_signal_is_ours(self):
        """Raise if SUMO shows a state other than the last one this handler
        set. A handler that has not set a state yet has nothing to check."""
        if self.last_state_set is None:
            return
        actual_state = self.env.get_current_phase(self.intersection_id)
        if actual_state != self.last_state_set:
            raise RuntimeError(
                f"{self.intersection_id} at t={self.env.get_current_step()}: SUMO "
                f"shows '{actual_state}', but the controller last set "
                f"'{self.last_state_set}' ({self.current_phase} {self.current_phase_type}).")

    def step(self):
        if self.current_phase_type == "ALL_RED":
            self.duration -= 1
            if self.duration <= 0:
                self.current_phase_type = "GREEN"
                if self.next_phase is None:
                    raise ValueError("Next phase has not been set before switching from ALL_RED to GREEN.")
                self.current_phase = self.next_phase
                self.duration = self.green_duration
                self._set_state(self.conf["phases"][self.current_phase]["green"])

        elif self.current_phase_type == "YELLOW":
            self.duration -= 1
            if self.duration <= 0:
                self.duration = self.conf["global_settings"]["red_duration"]
                self.current_phase_type = "ALL_RED"
                self._set_state(self.conf["global_settings"]["all_red_state"])
        elif self.current_phase_type == "GREEN":
            self.duration -= 1
            if self.duration <= 0:
                self.switch_phase = True
        else:
            raise ValueError(f"Unknown phase type: {self.current_phase_type}")

    def activate_phase(self, new_phase, green=None):
        if not self.switch_phase:
            raise ValueError("Cannot switch phase yet. Current phase duration has not ended.")

        if new_phase not in self.phases:
            raise ValueError(f"Phase {new_phase} is not a valid phase.")

        global_settings = self.conf["global_settings"]
        if green is None:
            green = global_settings["default_green_duration"]
        self.green_duration = min(max(green, global_settings["min_green"]), global_settings["max_green"])

        if new_phase == self.current_phase:
            self.duration = self.green_duration  # Reset duration if the same phase is activated again
            self._reapply_green_if_not_shown()

        if new_phase != self.current_phase:
            # Sent before current_phase_type changes: the replay record has
            # always named the yellow event "<phase>_GREEN".
            self._set_state(self.conf["phases"][self.current_phase]["yellow"])

            self.current_phase_type = "YELLOW"
            self.next_phase = new_phase
            self.duration = self.conf["global_settings"]["yellow_duration"]

        self.switch_phase = False

    def _set_state(self, state, initial=False):
        self.env.set_phase(intersection_id=self.intersection_id, phase_config=state,
                           initial=initial)
        self.replay_recorder.record_phase_change(
            step=self.env.get_current_step(), intersection_id=self.intersection_id,
            phase=state, phase_name=f"{self.current_phase}_{self.current_phase_type}")
        self.last_state_set = state

    def _reapply_green_if_not_shown(self):
        """Holding a phase sends nothing to SUMO, so it only holds what SUMO
        already shows. If that is not this phase's green, send it."""
        green_state = self.conf["phases"][self.current_phase]["green"]
        if self.env.get_current_phase(self.intersection_id) != green_state:
            self._set_state(green_state)
