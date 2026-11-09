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
        # Optional callable(intersection_id, phase) -> green seconds or None,
        # asked when a green starts after a transition. It lets a controller
        # fix a green's length at the green's start instead of at the
        # decision 5 s earlier; None keeps the length given to activate_phase.
        self.green_source = None
        self.env = env
        self.intersection_id = intersection_id
        if start_phase in self.phases:
            self.current_phase = start_phase
        else: 
            
            self.current_phase = list(self.phases)[0]  # default to the first phase if start_phase is not valid

    
    def step(self):
        if self.current_phase_type == "ALL_RED":
            self.duration -= 1
            if self.duration <= 0:
                self.current_phase_type = "GREEN"
                if self.next_phase is None:
                    raise ValueError("Next phase has not been set before switching from ALL_RED to GREEN.")
                self.current_phase = self.next_phase
                self.green_duration = self._green_at_start(self.current_phase)
                self.duration = self.green_duration
                phase_to_set = self.conf["phases"][self.current_phase]["green"]
                self.env.set_phase(intersection_id=self.intersection_id, phase_config=phase_to_set)
                self.replay_recorder.record_phase_change(step=self.env.get_current_step(), intersection_id=self.intersection_id, phase=phase_to_set, phase_name=f"{self.current_phase}_{self.current_phase_type}")

        elif self.current_phase_type == "YELLOW":
            self.duration -= 1
            if self.duration <= 0:
                self.duration = self.conf["global_settings"]["red_duration"]
                self.current_phase_type = "ALL_RED"
                phase_to_set = self.conf["global_settings"]["all_red_state"]
                self.env.set_phase(intersection_id=self.intersection_id, phase_config=phase_to_set)
                self.replay_recorder.record_phase_change(step=self.env.get_current_step(), intersection_id=self.intersection_id, phase=phase_to_set, phase_name=f"{self.current_phase}_{self.current_phase_type}")
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
        self.green_duration = self._clamp_green(green)

        if new_phase == self.current_phase:
            self.duration = self.green_duration  # Reset duration if the same phase is activated again
        
        if new_phase != self.current_phase:
            phase_to_set = self.conf["phases"][self.current_phase]["yellow"]
            
            self.env.set_phase(intersection_id=self.intersection_id, phase_config=phase_to_set)

            self.replay_recorder.record_phase_change(step=self.env.get_current_step(), intersection_id=self.intersection_id, phase=phase_to_set, phase_name=f"{self.current_phase}_{self.current_phase_type}")

            self.current_phase_type = "YELLOW"
            self.next_phase = new_phase
            self.duration = self.conf["global_settings"]["yellow_duration"]

        self.switch_phase = False

    def _clamp_green(self, green):
        global_settings = self.conf["global_settings"]
        return min(max(green, global_settings["min_green"]), global_settings["max_green"])

    def _green_at_start(self, phase):
        if self.green_source is None:
            return self.green_duration
        green = self.green_source(self.intersection_id, phase)
        if green is None:
            return self.green_duration
        return self._clamp_green(green)

