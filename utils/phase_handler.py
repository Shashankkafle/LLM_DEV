PHASE_TYPES = ["ALL_RED", "GREEN", "YELLOW"]

# handler to keep track of how long the current phase has been active and when to switch to the next phase
class PhaseHandler:
    def __init__(self,env, conf, intersection_id,start_phase='ETWT'):
        
        self.switch_phase = False
        self.current_phase_type = "GREEN"
        self.next_phase = None
        self.conf = conf
        self.phases = self.conf["phases"].keys()
        self.duration = self.conf["global_settings"]["default_green_duration"]
        self.env = env
        self.intersection_id = intersection_id
        if start_phase in self.phases:
            self.current_phase = start_phase
        else: 
            
            self.current_phase = list(self.phases)[0]  # default to the first phase if start_phase is not valid

    
    def step(self):
        print(f"PhaseHandler Step - Intersection: {self.intersection_id}, Current Phase: {self.current_phase}, Phase Type: {self.current_phase_type}, Duration Remaining: {self.duration}")
        # This method can be called at each simulation step to check if it's time to switch phases
        if self.current_phase_type == "ALL_RED":
            self.duration -= 1
            if self.duration <= 0:
                self.current_phase_type = "GREEN"
                self.duration = self.conf["global_settings"]["default_green_duration"]
                if self.next_phase is None:
                    raise ValueError("Next phase has not been set before switching from ALL_RED to GREEN.")
                self.current_phase = self.next_phase
                self.env.set_phase(intersection_id=self.intersection_id, phase_config=self.conf["phases"][self.current_phase]["green"])
        if self.current_phase_type == "YELLOW":
            self.duration -= 1
            if self.duration <= 0:
                self.duration = self.conf["global_settings"]["red_duration"]
                self.current_phase_type = "ALL_RED"
                self.env.set_phase(intersection_id=self.intersection_id, phase_config=self.conf["global_settings"]["all_red_state"])
        if self.current_phase_type == "GREEN":
            self.duration -= 1
            if self.duration <= 0:
                self.switch_phase = True  
    
    def activate_phase(self, new_phase):
        print("Activating phase:", new_phase, "for intersection:", self.intersection_id," with current phase:", self.current_phase, "and phase type:", self.current_phase_type)
        if not self.switch_phase:
            raise ValueError("Cannot switch phase yet. Current phase duration has not ended.")
        
        if new_phase not in self.phases:
            raise ValueError(f"Phase {new_phase} is not a valid phase.")
        
        if new_phase == self.current_phase:
            self.duration = self.conf["global_settings"]["default_green_duration"]  # Reset duration if the same phase is activated again
        
        if new_phase != self.current_phase:
            print(f"Switching from phase {self.current_phase} to {new_phase}")
            self.env.set_phase(intersection_id=self.intersection_id, phase_config=self.conf["phases"][self.current_phase]["yellow"])
            self.current_phase_type = "YELLOW"
            self.next_phase = new_phase
            self.duration = self.conf["global_settings"]["yellow_duration"]  
            
        self.switch_phase = False



        



    def switch_phase(self):
        return self._switch_phase
    
    def current_phase(self):
        return self._current_phase
    
            

