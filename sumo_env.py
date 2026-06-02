import traci

class SumoEnv:
    def __init__(self,sumo_config,steps):
        self.sumo_config = sumo_config
        self.steps = steps

    def start_simulation(self):
        traci.start(self.sumo_config)

    def step(self, step):
        traci.simulationStep()
    
    def set_phase(self, intersection_id, phase_id):
        traci.trafficlight.setPhase(intersection_id, phase_id)
    
        

