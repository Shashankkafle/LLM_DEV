import traci

class SumoEnv:
    def __init__(self, sumo_config, use_gui=False):
        self.sumo_config = sumo_config
        self.use_gui = use_gui
    def start_simulation(self):
        if self.use_gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"
        cmd = [sumo_binary, "-c", self.sumo_config]
        traci.start(cmd)

    def step(self):
        traci.simulationStep()
    
    def set_phase(self, intersection_id, phase_config):
        try:
            traci.trafficlight.setRedYellowGreenState(intersection_id, phase_config)
        except Exception as e:
            print(f"Error occurred while setting phase for intersection {intersection_id}: {e}")

    def close(self):
        traci.close()

    def get_intersections(self):
        return traci.trafficlight.getIDList()
    
    def get_state(self, intersection_id):
        state = {}
        state["current_phase"] = traci.trafficlight.getPhase(intersection_id)
        state["lane_states"] = {}
        
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(intersection_id)))
        # todo:move this to configurations.py
        v_stop = 1.39
        
        for lane_id in controlled_lanes:
            lane_length = traci.lane.getLength(lane_id)
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            
            # 1. Use explicit variables instead of lists to keep Pylance happy
            early_queued_count = 0
            segment_1_count = 0
            segment_2_count = 0
            segment_3_count = 0
            
            for veh_id in veh_ids:
                speed = traci.vehicle.getSpeed(veh_id)
                
                if speed < v_stop:
                    early_queued_count += 1
                else:
                    pos_from_start = traci.vehicle.getLanePosition(veh_id)
                    distance_to_stopline = max(0.0, lane_length - pos_from_start)
                    
                    # Divide lane into three equal chunks
                    seg_length = lane_length / 3.0
                    
                    # 2. Explicitly assign vehicles to segments using simple if/elif logic
                    if distance_to_stopline <= seg_length:
                        segment_1_count += 1
                    elif distance_to_stopline <= (2 * seg_length):
                        segment_2_count += 1
                    else:
                        segment_3_count += 1
            
            # 3. Build the final dictionary explicitly with no shortcuts
            state["lane_states"][lane_id] = {
                "early_queued": early_queued_count,
                "segments": {
                    "segment_1": segment_1_count,
                    "segment_2": segment_2_count,
                    "segment_3": segment_3_count
                }
            }
            
        return state