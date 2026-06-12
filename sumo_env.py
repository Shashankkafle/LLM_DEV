import traci
from configurations import INTERSECTION_CONFIG 
class SumoEnv:
    def __init__(self, sumo_config, use_gui=False):
        self.sumo_config = sumo_config
        self.use_gui = use_gui
        if self.use_gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"
        self.cmd = [sumo_binary, "-c", self.sumo_config]

    def _build_movement_lane_map(self, intersection_id):
        """
        Maps each phase movement key (e.g. 'ELWL') to the set of from-lane IDs
        that are given a protected green (G) in that phase.
        
        Returns:
            {
                "ETWT": {"E_through_lane_id", "W_through_lane_id"},
                "ELWL": {"E_left_lane_id",   "W_left_lane_id"},
                ...
            }
        """
        controlled_links = traci.trafficlight.getControlledLinks(intersection_id)

        movement_lane_map = {}

        for phase_name, phase_cfg in INTERSECTION_CONFIG["phases"].items():
            phase_string = phase_cfg["green"]
            lanes_for_movement = set()

            for i, char in enumerate(phase_string):
                if char == 'G':  # Only protected green, not permissive 'g'
                    links = controlled_links[i]
                    for (from_lane, to_lane, via) in links:
                        lanes_for_movement.add(from_lane)

            movement_lane_map[phase_name] = lanes_for_movement

        return movement_lane_map

    def step(self):
        traci.simulationStep()

    def start_simulation(self):
        traci.start(self.cmd)
        self.intersection_ids = traci.trafficlight.getIDList()
        self.movement_lane_map = {
        intersection_id: self._build_movement_lane_map(intersection_id)
        for intersection_id in self.intersection_ids
        }
        print("Initialized SumoEnv with the following movement-lane mapping:", self.movement_lane_map)

    
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
        state["movement_states"] = {}   # <-- new

        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(intersection_id)))
        v_stop = 1.39

        lane_data = {}  

        for lane_id in controlled_lanes:
            lane_length = traci.lane.getLength(lane_id)
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)

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
                    seg_length = lane_length / 3.0

                    if distance_to_stopline <= seg_length:
                        segment_1_count += 1
                    elif distance_to_stopline <= (2 * seg_length):
                        segment_2_count += 1
                    else:
                        segment_3_count += 1

            lane_data[lane_id] = {
                "early_queued": early_queued_count,
                "segments": {
                    "segment_1": segment_1_count,
                    "segment_2": segment_2_count,
                    "segment_3": segment_3_count
                }
            }

        state["lane_states"] = lane_data

        # --- Movement-grouped counts (aggregated over the movement's lanes) ---
        for movement_name, lane_ids in self.movement_lane_map[intersection_id].items():
            agg = {
                "early_queued": 0,
                "segments": {"segment_1": 0, "segment_2": 0, "segment_3": 0},
                "lanes": {}   # per-lane breakdown kept for debugging
            }

            for lane_id in lane_ids:
                if lane_id not in lane_data:
                    continue  

                d = lane_data[lane_id]
                agg["early_queued"]          += d["early_queued"]
                agg["segments"]["segment_1"] += d["segments"]["segment_1"]
                agg["segments"]["segment_2"] += d["segments"]["segment_2"]
                agg["segments"]["segment_3"] += d["segments"]["segment_3"]
                agg["lanes"][lane_id]         = d   
            state["movement_states"][movement_name] = agg

        return state