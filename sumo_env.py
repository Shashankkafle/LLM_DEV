import traci
from configurations import (
    INTERSECTION_CONFIG,
    SUMO_BINARY,
    SUMO_GUI_BINARY,
    STOP_SPEED_EARLY_QUEUE,
    PHASE_SEQUENCE_FILENAME_SUFFIX,
    sumo_metrics_args,
)
from utils.general_utils import append_to_json_file, get_phase_name
approaches = ["North", "South", "East", "West"]
movement_to_approach = {
    "ETWT": ["East", "West"],
    "ELWL": ["East", "West"],
    "NTST": ["South", "North"],
    "NLSL": ["South", "North"]
}
class SumoEnv:
    def __init__(self, sumo_config, phase_sequence_dir=None, use_gui=False,
                 intersection_config=INTERSECTION_CONFIG, output_dir=None,
                 seed=None):
        self.sumo_config = sumo_config
        self.use_gui = use_gui
        self.intersection_config = intersection_config
        self.approach_mapping = None
        self.set_phase_sequence_dir = phase_sequence_dir
        if self.use_gui:
            sumo_binary = SUMO_GUI_BINARY
        else:
            sumo_binary = SUMO_BINARY
        self.cmd = [sumo_binary, "-c", self.sumo_config]
        # When an output dir is given, disable teleport and emit
        # tripinfo/queue/statistics so cal_offline can report honest,
        # cross-controller-comparable metrics. SUMO flushes these on close.
        if output_dir is not None:
            self.cmd += sumo_metrics_args(output_dir)
        # None keeps SUMO's fixed default seed (deterministic reruns).
        if seed is not None:
            self.cmd += ["--seed", str(seed)]

    def _build_approach_mapping(self):
        mapping = {j: {} for j in traci.trafficlight.getIDList()}

        for lane_id in traci.lane.getIDList():
            if lane_id.startswith(":"):
                continue
                
            target_junctions = set()
            for link in traci.lane.getLinks(lane_id):
                internal_lanes = link
                for internal_lane in internal_lanes:
                    if isinstance(internal_lane, str) and internal_lane.startswith(":"):
                        for j_id in mapping:
                            if internal_lane.startswith(f":{j_id}_"):
                                target_junctions.add(j_id)
                        
            if target_junctions:
                shape = traci.lane.getShape(lane_id)
                start_point = shape[0]
                end_point = shape[-1]
                dx = start_point[0] - end_point[0]
                dy = end_point[1] - start_point[1]

                if abs(dx) > abs(dy):
                    approach = "East" if dx > 0 else "West"
                else:
                    approach = "South" if dy > 0 else "North"
                    
                for j_id in target_junctions:
                    mapping[j_id][lane_id] = approach

        return mapping

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

        for phase_name, phase_cfg in self.intersection_config["phases"].items():
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
        self.approach_mapping = self._build_approach_mapping()
        mapped_lanes = sum(len(lanes) for lanes in self.approach_mapping.values())
        print(f"SumoEnv initialized: {len(self.intersection_ids)} intersections, "
              f"{mapped_lanes} approach lanes mapped")

    def get_current_step(self):
        return traci.simulation.getCurrentTime() // 1000  # Convert milliseconds to seconds
    
    def set_phase(self, intersection_id, phase_config):
        n_links = len(traci.trafficlight.getControlledLinks(intersection_id))
        if len(phase_config) != n_links:
            raise ValueError(
                f"Phase string length {len(phase_config)} does not match "
                f"controlled link count {n_links} for intersection {intersection_id}. "
                f"Phase string: '{phase_config}'"
            )

        if self.set_phase_sequence_dir:
            log_file = self.set_phase_sequence_dir / f"{intersection_id}{PHASE_SEQUENCE_FILENAME_SUFFIX}"
            previous_phase = self.get_current_phase(intersection_id)
            change_dict = {
                "step": self.get_current_step(),
                "prev_phase": previous_phase,
                "prev_phase_name": get_phase_name(self.intersection_config, previous_phase),
                "new_phase": phase_config,
                "new_phase_name": get_phase_name(self.intersection_config, phase_config),
            }
            append_to_json_file(log_file, change_dict)

        traci.trafficlight.setRedYellowGreenState(intersection_id, phase_config)

        actual = traci.trafficlight.getRedYellowGreenState(intersection_id)
        if actual != phase_config:
           raise ValueError("setRedYellowGreenState did not set the expected phase. Check the log for details.")
        

    def close(self):
        traci.close()

    def get_intersections(self):
        return traci.trafficlight.getIDList()
    
    def get_current_phase(self, intersection_id):
        return traci.trafficlight.getRedYellowGreenState(intersection_id)
    
    def get_state(self, intersection_id):
        state = {}
        state["current_phase"] = traci.trafficlight.getPhase(intersection_id)
        state["lane_states"] = {}
        state["movement_states"] = {}   # <-- new

        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(intersection_id)))
        v_stop = STOP_SPEED_EARLY_QUEUE

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


                    # Seg1 = 0-L/10, Seg2 = L/10-L/3, Seg3 = rest. LightGPT was
                    # fine-tuned on these
                    if distance_to_stopline <= lane_length / 10:
                        segment_1_count += 1
                    elif distance_to_stopline <= lane_length / 3:
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
            agg = {}
            for approach in movement_to_approach[movement_name]:
                agg[approach] = {
                    "early_queued": 0,
                    "segments": {"segment_1": 0, "segment_2": 0, "segment_3": 0},
                    "lanes": {}   # per-lane breakdown kept for debugging
                }

            for lane_id in lane_ids:
                if lane_id not in lane_data:
                    continue  
                
                d = lane_data[lane_id]
                lane_approach = self.approach_mapping[intersection_id].get(lane_id, "Unknown")
                if lane_approach == "Unknown":
                    raise ValueError(f"Lane {lane_id} in movement {movement_name} does not have a known approach direction in the mapping.")
                
                agg[lane_approach]["early_queued"]          += d["early_queued"]
                agg[lane_approach]["segments"]["segment_1"] += d["segments"]["segment_1"]
                agg[lane_approach]["segments"]["segment_2"] += d["segments"]["segment_2"]
                agg[lane_approach]["segments"]["segment_3"] += d["segments"]["segment_3"]
                agg[lane_approach]["lanes"][lane_id]         = d   
            state["movement_states"][movement_name] = agg

        return state