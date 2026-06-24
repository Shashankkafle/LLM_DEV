import json
class ReplayRecorder:
    def __init__(self,simulation_config_path, record_dir ):
        self.replay_record = {} 
        self.simulation_config_path = simulation_config_path
        self.record_dir = record_dir

    def record_phase_change(self, step, intersection_id, phase, phase_name,phase_from_sumo=None):
        """
        Records one intersection's phase change at a given simulation step.

        Several intersections can change phase on the same step, so each step
        holds a list of events. The previous version keyed a single event by
        step alone, which silently dropped every event but the last one written
        for that step (most intersections never made it into the replay).
        """
        event = {
            "intersection_id": intersection_id,
            "phase": phase,
            "phase_name": phase_name,
            "phase_from_sumo": phase_from_sumo
        }
        self.replay_record.setdefault(step, []).append(event)

    def save_replay_data(self, original_run_details):
        self.replay_record["original_run_details"] = original_run_details
        with open(f"{self.record_dir}/replay_record.json", "w") as f:
            json.dump(self.replay_record, f, indent=4)