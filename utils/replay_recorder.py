import json
class ReplayRecorder:
    def __init__(self,simulation_config_path, record_dir ):
        self.replay_record = {} 
        self.simulation_config_path = simulation_config_path
        self.record_dir = record_dir

    def record_phase_change(self, step, intersection_id, phase, phase_name,phase_from_sumo=None):
        """
        Records the necessary information to replay the simulation later.
        This can include the sequence of states, actions, and any random seeds used.
        """
        self.replay_record[step] = {
            "intersection_id": intersection_id,
            "phase": phase,
            "phase_name": phase_name,
            "phase_from_sumo": phase_from_sumo
        }

    def save_replay_data(self):
        with open(f"{self.record_dir}/replay_record.json", "w") as f:
            json.dump(self.replay_record, f, indent=4)