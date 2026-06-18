import json
import time
from configurations import MIN_SPEED
import traci
from pathlib import Path
from datetime import datetime


class MetricsRecorder:
    def __init__(self,run_dir, verbose=True):
        self.verbose = verbose


        self.run_dir  = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.step_log_path = self.run_dir / "step_summaries.jsonl"
        self.vehicle_data = {}
        self._departure_times = {}
        self.queue_lengths = []
        self.total_decisions = 0
        self.total_hallucinations = 0



    def get_summary_from_vehicle_data(self):
        total_travel_time = sum(v["travel_time"] for v in self.vehicle_data.values())
        total_waiting_time = sum(v["waiting_time"] for v in self.vehicle_data.values())
        total_vehicles = len(self.vehicle_data)
        return {
            "total_completed_vehicles": total_vehicles,
            "average_travel_time_s": round(total_travel_time / total_vehicles, 2) if total_vehicles > 0 else None,
            "average_waiting_time_s": round(total_waiting_time / total_vehicles, 2) if total_vehicles > 0 else None,
        }
    
    def record_step_summary(self, step):
        current_time = traci.simulation.getTime()
        current_queue_length = 0
        for vehicle in traci.vehicle.getIDList():
            
            if vehicle not in self._departure_times:
                self._departure_times[vehicle] = current_time
            depart_time = self._departure_times[vehicle]        
            if depart_time is not None:
                travel_time = current_time - depart_time
                waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle)
                self.vehicle_data[vehicle] = {
                    "travel_time": travel_time,
                    "waiting_time": waiting_time,
                }
            
            if traci.vehicle.getSpeed(vehicle) < MIN_SPEED: 
                current_queue_length += 1 
        
        self.queue_lengths.append(current_queue_length)
        summary = self.get_summary_from_vehicle_data()
        summary["queue_length"] = current_queue_length 
        summary["step"] = step
        with open(self.step_log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def get_final_summary(self):
        """
        Returns overall episode-level metrics after the simulation ends.
        Call this once after the simulation loop exits.
        """
        arrived_vehicles = traci.simulation.getArrivedNumber()
        total_vehicles = len(self.vehicle_data)
        return {
            "total_completed_vehicles": arrived_vehicles,
            "average_travel_time_s": round(sum(v["travel_time"] for v in self.vehicle_data.values()) / total_vehicles, 2) if total_vehicles > 0 else None,
            "average_waiting_time_s": round(sum(v["waiting_time"] for v in self.vehicle_data.values()) / total_vehicles, 2) if total_vehicles > 0 else None,
            "average_queue_length": round(sum(self.queue_lengths) / len(self.queue_lengths), 2) if self.queue_lengths else None,
            "total_decisions": self.total_decisions,
            "total_hallucinations": self.total_hallucinations,
            "hallucination_rate": round(self.total_hallucinations / self.total_decisions, 4) if self.total_decisions > 0 else None,
        }

    def save_final_summary(self):
        """Writes the episode summary JSON to disk and prints it."""
        summary = self.get_final_summary()
        out_path = self.run_dir / "final_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("\n===== FINAL SIMULATION SUMMARY =====")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  Saved to: {out_path}")
        return summary

    def record_decision(self, step, state_dict, prompt, llm_output,
                        
                        previous_phase, final_phase, fallback_applied,
                        latency_ms, extracted_signal, intersection_id):
        self.total_decisions += 1
        if  extracted_signal not in ["NTST", "ETWT", "NLSL", "ELWL"] and extracted_signal != None:
            self.total_hallucinations += 1

        decision_event = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "phase_decision",
            "traffic_state": state_dict.get("movement_states", {}),
            "llm_input": {"user_prompt": prompt},
            "llm_output": {
                "raw_text": llm_output,
                "extracted_signal": extracted_signal,
                "parsing_valid": extracted_signal is not None,
            },
            "phase_action": {
                "requested_phase": extracted_signal,
                "previous_phase": previous_phase,
                "phase_changed": previous_phase != final_phase,
                "activated_phase": final_phase,
                "fallback_applied": fallback_applied,
            },
            "metrics": {
                "inference_latency_ms": round(latency_ms, 2),
                "hallucination_rate": round(
                    self.total_hallucinations / self.total_decisions, 2
                ),
            },
        }

        decision_log_file = self.run_dir / f"{intersection_id}/decisions.jsonl"
        decision_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_log_file, "a") as f:
            f.write(json.dumps(decision_event) + "\n")

        if self.verbose:
            print(f"\n--- Decision @ Step {step} ---")
            print(f"  Extracted: {extracted_signal} | Applied: {final_phase} | Fallback: {fallback_applied}")
            print(f"  Latency: {latency_ms:.1f}ms | Hallucination Rate: {decision_event['metrics']['hallucination_rate']*100:.1f}%")

