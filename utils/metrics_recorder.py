import json
import time
import re
import traci
from pathlib import Path
from datetime import datetime

class MetricsRecorder:
    def __init__(self, run_name="default_run", base_log_dir="logs", verbose=True):
        """
        Initializes the metrics recorder, creating a dedicated folder for this specific run.
        """
        self.verbose = verbose
        
        # Create a specific sub-directory for this run (e.g., logs/rough_test_20260612_153000/)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_log_dir) / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep filenames simple since they are safely isolated in their own folder
        self.decision_log_path = self.run_dir / "decisions.jsonl"
        self.step_log_path = self.run_dir / "step_summaries.jsonl"
        
        # In-memory tracking for End-to-End metrics (ATT / AWT)
        self.vehicle_tracking = {}  
        
        # Running aggregations
        self.total_completed_vehicles = 0
        self.cumulative_travel_time = 0.0
        self.cumulative_waiting_time = 0.0
        self.total_hallucinations = 0
        self.total_decisions = 0

    def record_step_summary(self, step):
        """
        Records per-step network aggregates (AQL, ATT, AWT). Fast execution.
        """
        current_time = traci.simulation.getTime()
        
        # 1. Track departed vehicles
        for vid in traci.simulation.getDepartedIDList():
            self.vehicle_tracking[vid] = {"depart_time": current_time, "accumulated_wait": 0.0}
            
        # 2. Track arrived vehicles to calculate Travel Time and Waiting Time
        for vid in traci.simulation.getArrivedIDList():
            if vid in self.vehicle_tracking:
                travel_time = current_time - self.vehicle_tracking[vid]["depart_time"]
                # traci.vehicle.getAccumulatedWaitingTime() might not be available for arrived vehicles,
                # so we rely on the last known wait time if we were tracking it, or we calculate it globally.
                self.cumulative_travel_time += travel_time
                self.total_completed_vehicles += 1
                del self.vehicle_tracking[vid]
                
        # 3. Aggregate current network state
        active_vehicles = traci.vehicle.getIDList()
        current_waiting = sum(traci.vehicle.getWaitingTime(vid) for vid in active_vehicles)
        self.cumulative_waiting_time += current_waiting
        
        # AQL: Number of halting vehicles (speed < 0.1 m/s)
        halting_vehicles = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in traci.edge.getIDList())
        
        # Safe averages
        att = (self.cumulative_travel_time / self.total_completed_vehicles) if self.total_completed_vehicles > 0 else 0
        awt = (self.cumulative_waiting_time / max(1, len(active_vehicles) + self.total_completed_vehicles))

        print(f"cumulative_waiting_time: {self.cumulative_waiting_time}, total_vehicles_in_network: {len(active_vehicles)}, total_completed_vehicles: {self.total_completed_vehicles}, att: {att}, awt: {awt}")

        summary = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "step_summary",
            "aggregate_queue_length": halting_vehicles,
            "total_vehicles_in_network": len(active_vehicles),
            "completed_vehicles": self.total_completed_vehicles,
            "average_travel_time": round(att, 2),
            "average_waiting_time": round(awt, 2)
        }
        
        with open(self.step_log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def record_decision(self, step, state_dict, prompt, llm_output, previous_phase, final_phase, fallback_applied, latency_ms,extracted_signal):
        """
        Records the full context of an LLM decision event.
        """
        self.total_decisions += 1
        
        if fallback_applied or extracted_signal not in ["NTST", "ETWT", "NLSL", "ELWL"]:
            self.total_hallucinations += 1
            
        decision_event = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "phase_decision",
            "traffic_state": state_dict.get("movement_states", {}),
            "llm_input": {
                "user_prompt": prompt
            },
            "llm_output": {
                "raw_text": llm_output,
                "extracted_signal": extracted_signal,
                "parsing_valid": extracted_signal is not None
            },
            "phase_action": {
                "requested_phase": extracted_signal,
                "previous_phase": previous_phase,
                "phase_changed": previous_phase != final_phase,
                "activated_phase": final_phase,
                "fallback_applied": fallback_applied
            },
            "metrics": {
                "inference_latency_ms": round(latency_ms, 2),
                "hallucination_rate": round(self.total_hallucinations / self.total_decisions, 2)
            }
        }
        
        with open(self.decision_log_path, "a") as f:
            f.write(json.dumps(decision_event) + "\n")
            
        if self.verbose:
            print(f"\n--- Decision @ Step {step} ---")
            print(f"Extracted: {extracted_signal} | Applied: {final_phase} | Fallback: {fallback_applied}")
            print(f"Latency: {decision_event['metrics']['inference_latency_ms']}ms | Hallucination Rate: {decision_event['metrics']['hallucination_rate'] * 100}%\n")