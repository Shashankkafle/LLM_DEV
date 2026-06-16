import json
import time
import traci
from pathlib import Path
from datetime import datetime


class MetricsRecorder:
    def __init__(self, run_name="default_run", base_log_dir="logs", verbose=True):
        self.verbose = verbose

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_log_dir) / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.decision_log_path = self.run_dir / "decisions.jsonl"
        self.step_log_path = self.run_dir / "step_summaries.jsonl"

        # Per-vehicle tracking: vid -> depart_time
        self.vehicle_entry_time = {}

        # Completed vehicle accumulators
        self.total_completed_vehicles = 0
        self.cumulative_travel_time = 0.0
        self.cumulative_accumulated_wait = 0.0  # sum of getAccumulatedWaitingTime at arrival

        # AQL tracking (averaged across steps)
        self.step_aql_sum = 0.0
        self.step_count = 0

        # Decision tracking
        self.total_hallucinations = 0
        self.total_decisions = 0

    def record_step_summary(self, step):
        current_time = traci.simulation.getTime()

        # 1. Register newly departed vehicles
        for vid in traci.simulation.getDepartedIDList():
            self.vehicle_entry_time[vid] = current_time

        # 2. Capture metrics for vehicles that just arrived (BEFORE they leave TraCI)
        for vid in traci.simulation.getArrivedIDList():
            if vid in self.vehicle_entry_time:
                travel_time = current_time - self.vehicle_entry_time.pop(vid)
                # getAccumulatedWaitingTime is still valid for this step (vehicle arrived, not yet purged)
                try:
                    acc_wait = traci.vehicle.getAccumulatedWaitingTime(vid)
                except traci.exceptions.TraCIException:
                    acc_wait = 0.0  # fallback if already purged
                self.cumulative_travel_time += travel_time
                self.cumulative_accumulated_wait += acc_wait
                self.total_completed_vehicles += 1

        # 3. AQL: halting vehicles across all edges (excluding internal junction edges)
        road_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        halting_vehicles = sum(
            traci.edge.getLastStepHaltingNumber(e) for e in road_edges
        )
        self.step_aql_sum += halting_vehicles
        self.step_count += 1

        # 4. Compute safe averages
        n = self.total_completed_vehicles
        att = (self.cumulative_travel_time / n) if n > 0 else 0.0
        awt = (self.cumulative_accumulated_wait / n) if n > 0 else 0.0
        aql = halting_vehicles  # current-step value; use step_aql_sum/step_count for episode average

        if self.verbose:
            print(
                f"[Step {step}] ATT={att:.1f}s | AWT={awt:.1f}s | "
                f"AQL={aql} | Completed={n} | Active={len(traci.vehicle.getIDList())}"
            )

        summary = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "step_summary",
            "aggregate_queue_length": aql,
            "total_vehicles_in_network": len(traci.vehicle.getIDList()),
            "completed_vehicles": n,
            "average_travel_time": round(att, 2),
            "average_waiting_time": round(awt, 2),
        }

        with open(self.step_log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def get_final_summary(self):
        """
        Returns overall episode-level metrics after the simulation ends.
        Call this once after the simulation loop exits.
        """
        n = self.total_completed_vehicles
        return {
            "total_completed_vehicles": n,
            "average_travel_time_s": round(self.cumulative_travel_time / n, 2) if n > 0 else None,
            "average_waiting_time_s": round(self.cumulative_accumulated_wait / n, 2) if n > 0 else None,
            "average_queue_length": round(self.step_aql_sum / self.step_count, 2) if self.step_count > 0 else None,
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
                        latency_ms, extracted_signal):
        self.total_decisions += 1
        if fallback_applied or extracted_signal not in ["NTST", "ETWT", "NLSL", "ELWL"]:
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

        with open(self.decision_log_path, "a") as f:
            f.write(json.dumps(decision_event) + "\n")

        if self.verbose:
            print(f"\n--- Decision @ Step {step} ---")
            print(f"  Extracted: {extracted_signal} | Applied: {final_phase} | Fallback: {fallback_applied}")
            print(f"  Latency: {latency_ms:.1f}ms | Hallucination Rate: {decision_event['metrics']['hallucination_rate']*100:.1f}%")