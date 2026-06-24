import json
import time
from configurations import MIN_SPEED
import traci
from pathlib import Path
from datetime import datetime


class MetricsRecorder:
    def __init__(self, run_dir, verbose=True):
        self.verbose = verbose

        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = self.run_dir / "step_summaries.jsonl"

        # Per-vehicle trip tracking, keyed by SUMO vehicle id.
        self.depart_times = {}         # veh -> sim time it entered the network
        self.waiting_accumulated = {}  # veh -> total seconds spent stopped so far
        self.completed_trips = {}      # veh -> {"travel_time", "waiting_time"} once it arrives

        # Cumulative id sets, so we can report loaded vs departed vs arrived.
        self.loaded_ids = set()
        self.departed_ids = set()
        self.arrived_ids = set()

        # One network-wide stopped-vehicle count per step (snapshot, time-averaged later).
        self.queue_lengths = []

        # LLM decision counters.
        self.total_decisions = 0
        self.total_hallucinations = 0

    def _trip_averages(self):
        """Averages over vehicles that have actually completed their trip so far."""
        count = len(self.completed_trips)
        if count == 0:
            return {
                "total_completed_vehicles": 0,
                "average_travel_time_s": None,
                "average_waiting_time_s": None,
            }
        total_travel = sum(t["travel_time"] for t in self.completed_trips.values())
        total_waiting = sum(t["waiting_time"] for t in self.completed_trips.values())
        return {
            "total_completed_vehicles": count,
            "average_travel_time_s": round(total_travel / count, 2),
            "average_waiting_time_s": round(total_waiting / count, 2),
        }

    def record_step_summary(self, step):
        current_time = traci.simulation.getTime()
        step_length = traci.simulation.getDeltaT()

        # Track vehicles entering the simulation this step.
        self.loaded_ids.update(traci.simulation.getLoadedIDList())
        for vehicle in traci.simulation.getDepartedIDList():
            self.departed_ids.add(vehicle)
            self.depart_times[vehicle] = current_time

        # Accumulate true waiting time and count the current network-wide queue.
        current_queue_length = 0
        for vehicle in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(vehicle) < MIN_SPEED:
                current_queue_length += 1
                self.waiting_accumulated[vehicle] = (
                    self.waiting_accumulated.get(vehicle, 0.0) + step_length
                )
        self.queue_lengths.append(current_queue_length)

        # Finalize trips for vehicles that arrived (left the network) this step.
        for vehicle in traci.simulation.getArrivedIDList():
            self.arrived_ids.add(vehicle)
            depart_time = self.depart_times.get(vehicle)
            if depart_time is None:
                continue  # never saw it depart; skip rather than fabricate a time
            self.completed_trips[vehicle] = {
                "travel_time": current_time - depart_time,
                "waiting_time": self.waiting_accumulated.get(vehicle, 0.0),
            }

        summary = self._trip_averages()
        summary["queue_length"] = current_queue_length
        summary["step"] = step
        with open(self.step_log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def get_final_summary(self):
        """
        Returns overall episode-level metrics after the simulation ends.
        Call this once after the simulation loop exits.

        Trip averages are computed over vehicles that actually completed their
        trip. average_queue_length is the network-wide count of stopped vehicles
        averaged over all recorded steps.
        """
        summary = self._trip_averages()
        summary["average_queue_length"] = (
            round(sum(self.queue_lengths) / len(self.queue_lengths), 2)
            if self.queue_lengths else None
        )

        # Episode-level vehicle accounting.
        total_loaded = len(self.loaded_ids)
        total_departed = len(self.departed_ids)
        total_arrived = len(self.arrived_ids)
        summary["total_loaded_vehicles"] = total_loaded
        summary["total_departed_vehicles"] = total_departed
        summary["loaded_but_never_departed"] = max(total_loaded - total_departed, 0)
        summary["still_running_at_end"] = max(total_departed - total_arrived, 0)

        summary["total_decisions"] = self.total_decisions
        summary["total_hallucinations"] = self.total_hallucinations
        summary["hallucination_rate"] = (
            round(self.total_hallucinations / self.total_decisions, 4)
            if self.total_decisions > 0 else None
        )
        return summary

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
