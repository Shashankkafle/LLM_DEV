import hashlib
import json
import time
import xml.etree.ElementTree as ET
from configurations import (
    MIN_SPEED,
    PHASE_NAMES,
    STEP_SUMMARIES_FILENAME,
    FINAL_SUMMARY_FILENAME,
    DECISIONS_FILENAME,
    SUMO_STATISTIC_FILENAME,
)
import traci
from pathlib import Path
from datetime import datetime


def _input_fingerprints(sumo_config):
    """Git-blob SHA-1 of the sumocfg and the input files it references, keyed
    by filename. Matches `git hash-object <file>`, so a run's inputs can be
    checked against a commit directly. Returns None if anything can't be read
    (a run must never fail over provenance).
    """
    def blob_sha1(path):
        # CRLF->LF so a Windows checkout hashes the same as Linux and as
        # git's autocrlf-normalized blobs.
        data = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()

    try:
        config_path = Path(sumo_config)
        fingerprints = {config_path.name: blob_sha1(config_path)}
        root = ET.parse(config_path).getroot()
        for tag in ("net-file", "route-files", "additional-files"):
            node = root.find(f"./input/{tag}")
            if node is None:
                continue
            for name in node.get("value", "").split(","):
                ref = config_path.parent / name.strip()
                if ref.exists():
                    fingerprints[ref.name] = blob_sha1(ref)
        return fingerprints
    except Exception:
        return None


class MetricsRecorder:
    def __init__(self, run_dir, verbose=True, phase_names=None, sumo_config=None):
        self.verbose = verbose
        self.input_files = (
            _input_fingerprints(sumo_config) if sumo_config else None
        )

        # Valid phase names used to flag LLM hallucinations. Defaults to the
        # default config's phases; pass the active config's names when running
        # a non-default intersection config.
        self.valid_phase_names = phase_names if phase_names is not None else PHASE_NAMES

        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.step_log_path = self.run_dir / STEP_SUMMARIES_FILENAME

        # Per-vehicle trip tracking, keyed by SUMO vehicle id.
        self.depart_times = {}         # veh -> sim time it entered the network
        self.waiting_accumulated = {}  # veh -> total seconds spent stopped so far
        self.completed_trips = {}      # veh -> {"travel_time", "waiting_time"} once it arrives

        # CityFlow-clock ATT: LLMTSCS times only the dwell on intersection
        # approach lanes, so junction crossings and the final route edge (the
        # boundary exit, ~70 s free-flow here) are never counted. Accumulate
        # the same quantity for a like-for-like comparison with its tables.
        self.final_edges = {}    # veh -> last edge of its route
        self.approach_time = {}  # veh -> seconds spent on non-final, non-internal edges

        # Cumulative id sets, so we can report loaded vs departed vs arrived.
        self.loaded_ids = set()
        self.departed_ids = set()
        self.arrived_ids = set()

        self.decision_wait_averages = []  # list of average waiting times(sum (seconds since current stop began per vehicle)/number of vehicles) at each decision point, for later averaging, counts consiquitive wait times
        # One network-wide stopped-vehicle count per step (snapshot, time-averaged later).
        self.queue_lengths = []

        # Sim time at the most recent recorded step. Used to charge vehicles
        # still in the network their time-so-far when the horizon cuts them off.
        self.last_sim_time = 0.0

        # Captured on the first recorded step (TraCI is not connected yet at
        # construction time, and is already closed by final-summary time).
        self.sumo_version = None

        # Decision-outcome counters. Every decision point is exactly one type;
        # see record_decision for the classification.
        self.total_decisions = 0            # all decision points, every type
        self.decisions_no_action_empty = 0  # intersection empty -> held phase, no LLM call
        self.decisions_llm_valid = 0        # LLM named a valid phase <signal>
        self.decisions_llm_no_action = 0    # LLM explicitly answered "no change" (e.g. None)
        self.decisions_no_answer = 0        # LLM queried but produced no <signal> tag
        self.total_hallucinations = 0       # LLM produced a <signal> tag naming an invalid phase

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
        self.last_sim_time = current_time
        if self.sumo_version is None:
            self.sumo_version = traci.getVersion()[1]

        # Track vehicles entering the simulation this step.
        self.loaded_ids.update(traci.simulation.getLoadedIDList())
        for vehicle in traci.simulation.getDepartedIDList():
            self.departed_ids.add(vehicle)
            self.depart_times[vehicle] = current_time
            self.final_edges[vehicle] = traci.vehicle.getRoute(vehicle)[-1]

        # Accumulate true waiting time and count the current network-wide queue.
        current_queue_length = 0
        for vehicle in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(vehicle) < MIN_SPEED:
                current_queue_length += 1
                self.waiting_accumulated[vehicle] = (
                    self.waiting_accumulated.get(vehicle, 0.0) + step_length
                )
            road = traci.vehicle.getRoadID(vehicle)
            if not road.startswith(":") and road != self.final_edges.get(vehicle):
                self.approach_time[vehicle] = (
                    self.approach_time.get(vehicle, 0.0) + step_length
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

    def _parse_sumo_statistics(self):
        """Parse SUMO's --statistic-output (written on close) into
        population-faithful fields.

        Unlike the trip averages above -- which start the clock at insertion and
        count only completed trips -- these use SUMO's own per-vehicle
        accounting: mean duration PLUS the wait-to-insert (departDelay), time
        loss, and the full inserted/running/never-inserted/teleported breakdown.
        Returns {} if the file is absent (e.g. a launcher that didn't enable the
        statistics output, or train-mode), so older runs keep working.
        """
        stats_path = Path(self.run_dir) / SUMO_STATISTIC_FILENAME
        if not stats_path.exists():
            return {}
        try:
            root = ET.parse(stats_path).getroot()
        except ET.ParseError:
            return {}

        out = {}
        vehicles = root.find("vehicles")
        if vehicles is not None:
            inserted = int(vehicles.get("inserted", 0))
            running = int(vehicles.get("running", 0))
            out["sumo_vehicles_loaded"] = int(vehicles.get("loaded", 0))
            out["sumo_vehicles_inserted"] = inserted
            out["sumo_vehicles_running_at_end"] = running
            # Scheduled to depart but never inserted -- source-edge gridlock.
            out["sumo_vehicles_not_inserted"] = int(vehicles.get("waiting", 0))
            out["sumo_vehicles_finished"] = inserted - running

        teleports = root.find("teleports")
        if teleports is not None:
            out["sumo_teleports_total"] = int(teleports.get("total", 0))

        trip = root.find("vehicleTripStatistics")
        if trip is not None:
            duration = float(trip.get("duration", 0.0))
            depart_delay = float(trip.get("departDelay", 0.0))
            out["sumo_trip_count"] = int(trip.get("count", 0))
            out["sumo_mean_trip_duration_s"] = round(duration, 2)
            out["sumo_mean_depart_delay_s"] = round(depart_delay, 2)
            out["sumo_mean_time_loss_s"] = round(float(trip.get("timeLoss", 0.0)), 2)
            # CityFlow-comparable travel time: charge the wait-to-insert on top
            # of in-network duration. Still excludes never-inserted vehicles,
            # which are reported separately as sumo_vehicles_not_inserted.
            out["sumo_effective_att_s"] = round(duration + depart_delay, 2)

        return out

    def _cityflow_style_averages(self):
        """Travel/wait averages that also count vehicles still in the network at
        the horizon, charging each its time-so-far (last_sim_time - depart).

        This mirrors CityFlow, which averages over in-flight vehicles too, so a
        hard simulation horizon doesn't silently drop the vehicles that depart
        too late to finish (~16% here). Vehicles that never got inserted are
        still excluded -- they have no in-network time -- and are reported
        separately as loaded_but_never_departed / sumo_vehicles_not_inserted.
        """
        still_running = self.departed_ids - self.arrived_ids
        count = len(self.completed_trips) + len(still_running)
        if count == 0:
            return {
                "cityflow_style_vehicle_count": 0,
                "cityflow_style_att_s": None,
                "cityflow_style_awt_s": None,
            }
        total_travel = sum(t["travel_time"] for t in self.completed_trips.values())
        total_waiting = sum(t["waiting_time"] for t in self.completed_trips.values())
        for veh in still_running:
            depart = self.depart_times.get(veh)
            if depart is None:
                continue  # shouldn't happen: departed vehicles always have a time
            total_travel += self.last_sim_time - depart
            total_waiting += self.waiting_accumulated.get(veh, 0.0)
        return {
            "cityflow_style_vehicle_count": count,
            "cityflow_style_att_s": round(total_travel / count, 2),
            "cityflow_style_awt_s": round(total_waiting / count, 2),
        }

    def get_final_summary(self):
        """
        Returns overall episode-level metrics after the simulation ends.
        Call this once after the simulation loop exits.

        Trip averages are computed over vehicles that actually completed their
        trip. average_queue_length is the network-wide count of stopped vehicles
        averaged over all recorded steps.
        """
        summary = self._trip_averages()
        summary["sumo_version"] = self.sumo_version
        summary["input_files"] = self.input_files
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
        print(f"decision_wait_averages: {self.decision_wait_averages}")
        average_per_decision_wait_s = (
            round(sum(self.decision_wait_averages) / len(self.decision_wait_averages), 2)
            if len(self.decision_wait_averages) > 0 else 0
        )
        summary["average_per_decision_wait_s"] = average_per_decision_wait_s
        # Fraction of loaded vehicles that actually finished their trip. A low
        # rate means the completed-only ATT/AWT above are optimistic (they drop
        # the worst-off, still-stuck vehicles) -- report it alongside them.
        summary["completion_rate"] = (
            round(total_arrived / total_loaded, 4) if total_loaded else None
        )

        # Decision-outcome breakdown. "LLM-queried" excludes empty-intersection
        # no-ops, so the rates below describe only the steps where a decision was
        # actually asked of the model -- the meaningful denominator.
        llm_queried = (
            self.decisions_llm_valid + self.decisions_llm_no_action
            + self.decisions_no_answer + self.total_hallucinations
        )
        valid_responses = self.decisions_llm_valid + self.decisions_llm_no_action
        summary["total_decisions"] = self.total_decisions
        summary["decisions_no_action_empty"] = self.decisions_no_action_empty
        summary["decisions_llm_queried"] = llm_queried
        summary["llm_phase_decisions"] = self.decisions_llm_valid
        summary["llm_no_action_decisions"] = self.decisions_llm_no_action
        summary["llm_no_answer"] = self.decisions_no_answer
        summary["total_hallucinations"] = self.total_hallucinations
        summary["valid_response_rate"] = (
            round(valid_responses / llm_queried, 4) if llm_queried > 0 else None
        )
        summary["parse_error_rate"] = (
            round((self.decisions_no_answer + self.total_hallucinations) / llm_queried, 4)
            if llm_queried > 0 else None
        )
        summary["hallucination_rate"] = (
            round(self.total_hallucinations / llm_queried, 4) if llm_queried > 0 else None
        )

        # CityFlow-style averages that also count vehicles still in the network
        # at the horizon, so late departures are not silently dropped.
        summary.update(self._cityflow_style_averages())

        # Same population, but with LLMTSCS's clock (approach dwell only) --
        # the number directly comparable to the LLMTSCS/paper ATT tables.
        summary["cityflow_clock_att_s"] = (
            round(
                sum(self.approach_time.get(v, 0.0) for v in self.departed_ids)
                / len(self.departed_ids), 2
            )
            if self.departed_ids else None
        )

        # Population-faithful metrics from SUMO's own statistics (incl. the
        # never-inserted vehicles that the completed-trip averages above hide).
        summary.update(self._parse_sumo_statistics())
        return summary

    def save_final_summary(self):
        """Writes the episode summary JSON to disk and prints it."""
        summary = self.get_final_summary()
        out_path = self.run_dir / FINAL_SUMMARY_FILENAME
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("\n===== FINAL SIMULATION SUMMARY =====")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  Saved to: {out_path}")
        return summary

    def record_decision_wait(self):
        """Sample the network-wide mean waiting time at a decision point.

        Matches CityFlow's AWT: getWaitingTime (seconds since the current stop
        began, resets on movement), averaged over only the vehicles currently
        stopped (wait > 0), 0.0 when none are. Called per decision so
        save_final_summary can average across decision points.
        """
        halted_waits = [
            w for w in (
                traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()
            )
            if w > 0
        ]
        average_wait = (
            sum(halted_waits) / len(halted_waits) if halted_waits else 0.0
        )
        if self.verbose:
            print(f"average waiting time at decision point: {average_wait:.2f}s")
        self.decision_wait_averages.append(average_wait)

    def record_decision(self, step, state_dict, prompt, llm_output,
                        previous_phase, final_phase, decision_type,
                        latency_ms, extracted_signal, intersection_id):
        """Record one decision point.

        decision_type is one of:
          - "no_action_empty"      intersection empty; phase held, no LLM call
          - "llm_decision"         LLM named a valid phase <signal>
          - "llm_no_action"        LLM explicitly answered "no change" (e.g. None)
          - "fallback_parse_error" LLM queried but gave no usable / valid answer

        extracted_signal is the RAW parse result: None when no <signal> tag was
        produced, otherwise the tag's text (which may be an invalid phase name).
        """
        self.total_decisions += 1

        if decision_type == "no_action_empty":
            self.decisions_no_action_empty += 1
            parsing_valid = None  # no LLM call was made, so the field is N/A
        elif decision_type == "llm_decision":
            self.decisions_llm_valid += 1
            parsing_valid = True
        elif decision_type == "llm_no_action":
            self.decisions_llm_no_action += 1
            parsing_valid = True  # a parseable, legitimate "hold" answer
        else:  # "fallback_parse_error" -- distinguish "no answer" from "hallucination"
            if extracted_signal is None:
                self.decisions_no_answer += 1   # truncated output / no tag at all
            else:
                self.total_hallucinations += 1  # tag present but not a valid phase
            parsing_valid = False

        print(f"\n--- Decision @ Step {step} ---")
        self.record_decision_wait()

        decision_event = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "phase_decision",
            "traffic_state": state_dict.get("movement_states", {}),
            "llm_input": {"user_prompt": prompt},
            "llm_output": {
                "raw_text": llm_output,
                "extracted_signal": extracted_signal,
                "parsing_valid": parsing_valid,
            },
            "phase_action": {
                "decision_type": decision_type,
                "requested_phase": extracted_signal,
                "previous_phase": previous_phase,
                "phase_changed": previous_phase != final_phase,
                "activated_phase": final_phase,
                # Kept for continuity with earlier logs; now means "real failure"
                # only (empty-intersection no-ops are NOT counted as fallbacks).
                "fallback_applied": decision_type == "fallback_parse_error",
            },
            "metrics": {
                "inference_latency_ms": round(latency_ms, 2),
            },
        }

        decision_log_file = self.run_dir / f"{intersection_id}/{DECISIONS_FILENAME}"
        decision_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_log_file, "a") as f:
            f.write(json.dumps(decision_event) + "\n")

        if self.verbose:
            print(f"\n--- Decision @ Step {step} ({decision_type}) ---")
            print(f"  Extracted: {extracted_signal} | Applied: {final_phase} | parsing_valid: {parsing_valid}")
            print(f"  Latency: {latency_ms:.1f}ms")
