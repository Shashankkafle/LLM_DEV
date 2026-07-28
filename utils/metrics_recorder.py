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
    OBSTACLE_VEHICLE_PREFIX,
    RUN_RECORD_SCHEMA_VERSION,
)
import traci
from pathlib import Path
from datetime import datetime

from utils.run_manifest import input_fingerprints


def _json_fallback(obj):
    """Last-resort serializer so a numpy scalar/array in a decision record
    (e.g. CoLight features) never crashes a run."""
    if hasattr(obj, "item") and getattr(obj, "size", None) == 1:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class MetricsRecorder:
    def __init__(self, run_dir, verbose=True, phase_names=None, sumo_config=None,
                 blockage_manager=None, run_info=None):
        self.verbose = verbose
        self.input_files = (
            input_fingerprints(sumo_config) if sumo_config else None
        )

        # Run identity (controller, seed, ablation flags, ...) -- the runner's
        # run_meta dict. Merged into final_summary.json so comparison tables
        # can tell controllers and ablation arms apart, and stamped into every
        # decision record.
        self.run_info = run_info or {}
        self.run_started = time.time()
        self.run_started_at = datetime.now().isoformat(timespec="seconds")

        # When set, each step summary line records the currently blocked lanes,
        # giving blockage runs a time series to slice before/during/after
        # windows. Runs without a manager keep their exact previous format.
        self.blockage_manager = blockage_manager

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
        self.step_length_s = None

        # Decision-outcome counters. Every decision point is exactly one type;
        # see record_decision for the classification.
        self.total_decisions = 0            # all decision points, every type
        self.decisions_no_action_empty = 0  # intersection empty -> held phase, no LLM call
        self.decisions_llm_valid = 0        # LLM named a valid phase <signal>
        self.decisions_llm_no_action = 0    # LLM explicitly answered "no change" (e.g. None)
        self.decisions_no_answer = 0        # LLM queried but produced no <signal> tag
        self.total_hallucinations = 0       # LLM produced a <signal> tag naming an invalid phase
        self.decisions_inference_error = 0  # LLM call itself raised; phase held

        # Per-call inference cost, aggregated into the final summary.
        self.inference_latencies_ms = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

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

    def _track_loaded_ids(self):
        """Fold this step's newly built vehicles into the cumulative set.

        Synthetic blockage obstacles are filtered out here for the same reason
        they are filtered everywhere else in this class: they are
        infrastructure, not demand.
        """
        self.loaded_ids.update(
            v for v in traci.simulation.getLoadedIDList()
            if not v.startswith(OBSTACLE_VEHICLE_PREFIX)
        )

    def record_initial_load(self):
        """Capture the vehicles SUMO builds while loading the route file, i.e.
        before the first simulation step. Call once, after traci.start().

        getLoadedIDList() only reports vehicles built since the previous TraCI
        step, so this first batch is already gone by the time the loop's first
        record_step_summary() runs -- 10 vehicles (ids 0-9) on hangzhou_real.
        Missing them made completion_rate's denominator too small, which let it
        exceed 1.0 on runs where nearly everything arrived.
        """
        self._track_loaded_ids()

    def record_step_summary(self, step):
        current_time = traci.simulation.getTime()
        step_length = traci.simulation.getDeltaT()
        self.last_sim_time = current_time
        if self.sumo_version is None:
            self.sumo_version = traci.getVersion()[1]
            self.step_length_s = step_length

        # Track vehicles entering the simulation this step. Synthetic blockage
        # obstacles are filtered out of ALL per-vehicle accounting here: they
        # are infrastructure, not demand -- and vehicle.remove() at the end of
        # a blockage emits no arrival, so an unfiltered obstacle would sit in
        # still_running forever and pollute the CityFlow-style averages and
        # completion_rate of the blockage arm only.
        self._track_loaded_ids()
        for vehicle in traci.simulation.getDepartedIDList():
            if vehicle.startswith(OBSTACLE_VEHICLE_PREFIX):
                continue
            self.departed_ids.add(vehicle)
            self.depart_times[vehicle] = current_time
            self.final_edges[vehicle] = traci.vehicle.getRoute(vehicle)[-1]

        # Accumulate true waiting time and count the current network-wide queue.
        current_queue_length = 0
        for vehicle in traci.vehicle.getIDList():
            if vehicle.startswith(OBSTACLE_VEHICLE_PREFIX):
                continue
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
            if vehicle.startswith(OBSTACLE_VEHICLE_PREFIX):
                continue
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
        if self.blockage_manager is not None:
            summary["blocked_lanes"] = self.blockage_manager.get_blocked_lane_ids()
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
        summary = {"schema_version": RUN_RECORD_SCHEMA_VERSION}
        # Run identity first, so a summary is self-describing: controller,
        # seed, intersection_config, ablation flags, ... (whatever the runner
        # put in its run_meta). Metric keys below override on any collision.
        summary.update(self.run_info)
        summary.update(self._trip_averages())
        summary["sumo_version"] = self.sumo_version
        summary["step_length_s"] = self.step_length_s
        summary["run_started_at"] = self.run_started_at
        summary["run_wall_clock_s"] = round(time.time() - self.run_started, 1)
        summary["input_files"] = self.input_files
        # Blockage runs must be distinguishable in comparison tables: a run
        # with an incident is a different experiment, not a worse controller.
        if self.blockage_manager is not None:
            summary["blockage_scenario"] = self.blockage_manager.scenario_name
        summary["average_queue_length"] = (
            round(sum(self.queue_lengths) / len(self.queue_lengths), 2)
            if self.queue_lengths else None
        )

        # Episode-level vehicle accounting. A vehicle cannot depart without
        # having been loaded, so the union is a backstop for a launcher that
        # forgot record_initial_load(): without it, completion_rate can print
        # above 1.0.
        total_loaded = len(self.loaded_ids | self.departed_ids)
        total_departed = len(self.departed_ids)
        total_arrived = len(self.arrived_ids)
        summary["total_loaded_vehicles"] = total_loaded
        summary["total_departed_vehicles"] = total_departed
        summary["loaded_but_never_departed"] = max(total_loaded - total_departed, 0)
        summary["still_running_at_end"] = max(total_departed - total_arrived, 0)
        average_per_decision_wait_s = (
            round(sum(self.decision_wait_averages) / len(self.decision_wait_averages), 2)
            if len(self.decision_wait_averages) > 0 else 0
        )
        summary["average_per_decision_wait_s"] = average_per_decision_wait_s
        # The raw samples behind the mean, so distributions/time slices can be
        # analyzed later without rerunning (~one float per decision point).
        summary["decision_wait_samples"] = [
            round(w, 2) for w in self.decision_wait_averages
        ]
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
            + self.decisions_inference_error
        )
        valid_responses = self.decisions_llm_valid + self.decisions_llm_no_action
        summary["total_decisions"] = self.total_decisions
        summary["decisions_no_action_empty"] = self.decisions_no_action_empty
        summary["decisions_llm_queried"] = llm_queried
        summary["llm_phase_decisions"] = self.decisions_llm_valid
        summary["llm_no_action_decisions"] = self.decisions_llm_no_action
        summary["llm_no_answer"] = self.decisions_no_answer
        summary["total_hallucinations"] = self.total_hallucinations
        summary["decisions_inference_error"] = self.decisions_inference_error
        summary["inference_latency_ms_mean"] = (
            round(sum(self.inference_latencies_ms) / len(self.inference_latencies_ms), 2)
            if self.inference_latencies_ms else None
        )
        summary["inference_latency_ms_max"] = (
            round(max(self.inference_latencies_ms), 2)
            if self.inference_latencies_ms else None
        )
        summary["total_prompt_tokens"] = self.total_prompt_tokens or None
        summary["total_completion_tokens"] = self.total_completion_tokens or None
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
        # The frozen blockage obstacle accrues waiting time continuously for
        # its whole window; unfiltered it would pump this paper-comparable AWT
        # in the blockage arm only.
        halted_waits = [
            w for w in (
                traci.vehicle.getWaitingTime(v)
                for v in traci.vehicle.getIDList()
                if not v.startswith(OBSTACLE_VEHICLE_PREFIX)
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
                        latency_ms, extracted_signal, intersection_id,
                        blockage_facts=None, exit_blockage_facts=None,
                        blockage_info_in_prompt=None, token_usage=None,
                        error=None):
        """Record one decision point.

        decision_type is one of:
          - "no_action_empty"      intersection empty; phase held, no LLM call
          - "llm_decision"         LLM named a valid phase <signal>
          - "llm_no_action"        LLM explicitly answered "no change" (e.g. None)
          - "fallback_parse_error" LLM queried but gave no usable / valid answer
          - "inference_error"      the LLM call itself raised; phase held

        extracted_signal is the RAW parse result: None when no <signal> tag was
        produced, otherwise the tag's text (which may be an invalid phase name).
        blockage_facts / exit_blockage_facts are the structured describer
        outputs, recorded even when --hide_blockage_info keeps them out of the
        prompt (blockage_info_in_prompt says whether the prompt showed them).
        """
        self.total_decisions += 1

        if decision_type == "no_action_empty":
            self.decisions_no_action_empty += 1
            parsing_valid = None  # no LLM call was made, so the field is N/A
        elif decision_type == "inference_error":
            self.decisions_inference_error += 1
            parsing_valid = None  # the call failed; there was no output to parse
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

        if decision_type != "no_action_empty":
            self.inference_latencies_ms.append(latency_ms)
        if token_usage:
            self.total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            self.total_completion_tokens += token_usage.get("completion_tokens", 0)

        decision_event = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "phase_decision",
            "intersection_id": intersection_id,
            "controller": self.run_info.get("controller", "llm"),
            "traffic_state": state_dict.get("movement_states", {}),
            "blockage_facts": blockage_facts,
            "exit_blockage_facts": exit_blockage_facts,
            "blockage_info_in_prompt": blockage_info_in_prompt,
            "llm_input": {"user_prompt": prompt},
            "llm_output": {
                "raw_text": llm_output,
                "extracted_signal": extracted_signal,
                "parsing_valid": parsing_valid,
                "error": error,
            },
            "phase_action": {
                "decision_type": decision_type,
                "requested_phase": extracted_signal,
                "previous_phase": previous_phase,
                "phase_changed": previous_phase != final_phase,
                "activated_phase": final_phase,
                # Kept for continuity with earlier logs; now means "real failure"
                # only (empty-intersection no-ops are NOT counted as fallbacks).
                "fallback_applied": decision_type in ("fallback_parse_error",
                                                      "inference_error"),
            },
            "metrics": {
                "inference_latency_ms": round(latency_ms, 2),
                "prompt_tokens": (token_usage or {}).get("prompt_tokens"),
                "completion_tokens": (token_usage or {}).get("completion_tokens"),
            },
        }

        self._append_decision(intersection_id, decision_event)

        if self.verbose:
            print(f"\n--- Decision @ Step {step} ({decision_type}) ---")
            print(f"  Extracted: {extracted_signal} | Applied: {final_phase} | parsing_valid: {parsing_valid}")
            print(f"  Latency: {latency_ms:.1f}ms")

    def record_simple_decision(self, step, intersection_id, previous_phase,
                               activated_phase, decision_type,
                               controller_state=None, traffic_state=None):
        """Record one decision point for a non-LLM controller.

        Leaner sibling of record_decision: same file and phase_action shape,
        no llm_input/llm_output. controller_state holds whatever the controller
        computed to decide (MaxPressure pressures, FixedTime cycle index,
        CoLight action/Q-values), so any run is explainable after the fact.
        """
        self.total_decisions += 1
        decision_event = {
            "step": step,
            "timestamp": time.time(),
            "event_type": "phase_decision",
            "intersection_id": intersection_id,
            "controller": self.run_info.get("controller"),
            "traffic_state": traffic_state,
            "phase_action": {
                "decision_type": decision_type,
                "previous_phase": previous_phase,
                "activated_phase": activated_phase,
                "phase_changed": previous_phase != activated_phase,
            },
            "controller_state": controller_state or {},
        }
        self._append_decision(intersection_id, decision_event)

    def _append_decision(self, intersection_id, decision_event):
        decision_log_file = self.run_dir / f"{intersection_id}/{DECISIONS_FILENAME}"
        decision_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(decision_log_file, "a") as f:
            f.write(json.dumps(decision_event, default=_json_fallback) + "\n")
