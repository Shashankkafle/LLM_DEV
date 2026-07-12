# Metrics guide

What every field in `final_summary.json` means, which number to quote for which
comparison, and what we learned reconciling this SUMO pipeline against the
LLMTSCS/CityFlow paper (LLMLight). Every runner (LLM, baselines, CoLight eval,
replay) shares `utils/metrics_recorder.py`, so all runs emit the same summary.

Last updated: 2026-07-06.

---

## Provenance fields

| Field | Meaning |
|---|---|
| `sumo_version` | Simulator that produced the run. Runs from different SUMO versions are NOT bit-comparable (measured drift 1.12 vs 1.26 on FixedTime HZ-1: ~0.8% ATT). |
| `input_files` | Git-blob SHA-1 of the `.sumocfg` and every net/route file it references, CRLF-normalized. Equals `git hash-object <file>` and `git rev-parse HEAD:<path>`, so "did this run use committed inputs?" is a one-command check. |

A result without matching hashes across machines is not the same experiment.

## Travel time — one population question, one clock question

Every ATT variant answers two independent questions: WHO is averaged, and WHEN
does each vehicle's clock start/stop.

| Field | Population | Clock | Use for |
|---|---|---|---|
| `average_travel_time_s` | Completed trips only | insertion → network exit | Nothing comparative. Survivorship-biased: drops the still-stuck vehicles (16% on FixedTime HZ-1). |
| `cityflow_style_att_s` | Everyone who entered; in-flight clipped at horizon | insertion → network exit | Honest SUMO-internal comparisons. |
| `cityflow_clock_att_s` | Same as above | Approach-lane dwell only: excludes junction-internal time and the entire final route edge | **The number to quote against LLMTSCS/paper tables.** |
| `sumo_effective_att_s` | SUMO's own trip stats | in-network duration + depart delay | High-demand scenarios (HZ-2), where wait-to-insert is real travel time. |

Why the clock matters: LLMTSCS never times a trip end-to-end. Each intersection
records dwell on its own entering lanes; the reported ATT is the per-vehicle sum
of those segments. Junction crossings (~2.6 s each) and the final boundary edge
(772–786 m, ~70 s free-flow) are never counted. Measured effect: ~63 s on
FixedTime HZ-1; re-scoring CoLight HZ-1 with the CityFlow clock moved it from an
apparent +20% deficit (389 vs 322.85) to within 4% of the paper (310.03).

## Waiting time — two different definitions, ~3x apart

| Field | Definition | Use for |
|---|---|---|
| `average_waiting_time_s` | Cumulative seconds stopped (speed < 0.1) over the whole trip, completed vehicles | SUMO-internal only. ~3x larger than CityFlow AWT by construction. |
| `cityflow_style_awt_s` | Same, including in-flight vehicles | SUMO-internal only. |
| `average_per_decision_wait_s` | Mean duration of the CURRENT stop, over currently-halted vehicles only, sampled at decision points, averaged over samples | **The number to quote against LLMTSCS/paper AWT.** Mirrors CityFlow's `waiting_vehicle_list` (timer resets when the vehicle moves). |

Caveats: the per-decision metric samples network-wide at each intersection
decision (16 near-identical samples per round under FixedTime — harmless).
It is wired into every runner: the shared control loop (`runner_common`)
samples once per intersection decision for the LLM and baseline runners, and
`runner_colight` samples once per synchronized decision round.

## Queue length

`average_queue_length`: network-wide count of halted vehicles (speed < 0.1),
sampled every second, time-averaged. CityFlow's AQL counts entering lanes only,
sampled per 30 s action — close enough for comparison, with one trap: vehicles
SUMO refused to insert queue invisibly outside the network, deflating AQL
exactly when congestion is worst (see insertion censoring below).

## Vehicle accounting

`total_loaded / total_departed / loaded_but_never_departed / still_running_at_end /
completion_rate` come from step-sampled TraCI ID lists; the `sumo_*` block comes
from SUMO's own statistics output and is authoritative when they disagree (the
step-sampled sets can miss a few vehicles). `sumo_trip_count` reads the `count`
attribute of `vehicleTripStatistics`, which old SUMO versions (e.g. 1.12) do not
write — 0 there means "attribute absent", not "no trips".

---

## Reconciliation findings (SUMO pipeline vs LLMTSCS/CityFlow)

Anchors: pipeline A (LLMTSCS repo) reproduces the paper's Table 2 to 4+ decimals,
so paper numbers = that code's behavior.

1. **Metric definitions were the biggest "discrepancy".** AWT definitions differ
   ~3x by construction; ATT clocks differ by ~63–70 s/vehicle (exit edge +
   junctions). Both are now measured like-for-like by the fields above. The
   AWT direction-flip (SUMO ATT better while AWT 2.5x worse) that motivated the
   audit was pure definition mismatch.
2. **Dead vType (fixed 2026-07-06).** The converted `.rou.xml` files defined
   `<vType id="pkw">` but vehicles never referenced it — every run before the
   fix used SUMO's default vehicle type (tau=1.0, accel 2.6, random speedFactor),
   not the CityFlow-matched parameters. Applies to all results in
   `Run results.xlsx` and the committed CoLight training.
3. **cfphys route variants** (`*_cfphys.rou.xml` / `.sumocfg`, alongside the
   originals): `type="pkw"` attached, `tau="2.0"` (CityFlow's headwayTime),
   `sigma="0" speedFactor="1" speedDev="0"` (CityFlow's identical deterministic
   drivers), `departLane="best" departSpeed="max"`. Effect on FixedTime HZ-1:
   CityFlow-clock ATT 483 → 505, per-decision AWT 38 → 51, AQL 186 → 192.
   Because all stochastic elements are pinned, **cfphys runs are fully
   deterministic: 5 seeds produced bit-identical results.** No error bars
   needed; any difference between cfphys runs is code/config/version, never noise.
   2026-07-11: cfphys variants extended to all 22 `dataset/sumo_version`
   networks (`make_cfphys_routes.py`), and the CoLight defaults in
   `configurations.py` now point at them. Besides the dead-vType fix, this
   removes the insertion phantom: with `departSpeed=0` every vehicle counted as
   halting on its entry lane for exactly one step (a fake queue of 1, ~290 m
   upstream), leaking into the advanced pressure feature, the CoLight reward,
   MaxPressure, and the queue metrics. Prior sumo_version results (committed
   CoLight training, Run results.xlsx) are NOT comparable to cfphys runs.
4. **Insertion censoring on HZ-2 is structural.** `anon_4_4_hangzhou_real_5816`
   actually holds 6,984 vehicles. Even with cfphys insertion attributes, ~1,760
   (25%) never enter — with tau=2.0 the demand physically exceeds entry-edge
   capacity, while CityFlow packs vehicles in more aggressively. Any HZ-2 number
   must be quoted with `sumo_vehicles_not_inserted`. HZ-1 (49 refused of 2,983)
   is the clean reconciliation scenario.
5. **FixedTime is a different signal plan, not a bug.** LLMTSCS FixedTime
   effectively holds each phase 60 s (240 s cycle, transition embedded in the
   30 s slot: 5 y + 25 g); ours is 30 g + 3 y + 2 r appended (140 s cycle).
   Their paper text says 3 y + 2 r; their code does 5 y + 0 r — paper != code.
   The remaining FixedTime ATT gap (~505 vs 616) is attributed to this plus
   faster SUMO queue discharge; deliberately not "fixed".
6. **Version drift.** SUMO 1.12 vs 1.26 changed FixedTime HZ-1 ATT by ~0.8% with
   identical inputs. Standardize on one version (target: 1.26) before quoting
   cross-machine comparisons; `sumo_version` in every summary makes mixing
   detectable.

## Known, deliberately-unfixed encoding deltas (LLM prompt path)

The state segment boundaries were aligned with LLMTSCS and committed
(Seg1 = 0–L/10, Seg2 = L/10–L/3, Seg3 = rest). The remaining deltas vs the
LLMTSCS template are KNOWN and currently left as-is (decision 2026-07-06:
document, don't churn the prompt while evals are in flight):

- "Early queued" cutoff: ours 1.39 m/s (`STOP_SPEED_EARLY_QUEUE`) vs LLMTSCS 0.1.
- No `Relieves:` line in the per-signal blocks.
- Direction labels quoted on Segment lines (`('East')`) vs unquoted; inconsistent
  with our own Early-queued lines.
- Block order ETWT, ELWL, NTST, NLSL vs LLMTSCS ETWT, NTST, ELWL, NLSL.
- One missing blank line between observation and "Please answer:".

If LightGPT results come back anomalous, these are the first suspects — the
model is a fine-tune and sensitive to its training template. Note the paper's
Table 11 template ("Allowed lanes:", with a "Note:" block) is the GPT-4 path
(`models/chatgpt.py`), NOT what the LightGPT eval uses (`my_utils.getPrompt`,
"Relieves:", no Note) — match the code, not the PDF.

## Blockage runs (`--blockage_scenario`)

A run with a lane blockage is a **different experiment**, not a worse
controller. Provenance: `final_summary.json` gains `blockage_scenario` (the
scenario name), every `step_summaries.jsonl` line gains `blocked_lanes`, the
scenario JSON is copied into the run dir, and `compare_controllers.py` shows a
scenario column. Never rank blockage and blockage-free runs in one table
without that column.

**Obstacle accounting.** The `obstacle_vehicle` method inserts a real (frozen)
vehicle. It is filtered out of every per-vehicle MetricsRecorder metric (queue
counts, accumulated waiting, trip averages, CityFlow-style averages,
per-decision AWT, completion accounting), out of `get_state` counts, and out
of `cal_offline`'s tripinfo parse. It CANNOT be filtered from:

- lane-level halting counts: MaxPressure pressure, the CoLight queue reward,
  and the queue-output AQL each see +1 on the blocked lane for the whole
  window (defensible: a real sensor would also see a stopped vehicle);
- SUMO's own statistics: `sumo_trip_count` / `sumo_mean_trip_duration_s` /
  `sumo_effective_att_s` / `sumo_vehicles_inserted|finished` absorb one
  synthetic ~window-long trip per obstacle.

**Which numbers to quote under blockage.** `cityflow_style_att_s` +
`completion_rate` (in-flight vehicles are charged their time-so-far, so
trapped vehicles are not dropped). `sumo_effective_att_s` and
`average_travel_time_s` are completed-only and biased DOWN under blockage --
they drop exactly the trapped, worst-off vehicles. Once the blockage queue
spills back to the entry edge, new demand piles up invisibly in SUMO's
insertion queue: quote `sumo_vehicles_not_inserted` / `sumo_mean_depart_delay_s`
alongside anything else. Blockage runs are SUMO-internal comparisons only --
`cityflow_*` names denote the averaging convention, not paper comparability
(the paper has no blockage counterpart).

**Behavioral notes.** Teleporting is disabled on every measured run already;
blockage runs also disable it when there is no output dir (CoLight training),
because SUMO's default 300 s teleport deletes the frozen obstacle mid-window
(verified). Blockage runs never early-stop -- trapped or never-inserted
vehicles keep `getMinExpectedNumber() > 0` -- so they run the full horizon.
Trapped vehicles count as `early_queued` in prompts and features by design;
the prompt's blockage section states that those counts include unservable
vehicles, while MaxPressure/CoLight see only the raw numbers (chasing
unservable queues is part of what the experiment measures). CoLight:
`--blockage_scenario` applies to eval only (zero-shot arm, matching the LLM);
`--train_blockage_scenario` is a separate trained-on-incident arm -- the state
has no time index or blockage signal, so training on a fixed scheduled
incident is memorization, not adaptation. Never pool the two.

**Information-asymmetry control.** The LLM gets an explicit blockage section
while MaxPressure/CoLight see only halting counts, so an LLM win conflates
privileged information with reasoning. `runner.py --hide_blockage_info`
injects the blockage physically but keeps it out of the prompt: the core
matrix for any adaptation claim is LLM-with-info vs LLM-without-info vs
baselines, all on the same scenario and seed.

## Which number for which comparison

- Against LLMTSCS/paper tables: `cityflow_clock_att_s`,
  `average_per_decision_wait_s`, `average_queue_length` (approximate).
- Between SUMO runs: `cityflow_style_att_s` (+ `sumo_effective_att_s` and
  `sumo_vehicles_not_inserted` on high demand).
- Never in a comparison column: `average_travel_time_s`,
  `average_waiting_time_s` (populations/definitions differ).
- Before comparing anything across machines: `sumo_version` and `input_files`
  must match.

## Open items

- LLM (LightGPT) HZ-1 eval in flight — the D2 verdict; expect the CityFlow-clock
  ATT near or below the paper's 310.78 if the exit-edge effect dominates there too.
- Standardize all machines on SUMO 1.26.
- Prompt-encoding alignment (list above) — deferred decision.
- Seed sweeps only matter for non-cfphys (stochastic) configs.
