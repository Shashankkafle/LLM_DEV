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

`peak_vehicles_in_network` / `peak_vehicles_in_network_time_s` (added 2026-09-24):
the highest step-sampled count of non-obstacle vehicles in the network and the
sim time it was first reached. The per-second series behind it is
`network_timeseries.csv` in the run dir (`t, vehicles_in_network, departed_cum,
arrived_cum, halting_total, not_inserted_pending`); `halting_total` is the same
speed < 0.1 count that feeds `average_queue_length`. See SURGE_SCENARIOS.md.

`edge_timeseries.csv` (added 2026-09-29): one wide row per step, `t`, then
`vehicles:<edge>` and `halting:<edge>` for every non-internal edge of the
network plus an `internal` bucket for junction lanes. Built inside the same
per-vehicle loop as `halting_total`, so per row `sum(halting:*) ==
halting_total` and `sum(vehicles:*) == vehicles_in_network` exactly, the
obstacle excluded. `MetricsRecorder(edge_series=False)` turns it off; every
other output is byte-identical either way. `surge_timeseries.csv` (`t, surge,
departed_cum, arrived_cum, pending`) and the summary's `surge_vehicles`
appear only when the route file carries injected `surge_<name>_<k>` vehicles
(SURGE_SCENARIOS.md, schema v2); `tools/analyze_surge.py` reads all three.

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

**Who hears about a blockage.** Each active blockage is described to TWO
controllers: the intersection the blocked lane feeds (downstream of the
blockage) via the `LANE BLOCKAGE REPORT` section, and the intersection whose
exit road it sits on (upstream of the blockage) via the `DOWNSTREAM BLOCKAGE
REPORT` section (template constants in `utils/prompt_builder.py`, goldens in
`tests/test_prompt_builder.py`). `--blockage_info_scope
{both,approach,exit}` (default `both`) restricts which side is informed
('approach' = only the controller whose approach lane is blocked, the
pre-existing behavior) and is recorded in run meta -- runs logged before this
key existed informed the approach side only. Blockages on fringe-origin edges
have no upstream traffic
light and appear on the downstream side only. An empty intersection makes no
LLM call, so it sees neither section (there is no decision the information
could alter).

**Event-text templates v2.1 (2026-07-13).** Both sections implement the v2.1
freeze-candidate templates: per-event text is factual only, no control advice.
Approach side: one bullet per blocked signal-served lane, full ("the lane is
fully blocked ... cannot reach the intersection") vs partial ("discharges at a
reduced rate") wording chosen by whether the lane is impassable, each bullet
naming the corrupted approach AND the clean paired approach of the same
signal. Blockages on lanes no listed signal serves (right-turn lanes) are
suppressed -- the controller has no action they could inform. Exit side: one
bullet per blocked exit road ("{K} of {M} lanes"), distance AND link length so
remaining storage is inferable, movement-level attribution ("North→South
through (part of signal NTST)"); roads only right-turning vehicles enter are
suppressed. Scenario JSONs may set a per-blockage `cause` shown verbatim
("collision", "roadworks", ...); omitted causes default per method
(`configurations.BLOCKAGE_DEFAULT_CAUSE`). The generic guidance (trapped
queues cannot discharge; blocked roads spill back) moved out of the per-event
text into the system prompt (`configurations.LLM_COMMONSENSE_BLOCK`), shared
by BOTH LLM arms in EVERY run -- so `--hide_blockage_info` isolates the event
information itself, not the concept of blockages. Blockage-section wording
from before this date (`LANE BLOCKAGE CONTEXT`, "Signal capacity impact"
lines) is superseded: prompts logged under it are not byte-comparable with
newer ones. The no-blockage USER prompt is byte-unchanged (the sha pin in
tests/test_prompt_builder.py is untouched), but the SYSTEM prompt now carries
the commonsense block in all runs, so LLM runs logged before this date differ
there.

**Information-asymmetry control.** The LLM gets explicit blockage sections
while MaxPressure/CoLight see only halting counts, so an LLM win conflates
privileged information with reasoning. `runner.py --hide_blockage_info`
injects the blockage physically but keeps BOTH sections out of the prompt: the
core matrix for any adaptation claim is LLM-with-info vs LLM-without-info vs
baselines, all on the same scenario and seed.

## Oversaturation indices (`oversaturation_indices.csv`)

Written by every FixedTime/MaxPressure/LLM run (`utils/oversaturation_indices.py`,
hooked into `runner_common`'s loops; CoLight and replay loops are not hooked).
The module only reads TraCI subscriptions, and a test pins that phase sequences
and tripinfo are byte-identical with it live or stubbed.
`--oversat_shadow` adds `oversaturation_shadow.csv` (per step, per lane:
halted and exit counts).

- **Green interval** (per lane): the maximal run of seconds in which any of
  the lane's links is `G`/`g`. **All-red onset**: the first second after it in
  which none of the lane's links is `G`/`g`/`y`.
- **TOSI** (per lane-green) = |H ∩ R| × 2.0 s / green_s, clamped to [0, 1].
  H = vehicles halted (< 0.1 m/s) on the lane when green came on, and R =
  vehicles on the lane at all-red onset. `discharged` = |H \ R|. When the lane
  had an active blockage at any point from green to all-red, TOSI is left
  empty (`blocked_flag` = 1).
- **SOSI** (per movement = lane → target lane) is the share of green seconds
  after the first 3 in which all three rules hold:
  - (a) a vehicle on the lane is halted within 15 m of the stop line;
  - (b) no vehicle left the lane (set difference, so a flat count can still
    hide an exit);
  - (c) a vehicle on the target lane is halted within 15 m of its entry.

  SOSI is empty when green_s ≤ 3.
- **ratio_start / ratio_end** = halted count / floor(lane length / 7.5), at
  green start and at all-red onset.

**Timing.** The hook runs after `simulationStep`, so the signal state it reads
governed the previous second. Boundaries are labelled with that second, and
H, R and the ratios come from the vehicle snapshot taken at that instant. If
H were sampled one second later, the head of the queue would already be
moving.

**Obstacle and right turns.** Unlike SUMO's lane halting numbers, every count
here excludes the obstacle vehicle, filtered by id prefix. Right-turn lanes
are `g` in every state, including all-red, so they never complete a green and
get no rows. Behind a c3-style obstacle 10 m from the stop line, the queue
head stands about 17 m back, so rule (a) cannot fire on the blocked lane
itself. Its SOSI stays 0, and what the blockage does to that lane shows up in
TOSI (suppressed) and in the ratios.

`tools/inspect_indices.py` replays the computation in sumo-gui, with the
arithmetic printed per green, for checking the numbers by eye.

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
