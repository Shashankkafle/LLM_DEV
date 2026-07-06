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
It is wired into the LLM and baseline runners; `runner_colight` does not call
`record_decision_wait()` yet, so CoLight summaries show 0 there.

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
- Wire `record_decision_wait()` into `runner_colight` for CoLight AWT.
- Prompt-encoding alignment (list above) — deferred decision.
- Seed sweeps only matter for non-cfphys (stochastic) configs.
