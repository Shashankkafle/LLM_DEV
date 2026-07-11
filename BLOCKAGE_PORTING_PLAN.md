# Blockage feature porting plan

Port lane-blockage injection + blockage-aware LLM prompts from
`C:\Users\m6722\Research\LLMTSCS-custom_prompts` (branch `sumo-backend`) into this repo.

Guiding rule: reuse the source's proven TraCI mechanics, JSON scenario format, and prompt
phrasing — but rebuild all the thin wiring natively in this repo's style. The source's
wiring is welded to a Gym-style env, episode loops, wandb, and CSV logging that this repo
deliberately does not have. Rebuilding ~40 lines of wiring is cheaper than adapting ~700
lines of theirs.

Revision 2 (2026-07-11): updated after `advanced_colight_staging` was merged in (b6a9804)
and after live SUMO experiments verified the physics risks (probe results in §3). Items
marked **[verified]** were confirmed by running SUMO 1.26 headless on
`simulations/single_intersection`.

---

## 0. Base branch — RESOLVED

`advanced_colight_staging` is merged (b6a9804). Consequences for this plan:

- The shared step loop now lives in `runner_common.py` — the blockage tick and wiring
  land there once, not per-runner.
- `SumoEnv` already has `--seed`; seeded A/B pairs are possible (and **[verified]**
  meaningful: a mid-run obstacle add/freeze/remove leaves every other vehicle's
  trajectory bit-identical under the same seed — with/without-blockage pairs differ only
  by the physical blockage).
- `sumo_metrics_args()` already passes `--time-to-teleport -1` on every measured run —
  this changes §2.3/§3.1 vs. revision 1. **Do not blindly re-add the flag: SUMO 1.26
  hard-errors on a duplicated option [verified], which would crash every blockage run.**
- METRICS.md exists; blockage metric caveats (§3.3) go there.

---

## 1. What the source feature actually is

A scenario JSON declares scheduled blockages. A `BlockageManager` executes them against
the live TraCI session, ticked once per simulation step. Two methods:

1. **`obstacle_vehicle`** — insert a vehicle (`traci.route.add` + `traci.vehicle.add`),
   then position and freeze it
   (`moveTo` + `setSpeed(0)` + `setSpeedMode(0)` + `setLaneChangeMode(0)`, retried on
   `TraCIException`). Removed with `vehicle.remove` at `end_step`. Physically blocks one
   lane at a given distance from the stop line. **[verified]** the freeze holds perfectly
   (zero drift over 900 steps) once teleporting is disabled, and removal is clean.
2. **`speed_restriction`** — `traci.lane.setMaxSpeed(original * (1 - severity))`,
   original restored at `end_step`. Whole-lane slowdown; `position` is ignored, and at
   severity 1.0 it also chokes vehicle *insertion* on that lane **[verified]**.

The prompt side renders active blockages as a bullet list ("LANE BLOCKAGE CONTEXT")
appended to the traffic-state prompt: lane, position, and either
"stopped vehicle — full blockage" or "speed restriction — N% reduction".

### Trust warnings (verified in the source — do not assume "it worked there")

- The feature was **never exercised against traffic**: every `<flow>` in the source's
  route file is commented out, and the LLM runner hardcodes `model = None`, so blockage
  prompts were never sent to any model.
- Two of the three shipped scenario JSONs **crash the loader** (`KeyError:
  'intersection_id'`).
- Activation uses exact equality `current_step == start_step`, so a missed tick silently
  never fires.
- Leftover debug `print`s fire every step while an obstacle is pending.

The TraCI mechanics themselves are correct SUMO idioms — now **[verified]** live against
this repo's toy net — and are worth keeping; everything around them gets rebuilt.

---

## 2. Target design

One new module, small edits at existing seams. No new abstractions.

### 2.1 New: `utils/blockage_manager.py` (~150 LOC) — copy-with-edits

Copy the source module, then trim to this repo's style:

- Blockages are **plain dicts** (this repo has zero dataclasses/typing): the JSON entries
  themselves. Keys: `blockage_id`, `lane_id`, `position` (metres upstream of the stop
  line), `start_step`, `end_step` (null = forever), `method`, `severity` (default 1.0).
- Fold `load_scenario(path)` into this module. Because dicts lose the dataclass's
  load-time TypeError on bad fields, the loader must be **strict** to compensate:
  required keys present, **unknown keys rejected** (catches typos like `startstep`),
  `severity` in [0,1], `end_step` null or > `start_step`, duplicate `blockage_id`
  rejected, and no two `speed_restriction`s on the same lane with overlapping windows
  (the source silently corrupts lane speeds in that case — `_original_speeds` is keyed
  by lane and gets clobbered).
- **`import traci` inside the TraCI-touching methods only**, matching the deliberate
  convention in `utils/state_features.py` — otherwise the pure core and loader become
  untestable/uncollectable on machines without SUMO, breaking §2.8.
- **Keep verbatim:** the two-step obstacle insertion with deferred positioning + retry,
  *including its ASSUMPTION comments* — they explain why SUMO's insertion model forces
  the dance, and without them the retry will get "simplified" away someday. (On SUMO
  1.26 `moveTo` force-inserts instantly even into a jam **[verified]**, so the retry is
  currently dead code — keep it anyway as cheap version tolerance.) Also keep the
  speed-restriction save/restore and the runtime `ValueError` on unknown method as a
  backstop behind the load-time check.
- **New before `moveTo`: an occupancy check** on the target position
  (`lane.getLastStepVehicleIDs` + positions). `moveTo` onto an occupied spot succeeds and
  overlaps vehicles **[verified]**: under `collision.action=warn` it spams collisions
  every step for the obstacle's whole life; under `teleport` (SUMO default) the obstacle
  is deleted one step after placement and the blockage never happens. If occupied, keep
  the blockage pending and retry next tick.
- **Fix activation:** `current_step >= start_step` guarded by a `finished` set; a
  blockage whose `end_step` is *also* already past goes **straight to finished without
  activating** (otherwise it add-then-removes a vehicle in one tick and can leak an
  unmanaged phantom vehicle).
- **Keep a ~5-line `reset()`** clearing `_active`/`_pending`/`_original_speeds`/
  `finished` (no traci calls). Revision 1 deleted it; CoLight's per-round fresh-env loop
  is exactly the reuse case where a stale manager would silently never fire again —
  cheap insurance against a natural future refactor.
- **Delete:** `validate_blockages()` (half-implemented dead code — the loader plus a
  small post-start check replace it: lane exists, position within lane length, hoisted
  here so it fails before the run, not inside the retry loop); the `intersection_id`
  field (see §2.3 for the mapping-holes caveat); logging/typing ceremony and the Gym
  usage docstring.
- **Keep the read API:** `step(current_step)`, `get_active_blockages()`,
  `get_blocked_lane_ids()`.

Clock convention: `step(t)` is always called with **simulation seconds**
(`traci.simulation.getTime()` / `env.get_current_step()`), in every loop that ticks it.

### 2.2 Scenario JSONs — copy-paste

`accident_single_lane.json` and `construction_zone.json` copy verbatim: their
`W2TLS_0`/`W2TLS_1` lane IDs exist in `simulations/single_intersection/net.xml` (189.6 m,
so positions 80/100 m are in range), and that net has real demand on the blocked lane.
Put them in `simulations/single_intersection/scenarios/`. Skip `baseline.json` (empty
list == not passing the flag). Scenarios for hangzhou/jinan nets are new data authoring,
later — and when authoring them, check the lane-connection map first: "full blockage" is
a **movement-level** claim, not approach-level. **[verified]** on the toy net one
obstacle froze the W→E movement completely (zero route-around, because the adjacent lane
serves a different movement); on approaches where an adjacent lane serves the *same*
movement, traffic will likely bypass.

### 2.3 `sumo_env.py` — rewrite the hooks (~20 LOC total)

- `__init__` gains `blockage_manager=None`.
  **Teleport flag (corrected from rev 1):** `sumo_metrics_args()` already adds
  `--time-to-teleport -1` to every run with an `output_dir` — appending it again crashes
  SUMO 1.26 with "option was already set" **[verified]**. Append it **only when
  `blockage_manager` is set and the cmd doesn't already contain it** — in practice that
  covers exactly the one uncovered path, the CoLight *training* env (built without
  `output_dir`), where a 600 s blockage would otherwise dissolve at +300 s and the policy
  would train on different physics than its eval.
- `step()`: `traci.simulationStep()`, then
  `if self.blockage_manager: self.blockage_manager.step(self.get_current_step())`.
  Single choke point for runner/baselines/CoLight via `runner_common.py`
  (replay_runner is separate, §2.7).
- `get_state()`: skip vehicles whose ID starts with the obstacle prefix (§3.3), and add a
  `blocked` flag to each lane's dict. Note precisely what that buys: the flag flows into
  `decisions.jsonl` (via `record_decision`'s `movement_states` copy) — it does **not**
  reach CoLight features, which extract only count keys. That is acceptable and should be
  stated in the experiment notes (CoLight is deliberately blockage-blind, see §3.6).
- New helper `describe_blockages(intersection_id)` → list of dicts
  `{approach, movement, segment, method, severity}`. The env owns topology; the prompt
  builder stays topology-free. **Define behavior for unmappable lanes explicitly**: both
  topology maps have holes today — right-turn lanes have an approach but no movement
  (never in any phase), and upstream edge segments on multi-segment approaches
  (hangzhou/jinan) are in neither map. Policy: the post-start validation **rejects** a
  scenario whose lane doesn't resolve to (intersection, approach); if a raw-lane fallback
  is ever wanted instead, render it as a warning bullet — never silently omit a blockage
  from the prompt while it physically jams traffic.

### 2.4 `utils/prompt_builder.py` — rewrite, reusing source phrasing (~20 LOC)

Do **not** port `prompts/sumo_blockage_prompt.py` (345 LOC). Its `<phase>0–3</phase>`
answer contract, hardcoded lane tables, duplicate no-blockage builders, system-prompt
loading, and two-message format all duplicate or contradict what this repo already has.
About 10 lines of phrasing survive.

- New `build_blockage_section(blockage_descriptions)`: the source's all-caps
  "LANE BLOCKAGE CONTEXT" header (good salience engineering — keep verbatim) + one bullet
  per blockage in the prompt's own vocabulary (approach + movement + segment, not raw
  lane IDs), with the source's method wording: "stopped vehicle — full blockage" /
  "speed restriction — N% reduction". Returns `''` for an empty list.
- Include the trapped-vehicle sentence **inside the section, worded as an explicit
  override** of the intro's definition: "on blocked lanes, the early-queued count above
  INCLUDES vehicles trapped behind the blockage that cannot reach the intersection."
  Never touch the shared intro paragraph.
- Insert between the observation text and "Please answer:"; signature gains
  `blockages=None`.
- **Invariant: with no active blockages the model input is byte-identical to today's.**
  Test it properly: assert on the full (system + user) input, built from a state dict
  the env actually emits (i.e. containing `blocked: false` flags), not a hand-built one;
  and pin an **active-blockage golden string too**, so the shared text can't silently
  fork in the active branch.
- Keep the `<signal>NAME</signal>` answer contract unchanged.

### 2.5 Runner wiring — rewrite (~10 LOC in runner_common + flags)

- `--blockage_scenario` flag (path to JSON, default `None`) on the LLM, baselines, and
  CoLight runners; manager constructed in `main()` and passed to `SumoEnv(...)`;
  post-start validation right after `start_simulation()`.
- **Drop rev 1's `state_is_empty` veto — it was wrong.** Trapped vehicles are real
  vehicles on controlled lanes and keep `early_queued` non-zero, so a blocked approach
  never reads as empty; only a genuinely vehicle-free intersection skips the LLM, and
  holding is correct then. Worse, the veto would fire LLM calls on all-zero observations
  only in the blockage arm, changing the decision-type mix and making
  hallucination/parse-rate comparisons across arms invalid.
- Decision call site passes `env.describe_blockages(iid)` into the prompt function.
- `runner_colight.py`: construct the manager (or call `reset()`) **inside the round
  loop**, next to the per-round `SumoEnv`. Experiment design: make **eval-only blockage
  the primary arm** (zero-shot robustness — matches the LLM's zero-shot setting; CoLight
  state has no time index or blockage signal, so training on a fixed scheduled incident
  is memorization, not adaptation). A trained-on-incident arm, if run, is a separately
  labeled arm, never pooled.

### 2.6 Provenance & metrics (~20 LOC)

- `run_details` / replay meta / `final_summary.json`: record `blockage_scenario` path and
  name; copy the scenario JSON into the run dir; `compare_controllers.py` gains a
  scenario column (implement **before** the first blockage table — the repo's
  `cityflow_*` metric names actively invite quoting blockage runs against paper tables,
  which would be invalid).
- `MetricsRecorder`: accept the manager (or `None`); each `record_step_summary` line
  gains `blocked_lanes`.
- **Obstacle filtering — the exact loops (this is where rev 1 was too vague):** filter
  IDs with the obstacle prefix in (a) the departed/arrived **id-set updates** —
  `traci.vehicle.remove()` emits no arrival **[verified]**, so an unfiltered obstacle
  sits in `still_running` forever and pollutes `cityflow_style_att_s/awt_s` and
  `completion_rate` in the blockage arm only; (b) the `record_step_summary` per-vehicle
  loop; (c) **`record_decision_wait`** — the frozen obstacle accrues `getWaitingTime`
  continuously (189 s after 190 s frozen **[verified]**), which would pump
  `average_per_decision_wait_s`, the paper-comparable AWT, by enough to flip an FT/MP
  row. Optionally add the same prefix filter to `cal_offline`'s tripinfo parse (the one
  place post-hoc filtering is possible).
- What **cannot** be filtered, accept and document in METRICS.md: lane-level halting
  counts (MaxPressure pressure, CoLight reward, offline AQL) see the obstacle as +1 —
  sensor-realistic; and SUMO's own `vehicleTripStatistics` counts the removed obstacle
  as one finished ~600 s trip **[verified]**.

### 2.7 `replay_runner.py` — small rewrite (~10 LOC)

It does not use `SumoEnv` (raw `traci.start` + `simulationStep` loop), so the env tick
never fires there. Load the scenario recorded in replay meta, build its own manager, tick
it after `simulationStep()` with `traci.simulation.getTime()`. Without this, reruns of
blockage runs silently replay blockage-free and diverge (the obstacle also suppresses the
`getMinExpectedNumber` early-stop in the original but not the rerun).

### 2.8 Tests — new (source test asserts nothing)

- Pure-core unit test per the `tests/test_state_features.py` pattern (works because of
  the deferred-traci-import rule in §2.1): schedule + current step + active/finished sets
  → activate/deactivate lists. Cases: start, end, `end_step=None`, mid-window start,
  **both-past-at-start (must go straight to finished)**, overlap, duplicate-id rejection,
  same-lane speed-restriction overlap rejection, unknown-key rejection.
- Prompt tests: byte-identical no-blockage model input through env-shaped state;
  active-blockage golden string.
- Headless smoke run on `simulations/single_intersection` + `accident_single_lane.json`,
  asserting: the obstacle ID stays in `getIDList()` for the **whole** window (the direct
  probe for teleport regressions — teleport *counters* can miss it), the queue forms on
  `W2TLS_0`, and blocked-movement arrivals stay frozen during the window.
- One metric-isolation check: a run with an obstacle and zero traffic reports AWT 0 and
  zero completed trips.

---

## 3. Verified risk register

Live-probe results (SUMO 1.26.0, headless, `simulations/single_intersection`, probe
script preserved in the session scratchpad `blockage_probe/probe.py`):

### 3.1 Teleporting — mostly already solved, one gap **[verified]**
Under SUMO's default 300 s time-to-teleport, the frozen obstacle deletes **itself** at
exactly +300 s ("teleports beyond arrival edge" — its one-edge route ends there), while
`BlockageManager` still reports it active; queued real vehicles mostly don't teleport
(the obstacle's clock started first), so the failure is deceptive — the symptom is the
blockage vanishing, not mass teleports. `speed_restriction` at severity 1.0 is worse:
frozen vehicles teleport in a wave at +300 s and **reappear downstream of the junction at
full speed, skipping the intersection entirely**. On the merged branch,
`sumo_metrics_args()` already passes `--time-to-teleport -1` on every measured run, so
both arms of any A/B are already symmetric; the only gap is the CoLight training env
(§2.3), and re-adding the flag naively crashes SUMO (duplicate option).

### 3.2 Collisions from `moveTo` **[verified]**
`moveTo` force-inserts instantly, even into a jam — placement onto occupied space is the
real hazard, in both directions: `collision.action=warn` (toy cfg) keeps an overlapping
pair reported as a collision every step for the obstacle's life; `collision.action=
teleport` (SUMO default, dataset configs) deletes the obstacle one step after placement.
Mitigations: occupancy check before `moveTo` (§2.1), `collision.action=warn` required on
blockage runs, never schedule a second blockage inside an already-queued stretch.

### 3.3 Which metrics to trust under blockage **[verified]**
Back-spill starves vehicle *insertion* for the whole approach: pending vehicles
accumulate invisibly (207 pending by step 1000 in the probe) — absent from every lane
metric, the prompt, and completed-trips ATT. Consequently: quote
**`cityflow_style_att_s` + `completion_rate`** for blockage A/Bs (the CityFlow-style
average charges in-flight vehicles time-so-far at the horizon, so it has no survivorship
bias); `sumo_effective_att_s` and `average_travel_time_s` are **completed-only and biased
DOWN** under blockage (they drop exactly the worst-off vehicles) — footnote them in
METRICS.md and in any table where the scenario column is non-empty. `sumo_vehicles_
not_inserted` and departDelay are where the starved demand actually shows up.

### 3.4 Runs don't early-stop
While vehicles are trapped or never inserted, `getMinExpectedNumber()` never reaches 0,
so the early-exit in `runner_common.py` / `replay_runner.py` / `runner_colight.py` never
fires — blockage runs execute the full horizon. Expected, not a bug; size `end_step` to
leave drain time before `simulation_steps`, and note that runs which would otherwise end
early get a different step-count denominator for `average_queue_length`.

### 3.5 Determinism — good news **[verified]**
Same seed, with/without a non-interacting obstacle: bit-identical trajectories for all
other vehicles (positions, speeds, departures, arrivals). Seeded pairs isolate the pure
blockage effect. Keep the manager write-free before `start_step`; still run ≥3 seeds per
arm since post-incident divergence is chaotic.

### 3.6 Experiment-validity decisions to record
- **Information asymmetry**: the LLM gets an explicit blockage section; MP/CoLight see
  only halting counts. Any LLM win conflates privileged information with reasoning.
  The `--hide_blockage_info` ablation (physically inject, hide from prompt) is the
  cleanest headline experiment this feature enables — promote it into the core matrix
  (LLM-with-info vs LLM-without-info vs baselines), not "later, optional".
- **Trapped-vehicle semantics**: `early_queued` includes unservable vehicles on blocked
  lanes; the prompt override sentence (§2.4) is the minimal fix. MP/CoLight chasing
  unservable queues is part of what the experiment measures — a documented decision.
- **CoLight arms**: eval-only blockage is the defensible "adaptation" claim (§2.5).

---

## 4. Copy / adapt / rewrite summary

| Unit | Verdict | Notes |
|---|---|---|
| Obstacle-insertion + speed-restriction TraCI code | **copy-paste** | best available idiom (alternatives checked: `setStop` = nondeterministic onset through live traffic; parkingArea/closingReroute = XML baked per scenario); freeze verified drift-free |
| `accident_single_lane.json`, `construction_zone.json` | **copy-paste** | lane IDs verified in the toy net |
| `utils/blockage_manager.py` | **copy-with-edits** | 290 → ~150 LOC: dicts + strict loader, deferred traci imports, activation fix, occupancy check, keep `reset()` + ASSUMPTION comments, drop dead code |
| `utils/scenario_config.py` loader | **copy-with-edits** | folded in; validation strengthened to replace what the dataclass caught at load time |
| Env hooks (init/tick/state/describe) | **rewrite** (~20 LOC) | teleport flag only for the no-output_dir case; unmappable-lane policy explicit |
| Blockage prompt section | **rewrite** (~20 LOC) | keep header + method phrasing verbatim; `<signal>` contract, not `<phase>` |
| Runner/CLI wiring | **rewrite** (~10 LOC, mostly runner_common) | no `state_is_empty` veto; fresh/reset manager per CoLight round |
| Metrics filtering/provenance, replay support, tests | **new** | filter points enumerated in §2.6; no usable source equivalent |
| `run_sumo_*.py`, `sumo_blockage_prompt.py` bulk, source `sumo_env.py`, `test_sumo_env.py`, `baseline.json`, wandb/CSV helpers | **skip** | dead, duplicated, or contradicts this repo's design |

Net effect: one new ~150-line module + ~120 lines of edits.

---

## 5. Implementation order

1. `utils/blockage_manager.py` (strict loader, pure core, occupancy check, `reset()`) +
   scenario JSONs + unit tests.
2. `SumoEnv` integration (init, tick, conditional teleport flag, state flags, obstacle
   filter, `describe_blockages` + unmappable-lane rejection) → GUI smoke run watching the
   queue form; assert obstacle persistence for the full window.
3. Metrics filtering (§2.6 — id-sets, step loop, `record_decision_wait`) + the
   obstacle-only/zero-traffic isolation check. Do this **before** any results are read.
4. Prompt section + `--blockage_scenario` in runner wiring → both golden tests, then one
   LLM run with the accident scenario reading `decisions.jsonl`.
5. Baselines + CoLight (per-round manager, eval-only default) + provenance
   (final_summary scenario, compare_controllers column, METRICS.md caveats).
6. Replay support.
7. Then, as the experiments demand: `--hide_blockage_info` ablation (early — it's the
   headline experiment), hangzhou/jinan scenario JSONs (check movement-level semantics),
   optional servable/trapped state split.
