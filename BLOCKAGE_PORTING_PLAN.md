# Blockage feature porting plan

Port lane-blockage injection + blockage-aware LLM prompts from
`C:\Users\m6722\Research\LLMTSCS-custom_prompts` (branch `sumo-backend`) into this repo.

Guiding rule: reuse the source's proven TraCI mechanics, JSON scenario format, and prompt
phrasing — but rebuild all the thin wiring natively in this repo's style. The source's
wiring is welded to a Gym-style env, episode loops, wandb, and CSV logging that this repo
deliberately does not have. Rebuilding ~40 lines of wiring is cheaper than adapting ~700
lines of theirs.

---

## 0. Decide the base branch FIRST

This branch (`blockage`) was cut from `main` (d249065). `advanced_colight_staging` is
**25 commits ahead** and rewrites the exact places this port hooks into:

- `88a60f2` refactors runners to share one run loop (where the blockage tick + empty-state veto go)
- `302438c` adds `--seed` to `SumoEnv.__init__` (whose signature we extend)
- `ae6bcf5` renames the prompt functions (whose signature we extend)
- `9223e68` adds METRICS.md (where blockage metric caveats belong)

**Recommendation: merge `advanced_colight_staging` into `blockage` before starting.**
Deterministic seeded runs are also what makes blockage A/B comparisons meaningful.
If you deliberately want to stay on `main`, re-check every file/line reference below —
they were verified against `main`.

---

## 1. What the source feature actually is

A scenario JSON declares scheduled blockages. A `BlockageManager` executes them against
the live TraCI session, ticked once per simulation step. Two methods:

1. **`obstacle_vehicle`** — insert a vehicle (`traci.route.add` + `traci.vehicle.add`),
   then on the next step position and freeze it
   (`moveTo` + `setSpeed(0)` + `setSpeedMode(0)` + `setLaneChangeMode(0)`, retried on
   `TraCIException` until SUMO actually places it). Removed with `vehicle.remove` at
   `end_step`. This physically blocks one lane at a given distance from the stop line.
2. **`speed_restriction`** — `traci.lane.setMaxSpeed(original * (1 - severity))`,
   original speed restored at `end_step`. Whole-lane slowdown; `position` is ignored.

The prompt side renders active blockages as a bullet list ("LANE BLOCKAGE CONTEXT")
appended to the traffic-state prompt, one line per blockage: lane, position, and either
"stopped vehicle — full blockage" or "speed restriction — N% reduction".

### Trust warnings (verified in the source — do not assume "it worked there")

- The feature was **never exercised against traffic**: every `<flow>` in the source's
  route file is commented out, and the LLM runner hardcodes `model = None`, so blockage
  prompts were never sent to any model.
- Two of the three shipped scenario JSONs **crash the loader** (`KeyError:
  'intersection_id'` — the field was added to the dataclass but never to the data files).
- Activation uses exact equality `current_step == start_step`, so a `start_step` the tick
  never lands on silently never fires.
- Leftover debug `print`s fire every step while an obstacle is pending.

The TraCI mechanics themselves are correct SUMO idioms and worth keeping; everything
around them gets rebuilt.

---

## 2. Target design

One new module, small edits at existing seams. No new abstractions.

### 2.1 New: `utils/blockage_manager.py` (~130–150 LOC) — copy-with-edits

Copy the source module, then trim to this repo's style:

- Blockages are **plain dicts** (this repo has zero dataclasses/typing): the JSON entries
  themselves, validated and defaulted by the loader. Keys: `blockage_id`, `lane_id`,
  `position` (metres upstream of the stop line), `start_step`, `end_step` (null = forever),
  `method`, `severity` (default 1.0).
- Fold `load_scenario(path)` (source `utils/scenario_config.py`, ~20 LOC after trim) into
  this module — one file owns the whole feature.
- **Keep verbatim:** the two-step obstacle insertion with deferred positioning + retry;
  the speed-restriction save/restore.
- **Fix:** activate on `current_step >= start_step` guarded by a `finished` set (not `==`);
  rename the typo'd `intesection_pos_to_lane_pos` helper; delete its debug print.
- **Delete:** `validate_blockages()` (half-implemented dead code — replace with an
  ~8-line real check: lane exists, method known, `0 <= severity <= 1`, called once after
  the simulation starts); `clear_all()` (no `reset()` here — one run = one fresh SUMO
  process); the `intersection_id` field (derivable from the lane at runtime); all
  `logging`/`typing`/docstring ceremony.
- **Keep the read API:** `step(current_step)`, `get_active_blockages()`,
  `get_blocked_lane_ids()`.

Clock convention: `step(t)` is always called with **simulation seconds**
(`traci.simulation.getTime()` / `env.get_current_step()`), in every loop that ticks it.
One clock everywhere avoids the off-by-one between the runners' 0-based loop variable and
sim time.

### 2.2 Scenario JSONs — copy-paste

`accident_single_lane.json` and `construction_zone.json` copy **verbatim**: their
`W2TLS_0`/`W2TLS_1` lane IDs exist in `simulations/single_intersection/net.xml` (189.6 m,
so positions 80/100 m are in range), and that net has real demand on the blocked lane
(0.15 prob flow). Put them in `simulations/single_intersection/scenarios/` — scenarios are
lane-ID-bound to a network, so they live next to it. Skip `baseline.json` (empty list ==
not passing the flag). Scenarios for hangzhou/jinan nets are new data authoring, later.

### 2.3 `sumo_env.py` — rewrite the hooks (~20 LOC total)

The source's env hooks can't be diff-applied (different env shape); re-express them:

- `__init__` gains `blockage_manager=None` (matches the existing `output_dir=None` style).
  When set, also append `['--time-to-teleport', '-1']` to the SUMO command — see §3.1.
- `step()` becomes: `traci.simulationStep()`, then
  `if self.blockage_manager: self.blockage_manager.step(self.get_current_step())`.
  This is the single choke point shared by runner.py / runner_baselines.py /
  runner_colight.py (replay_runner is separate, §2.7).
- `get_state()`: skip vehicles whose ID starts with the obstacle prefix (§3.2), and add a
  `blocked` flag to each lane's dict so blockage state flows into `decisions.jsonl` and
  CoLight features for free.
- New small helper `describe_blockages(intersection_id)` → list of dicts
  `{approach, movement, segment, method, severity}` for that intersection's blocked lanes.
  The env owns topology (`approach_mapping`, `movement_lane_map`, lane lengths), so the
  lane-ID → approach/segment translation happens here; the prompt builder stays
  topology-free.

### 2.4 `utils/prompt_builder.py` — rewrite, reusing source phrasing (~20 LOC)

Do **not** port `prompts/sumo_blockage_prompt.py` (345 LOC). Its `<phase>0–3</phase>`
answer contract, hardcoded lane tables, duplicate no-blockage builders, system-prompt
loading, and two-message format all duplicate or contradict what this repo already has.
About 10 lines of phrasing survive.

- New `build_blockage_section(blockage_descriptions)`: header + one bullet per blockage in
  the prompt's own vocabulary (approach + movement + segment, not raw lane IDs), reusing
  the source's method wording: "stopped vehicle — full blockage" /
  "speed restriction — N% reduction". Include one sentence stating that queued counts on
  blocked lanes include vehicles trapped behind the blockage that cannot reach the
  intersection (§3.3). Returns `''` for an empty list.
- Insert between the observation text and "Please answer:" in the existing prompt
  function; signature gains `blockages=None` (runner.py is the only caller).
- **Invariant: with no active blockages the prompt is byte-identical to today's** —
  preserves comparability with all existing runs. Enforce with a small test.
- Keep the `<signal>NAME</signal>` answer contract unchanged.

### 2.5 Runner wiring — rewrite (~10 LOC per runner)

In `runner.py`, `runner_baselines.py`, `runner_colight.py`:

- `--blockage_scenario` flag (path to JSON, default `None`; use `action`-free plain str).
- In `main()`: `blockage_manager = BlockageManager(load_scenario(path)) if path else None`,
  passed to `SumoEnv(...)`. Construction touches no traci, so building it before
  `start_simulation()` is safe; run the validation check right after.
- `runner.py`: the `state_is_empty` skip must **not** suppress the LLM call while
  blockages are active (a fully blocked, starved approach would otherwise never be
  decided on). Two-line veto.
- `runner.py` decision call site: pass `env.describe_blockages(iid)` into the prompt
  function.
- `runner_colight.py`: constructs a fresh `SumoEnv` **per training round** — construct the
  `BlockageManager` inside the round loop too, next to the env, so no stale active/finished
  state leaks across rounds. Also decide: blockages during training bake the blockage into
  the policy; for eval-only comparisons, pass the flag only to eval runs (document choice).

### 2.6 Provenance & metrics — new, no source equivalent (~15 LOC)

So blockage runs are distinguishable and analyzable afterwards:

- `run_details` / replay meta: record `blockage_scenario` path and scenario name; copy the
  scenario JSON into the run dir.
- `MetricsRecorder`: accept the manager (or `None`) at construction; each
  `record_step_summary` line gains `blocked_lanes` (list or count) — gives a time series
  to slice before/during/after windows.
- `final_summary.json` gains `blockage_scenario` name; `compare_controllers.py` gains a
  scenario column.

### 2.7 `replay_runner.py` — small rewrite (~10 LOC)

It does **not** use `SumoEnv` (raw `traci.start` + `simulationStep` loop), so the env tick
never fires there. Load the scenario recorded in replay meta, build its own manager, tick
it after `simulationStep()` with `traci.simulation.getTime()`. Without this, reruns of
blockage runs silently replay blockage-free and diverge (the obstacle also suppresses the
`getMinExpectedNumber` early-stop in the original but not the rerun).

### 2.8 Tests — new (source test asserts nothing)

Source `test_sumo_env.py` is a print-only smoke script on their Gym API — skip it.

- Pure-core unit test per the `tests/test_state_features.py` pattern: extract the schedule
  decision into a pure function (schedule + current step + active/finished sets →
  activate/deactivate lists) and test start/end/None-end/overlap/step-past-start cases
  with plain dicts, no SUMO.
- Prompt test: empty list → byte-identical prompt; one blockage → expected section text.
- One headless smoke run: `simulations/single_intersection` + `accident_single_lane.json`,
  assert the obstacle vehicle exists for the whole window and queue forms on `W2TLS_0`
  (this validates what the source never could — its demand was commented out).

---

## 3. Gotchas that must be handled (found during assessment)

### 3.1 SUMO teleport silently dissolves blockages
Default `--time-to-teleport` is 300 s. The accident scenario blocks a lane for 600 s —
jammed vehicles (and plausibly the frozen obstacle itself, which is not an exempt
`<stop>`) get teleported away mid-blockage, un-blocking the lane while the manager still
reports it active. No sumocfg in this repo sets the option. Fix: blockage runs pass
`--time-to-teleport -1` (§2.3). For strict A/B fairness, give the paired no-blockage
baseline run the same flag. Teleports already surface in `final_summary` /
`compare_controllers` — watch that column.

### 3.2 The obstacle vehicle pollutes every vehicle count
It is a real vehicle: it shows up in queue counts, accumulated waiting, trip metrics,
prompt state, CoLight reward, and MaxPressure pressure. Decisions:
- Reserve an ID prefix (`OBSTACLE_VEHICLE_PREFIX = 'obstacle_'` in `configurations.py`).
- Filter it in `MetricsRecorder` per-vehicle loops and in `get_state()`.
- Lane-level `getLastStepHaltingNumber` (MaxPressure, queue-length reward) cannot filter
  by ID — accept the +1 and document it in METRICS.md (defensible: a real sensor would
  also see a stopped vehicle).

### 3.3 Trapped vehicles are advertised to the LLM as servable
`get_state` counts every slow vehicle as `early_queued`, and the prompt defines
early-queued as "await passage permission" — so vehicles trapped *behind* the blockage
inflate exactly the number the LLM is taught to relieve with green. Minimal fix (chosen
here): the one-sentence disclaimer in the blockage section (§2.4). Optional follow-up:
split servable (downstream of blockage) vs trapped counts in `get_state`. Note MaxPressure
and CoLight will also chase unservable queues — that is part of what the experiment
measures; record it as a design decision, not an accident.

### 3.4 Small correctness items
- Activation `==` → `>=` with a finished-set (§2.1); with `end_step=None` blockages this
  also prevents re-activation.
- One clock (sim seconds) for every `step()` call site, including replay (§2.1, §2.7).
- `collision.action` defaults to `teleport` on dataset configs; the toy net already uses
  `warn`. If `moveTo` lands the obstacle on an occupied position it can remove the vehicle
  it hits — keep obstacle positions on lane stretches, and check teleport counts in smoke
  runs.

---

## 4. Copy / adapt / rewrite summary

| Unit | Verdict | Notes |
|---|---|---|
| Obstacle-insertion + speed-restriction TraCI code | **copy-paste** | proven SUMO idioms, keep verbatim inside the ported module |
| `accident_single_lane.json`, `construction_zone.json` | **copy-paste** | lane IDs match the toy net; drop nothing |
| `utils/blockage_manager.py` | **copy-with-edits** | 290 → ~140 LOC: dicts not dataclasses, fix activation + typo, delete dead code/logging |
| `utils/scenario_config.py` loader | **copy-with-edits** | ~20 LOC, folded into blockage_manager.py, drop `intersection_id` |
| Env hooks (init/tick/state) | **rewrite** (~20 LOC) | source hooks live in Gym-shaped methods this repo doesn't have |
| Blockage prompt section | **rewrite** (~20 LOC) | keep ~10 lines of phrasing; keep `<signal>` contract, not `<phase>` |
| Runner/CLI wiring | **rewrite** (~10 LOC × 3 runners) | one `--blockage_scenario` flag; no `--prompt_type`, no fallback routing |
| Metrics/provenance, replay support, tests | **new** | no usable source equivalent |
| `run_sumo_*.py`, `sumo_blockage_prompt.py` bulk, source `sumo_env.py`, `test_sumo_env.py`, `baseline.json`, wandb/CSV helpers | **skip** | dead, duplicated, or contradicts this repo's design |

Net effect: **one new ~150-line module + ~100 lines of edits** across existing files.

---

## 5. Implementation order

1. **Branch decision** (§0). Re-verify hook points if merging `advanced_colight_staging`.
2. `utils/blockage_manager.py` + scenario JSONs + pure-core unit tests.
3. `SumoEnv` integration (init, tick, teleport flag, state flags, obstacle filter) →
   GUI smoke run on `single_intersection` watching the queue form behind the obstacle.
4. Prompt section + `state_is_empty` veto + `runner.py` flag → byte-identical-prompt test,
   then one LLM run with the accident scenario reading `decisions.jsonl`.
5. Baselines + CoLight flags (fresh manager per round) + metrics provenance
   (step-summary field, final_summary name, compare_controllers column).
6. Replay support.
7. Later, as needed: scenario JSONs for hangzhou/jinan nets; optional servable/trapped
   state split; optional `--hide_blockage_info` ablation flag.
