# CoLight → SUMO Porting Plan (Phase 1)

Status: **decisions locked (§13). Awaiting go-ahead to start Phase 2 implementation.**

Locked: framework = **TF/Keras verbatim** (add `tensorflow-cpu`, `.h5` weights);
agent path = **`models_inference/RL/colight_agent.py`**; `lane_num_vehicle` = **8 movements**;
Phase-2 net = `hangzhou_1x1_bc-tyc_18041607_1h`; reward threshold = `MIN_SPEED=0.1`.

This plan ports the **CoLight method** from the CityFlow/TensorFlow source repo into
this SUMO/TraCI pipeline by *lifting the agent* and *rebuilding the round/replay loop
natively in SUMO*. It implements **CoLight only**, but designs the seam so FixedTime /
MaxPressure / AdvancedCoLight / MPLight slot in later by config, not rewrite.

---

## 1. Provenance

| Item | Value |
|---|---|
| Source repo | `github.com/usail-hkust/LLMTSCS` |
| Pinned commit | `d5d4180f34edb843e1d1b462d5846c75d6d4533a` |
| Commit date / subject | 2025-08-12 20:09:44 +0800 — "Update README.md" |
| Local read-only clone | `C:\Users\m6722\Research\LLMTSCS_ref` (OUTSIDE this repo) |
| Lifted file | `models_inference/RL/colight_agent.py` (graph-attn Q-net, replay, target net, `prepare_Xs_Y`, `train_network`) |
| Base classes it needs | `models/agent.py` (`Agent`), and the load/save + epsilon-decay glue from `models/network_agent.py` |

`colight_agent.py` will be kept **byte-diffable** against the pinned source: the network
graph, `MultiHeadsAttModel`, `RepeatVector3D`, `prepare_Xs_Y`, and `train_network` stay
verbatim. Only **I/O glue** (model dir listing, weight load/save filenames, the
`dic_path` plumbing) is isolated into thin overridable methods so the math stays diffable.

---

## 2. Repo reality check (must resolve before coding)

Three findings contradict or constrain the task spec; surfaced as decisions in §13.

1. **Framework conflict.** The venv has **torch 2.12.0 only — no TensorFlow/Keras**.
   The source agent is TF/Keras (`tensorflow.keras`, custom `RepeatVector3D` Keras
   layer, `.h5` weights). A *verbatim* lift therefore adds **tensorflow-cpu** as a new
   heavy dependency to a CPU-only torch stack.
2. **A prior PyTorch attempt existed.** `models_inference/RL/__pycache__/` still holds
   compiled `colight_agent`, `colight_network`, `replay_buffer` — PyTorch-style module
   names. The `.py` sources are gone and were **never committed** (not in git on any
   branch). This matches the project memory note: *"PyTorch CoLight RL baseline … to
   match the torch-based LLM stack."*
3. **Memory vs task spec.** Memory (June 2026) records a **PyTorch** decision; the task
   spec says **TF/Keras verbatim**. These are mutually exclusive. The task is the most
   recent instruction, but the conflict is large enough that I will not pick silently.

Everything else in the plan is **framework-agnostic** (features, cadence, adjacency,
reward, runner). Only §1's lifted file and the weight-save format depend on the §13.A
decision.

---

## 3. Component ledger — lift vs reuse vs new

| Component | Disposition | Notes |
|---|---|---|
| `colight_agent.py` (CoLightAgent, MultiHeadsAttModel, RepeatVector3D) | **LIFT verbatim** | network + replay + `prepare_Xs_Y` + `train_network` |
| `agent.py` base, epsilon-decay, load/save | **LIFT (trim)** | keep behavior; isolate `.h5`/path glue |
| `generator.py`, `construct_sample.py`, `pipeline.py`, `updater.py`, `cityflow_env.py` | **DO NOT LIFT** | CityFlow-welded; reimplement their flow in `runner_colight.py` |
| `utils/config.py` (source) | **READ ONLY** | hyperparameters → copied into our config (§11) |
| `sumo_env.py` (`SumoEnv`) | **REUSE unchanged** | `get_state` NOT modified — it serves the LLM path |
| `utils/phase_handler.py` (`PhaseHandler`) | **REUSE unchanged** | chosen phase routed through it (identical transitions) |
| `utils/metrics_recorder.py` | **REUSE unchanged** | eval reports through it for LLM-comparability |
| `utils/replay_recorder.py` | **REUSE unchanged** | phase-timeline replay |
| `configurations.py` | **EXTEND additively** | new `PHASE` map + `COLIGHT_AGENT_CONF` (existing keys untouched) |
| `utils/state_features.py` | **NEW** | the translator: feature registry (3 keys) |
| `models_inference/RL/colight_agent.py` | **NEW (= lifted)** | see §1 |
| `runner_colight.py` | **NEW** | SUMO train+eval loop; mirrors `runner.py`; does **not** modify it |
| `VALIDATION.md` | **NEW (built alongside)** | §12 |

---

## 4. Feature registry design (`utils/state_features.py`)

A dict mapping `feature_name -> fn(env, intersection_id) -> value`, mirroring the
source `LIST_STATE_FEATURE`. A model declares the features it wants by name; the runner
emits a **feature-KEYED dict** per intersection (never a positional tuple), so adding
`pressure` / `lane_enter_running_part` / `time_this_phase` later is purely additive.

```
FEATURE_REGISTRY = {
    "cur_phase":        fn,   # -> [phase_index]   (agent expands to 8-bit via PHASE)
    "lane_num_vehicle": fn,   # -> [8 movement counts] in canonical movement order
    "adjacency_matrix": fn,   # -> [self_idx, nbr1_idx, ... ] length num_neighbors
}
def build_state(env, intersection_id, feature_names): return {f: REGISTRY[f](env, id) ...}
```

Design rules:
- `cur_phase` and `lane_num_vehicle` are **derived from `env.get_state(id)`** — no new
  TraCI calls (reuse the early_queued + 3-segment counts it already computes).
- `adjacency_matrix` needs global topology; a one-time `AdjacencyBuilder` precomputes the
  neighbor map from SUMO junction positions and the per-id row is cached.
- Signature is exactly `fn(env, intersection_id)` as the task specifies. `cur_phase`
  recovers the *logical* phase index from `env.get_current_phase(id)` +
  `env.intersection_config` via `get_phase_name` (valid because decisions are read at the
  end of a GREEN window, when the RYG string is a known green phase).

---

## 5. State / Action / Reward contract (resolved)

**State (per intersection):** `{"cur_phase":[idx], "lane_num_vehicle":[8], "adjacency_matrix":[k]}`.
This is exactly what `CoLightAgent.convert_state_to_input` / `prepare_Xs_Y` consume.

**Action:** `choose_action` returns a phase **index** per agent (0..3). The runner maps
index → phase **name** (`ETWT/NTST/ELWL/NLSL`) and routes it through
`PhaseHandler.activate_phase(name)`. Transitions (green→yellow→all-red→green) are thus
**identical** to the LLM and every future controller.

**Reward:** `-0.25 × queue_length`, where `queue_length` = count of **stopped vehicles**
(`speed < MIN_SPEED = 0.1`, the *same predicate MetricsRecorder uses*) on that
intersection's controlled lanes, **averaged over the decision window** (mirrors source
`reward_average` over `MEASURE_TIME`). Note: this uses `MIN_SPEED` (0.1) — matching the
eval metric and CityFlow's waiting-count semantics — **not** the `STOP_SPEED_EARLY_QUEUE`
(1.39) that `get_state`'s `early_queued` uses. Reward stays consistent with the reported
metric. (Coefficient and predicate are config-exposed.)

---

## 6. The #1 risk — lane count & ordering (RESOLVED)

Source `_cal_len_feature` hardcodes **12** lane features (+8 phase) because CityFlow's
nets are 3 lanes/approach → 12 entering lanes. My hangzhou 1x1 is **2 lanes/approach**
(8 entering lanes); `three_lane` would be 12. A fixed "12" is **not portable**.

**Resolution: represent `lane_num_vehicle` over the 8 canonical movements, not raw lanes.**
- Canonical movement order (from source `list_lane_order`): `[WL, WT, EL, ET, NL, NT, SL, ST]`.
- Each entry = total vehicles on that movement's lane(s) = `early_queued + seg1 + seg2 + seg3`,
  aggregated across however many lanes (2 or 3) serve the movement. Sourced from
  `get_state(id)["movement_states"]`:
  `WT=ETWT.West, ET=ETWT.East, WL=ELWL.West, EL=ELWL.East, NT=NTST.North, ST=NTST.South,
   NL=NLSL.North, SL=NLSL.South`.
- This yields a **fixed length-8 vector for any net** (2-lane *or* 3-lane), satisfying the
  memory's "dataset-flexible, not fixed-size" goal. It also aligns 1:1 with the phase
  one-hot's 8 movements (§7), which the source's mismatched 12-vs-8 concat does *not*.

**Consequence for the lifted agent:** the only change to `_cal_len_feature`'s `12` is to
read the lane-feature width from config (default 12 → set 8). One commented line; stays
diffable. The agent's MLP is width-driven by `len_feature`, so nothing else changes.

> Deviation from source is deliberate and isolated to **the translator + one config-driven
> width**, not the agent math. Flagged for your objection in §13.C.

---

## 7. PHASE one-hot mapping (additive config)

`cur_phase=[idx]` is expanded by the agent via `dic_traffic_env_conf['PHASE'][idx]` to an
8-bit movement vector over `[WL, WT, EL, ET, NL, NT, SL, ST]`. Source uses **1-based** phase
keys; my logical ids are **0-based**, so I define a fresh `PHASE` keyed to *my* ids for the
active config. For `two_lane_1x1` (NTST=0, ETWT=1, NLSL=2, ELWL=3):

```
PHASE = {
  0: [0,0,0,0,0,1,0,1],  # NTST -> NT,ST
  1: [0,1,0,1,0,0,0,0],  # ETWT -> WT,ET
  2: [0,0,0,0,1,0,1,0],  # NLSL -> NL,SL
  3: [1,0,1,0,0,0,0,0],  # ELWL -> WL,EL
}
```

(Vectors are the source's, re-keyed to my phase ids. The bit positions match the §6
movement order, so phase one-hot and `lane_num_vehicle` are co-indexed.)

---

## 8. Adjacency construction

Replicates source `_adjacency_extraction` using SUMO instead of CityFlow:
- Assign each controlled junction a stable global index (sorted id order).
- Positions from `traci.junction.getPosition(id)`; pairwise Euclidean distance.
- For each agent: `num_neighbors = min(TOP_K_ADJACENCY=5, num_agents)` nearest **including
  self first**, then nearest others → that agent's `adjacency_matrix` row of indices.
- `CoLightAgent.adjacency_index2matrix` turns rows into the one-hot `[batch, agent, nei, agent]`.
- **1x1:** `num_agents=1 → num_neighbors=1 → row=[0]` (collapses to self; wiring-correct,
  attention is a no-op). **4x4:** real 5-neighbor attention (Phase 3).

Precomputed once per run by `AdjacencyBuilder`; static topology ⇒ no per-step cost.

---

## 9. Decision cadence & multi-agent time-alignment (silent-mismatch risk — RESOLVED)

**Cadence:** one decision per **green window = `default_green_duration = 30` = source
`MIN_ACTION_TIME=MEASURE_TIME`**, identical to the LLM. The trigger is
`PhaseHandler.switch_phase` after green elapses.

**Alignment problem:** CoLight stacks per-agent samples by index into `[batch, agent, …]`
and its neighbor attention mixes agents *at the same sample index* — so the i-th transition
of **every** agent must come from the **same decision round**. But `PhaseHandler` makes a
phase *change* cost +5s (yellow 3 + red 2) while *no-change* costs 0, which would desync
agents across rounds.

**Resolution — round-synchronized batched decisions:** the runner advances the sim one
step at a time calling every `handler.step()`, and issues a decision round **only when
`all(h.switch_phase)`**. Non-changers reach `switch_phase` at +30 and *hold* their green
(still flowing) until changers finish at +35; then all agents decide **together**. This
keeps every agent's i-th sample on the same round boundary using `PhaseHandler` *unchanged*.
Trivially correct for 1x1 (single agent); essential for 4x4.

Reward window per round = that shared interval (30 or 35 steps), so per-agent reward
averages cover the same wall-clock span.

---

## 10. Transition tuple shape & training loop (`runner_colight.py`)

`CoLightAgent.prepare_Xs_Y(memory)` expects `memory[j][i]` = the **j-th agent's i-th
transition**, a 7-tuple unpacked as `(state, action, next_state, reward, _, _, _)`. So the
runner accumulates, per round, per agent:

```
(state_dict_j, action_j, next_state_dict_j, reward_avg_j, reward_avg_j, t, round_tag)
```

with `state_dict` carrying `cur_phase / lane_num_vehicle / adjacency_matrix`.

**Per-round loop (mirrors source generator → construct_sample → updater → model_test):**
1. `reset` sim; build `PhaseHandler` per intersection (start phase), `MetricsRecorder`,
   `ReplayRecorder`.
2. Run to the first synchronized decision point; record `state_k` for all agents.
3. `actions = agent.choose_action(round, [state_k...])` (ε-greedy in training).
4. `handler.activate_phase(idx→name)` for all; advance to next sync point accumulating
   per-agent stopped counts.
5. At next sync point: record `state_{k+1}`, compute `reward_avg`, append transitions.
   The LAST decision (no following sync point) is finalized at episode end using the
   terminal observation as next_state (cur_phase from the handler's logical phase, since
   the terminal RYG may be mid-transition) — mirrors source `construct_sample` keeping the
   last full-window decision.
6. End of episode → `agent.prepare_Xs_Y(transitions) → train_network()`; every
   `UPDATE_Q_BAR_FREQ` rounds copy q→q_bar (target net); `save_network(round)`.

**EVAL:** load weights, set `EPSILON=0` (greedy), run ONE episode driving
`MetricsRecorder.record_step_summary` each step + `save_final_summary` — the *same*
comparable metrics path as the LLM runner. (`record_decision` is LLM-shaped; for RL I
either skip it or pass RL-adapted fields without touching `MetricsRecorder`.) Fixed seeds;
report mean±std over seeds.

---

## 11. Hyperparameter parity (reproduced exactly from source `utils/config.py`)

| Param | Source | This port |
|---|---|---|
| GAMMA | 0.8 | 0.8 |
| NORMAL_FACTOR | 20 | 20 |
| LEARNING_RATE | 0.001 | 0.001 |
| LOSS | mean_squared_error | mean_squared_error |
| BATCH_SIZE | 20 | 20 |
| SAMPLE_SIZE | 3000 | 3000 |
| MAX_MEMORY_LEN | 12000 | 12000 |
| EPOCHS | 100 | 100 |
| UPDATE_Q_BAR_FREQ | 5 | 5 |
| UPDATE_Q_BAR_EVERY_C_ROUND | False | False |
| EPSILON / DECAY / MIN | 0.8 / 0.95 / 0.2 | 0.8 / 0.95 / 0.2 |
| TOP_K_ADJACENCY | 5 | 5 |
| CNN_layers | [[32,32]] | [[32,32]] |
| NUM_ROUNDS | 100 | 100 |
| MIN_ACTION_TIME = MEASURE_TIME | 30 | 30 (= green duration) |
| reward: queue_length coeff | −0.25 | −0.25 |

This table also goes in `VALIDATION.md`.

---

## 12. Validation plan (`VALIDATION.md`, built alongside in Phases 2–3)

Not validated by matching CityFlow numbers (different simulators). Component-level fidelity:
- **Provenance**: pinned commit (§1) + diff of `colight_agent.py` vs source.
- **Parity table**: §11.
- **Per-feature unit tests**: a hand-built SUMO state → assert `state_features` emits the
  exact dict shape + the §6 movement ordering CoLight expects (one test per feature key).
- **Behavioral (SUMO)**: (a) training reward improves over rounds; (b) CoLight > MaxPressure
  > FixedTime once those exist; (c) attention ablation (neighbors=self) degrades 4x4.
- **Fairness harness**: all controllers share env, traffic files, incident injection,
  `MetricsRecorder`, 30s cadence, warmup, seeds — only the decision-maker varies.
- **Eval**: greedy, fixed seeds, mean±std over seeds.

---

## 13. Decisions (RESOLVED 2026-06-24)

**A. Framework → TF/Keras verbatim.** Add `tensorflow-cpu` to the venv; lift the agent
  unchanged (Keras graph, `.h5` weights), keep byte-diffable vs the pinned source. Only the
  §6 `_cal_len_feature` width and isolated I/O glue change.

**B. Agent file location → `models_inference/RL/colight_agent.py`.** Matches the repo's
  `models_inference/{LLM,RL}` convention and the prior attempt's location.

**C. `lane_num_vehicle` → 8 movements.** As §6. Translator-side aggregation; agent width
  config-driven (12→8).

**D. Phase-2 net → `hangzhou_1x1_bc-tyc_18041607_1h`** (config `two_lane_1x1`).

**E. Reward threshold → `MIN_SPEED=0.1`** (matches MetricsRecorder & source semantics).
