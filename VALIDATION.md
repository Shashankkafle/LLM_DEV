# CoLight Port — Validation

Component-level fidelity checks (NOT a match against CityFlow numbers — different
simulators, uninterpretable). Built alongside the implementation. Phase 2 covers the
hangzhou 1x1 single-intersection case; Phase 3 adds the 4x4 / attention checks.

How to run everything in this doc:
```
# repo root; traci lives in $SUMO_HOME/tools, agent needs legacy Keras 2
PYTHONPATH="<repo>;%SUMO_HOME%\tools" TF_USE_LEGACY_KERAS=1 \
  <venv>/Scripts/python.exe tests/test_state_features.py
PYTHONPATH="<repo>;%SUMO_HOME%\tools" TF_USE_LEGACY_KERAS=1 \
  <venv>/Scripts/python.exe runner_colight.py --mode train_eval \
  --num_rounds 3 --simulation_steps 360 --epochs 3 --test_name colight_smoke
```

---

## 1. Provenance

| Item | Value |
|---|---|
| Source repo | `github.com/usail-hkust/LLMTSCS` |
| Pinned commit | `d5d4180f34edb843e1d1b462d5846c75d6d4533a` (2025-08-12) |
| Read-only clone | `C:\Users\m6722\Research\LLMTSCS_ref` |
| Lifted file | `models/colight_agent.py` → `models_inference/RL/colight_agent.py` |
| Base class | `models/agent.py` → `models_inference/RL/agent.py` (byte-identical) |

**Diffability check** (`diff --strip-trailing-cr -b source mine`): the agent body —
network graph, `MultiHeadsAttModel`, `RepeatVector3D`, `prepare_Xs_Y`, `train_network`,
load/save — is **byte-identical** to source. The only deviations are:

1. A clearly delimited 29-line **PORT GLUE** header (sets `TF_USE_LEGACY_KERAS=1`;
   flips tf-keras `enable_unsafe_deserialization`). Not in source.
2. One line in `_cal_len_feature`: `N += 12` → `N += self.dic_traffic_env_conf.get(
   "NUM_LANE_FEATURES", 12)` (+ explanatory comment). Inert for source-shaped configs
   (defaults to 12).

`agent.py` is byte-identical to source.

---

## 2. Hyperparameter parity (this port vs source `utils/config.py`)

| Param | Source | Port (`COLIGHT_AGENT_CONF`) | ✓ |
|---|---|---|---|
| GAMMA | 0.8 | 0.8 | ✓ |
| NORMAL_FACTOR | 20 | 20 | ✓ |
| LEARNING_RATE | 0.001 | 0.001 | ✓ |
| LOSS_FUNCTION | mean_squared_error | mean_squared_error | ✓ |
| BATCH_SIZE | 20 | 20 | ✓ |
| SAMPLE_SIZE | 3000 | 3000 | ✓ |
| MAX_MEMORY_LEN | 12000 | 12000 | ✓ |
| EPOCHS | 100 | 100 | ✓ |
| PATIENCE | 10 | 10 | ✓ |
| UPDATE_Q_BAR_FREQ | 5 | 5 | ✓ |
| UPDATE_Q_BAR_EVERY_C_ROUND | False | False | ✓ |
| EPSILON / DECAY / MIN | 0.8 / 0.95 / 0.2 | 0.8 / 0.95 / 0.2 | ✓ |
| TOP_K_ADJACENCY | 5 | 5 | ✓ |
| CNN_layers | [[32,32]] | [[32,32]] | ✓ |
| NUM_ROUNDS | 100 | 100 (`COLIGHT_NUM_ROUNDS`) | ✓ |
| MIN_ACTION_TIME = MEASURE_TIME | 30 | 30 (= green duration) | ✓ |
| reward queue coeff | −0.25 | −0.25 (`COLIGHT_REWARD_QUEUE_COEFF`) | ✓ |

---

## 3. Per-feature unit tests (`tests/test_state_features.py`) — **8/8 PASS**

Each feeds a hand-built SUMO state into a pure core (no live simulator) and asserts the
exact dict shape + lane ordering CoLight expects.

| Test | Asserts |
|---|---|
| `test_build_phase_onehot_matches_source_vectors` | PHASE one-hot for two_lane_1x1 equals source's 8-bit vectors, re-keyed to my 0-based ids; each lights exactly 2 of 8 movements |
| `test_phase_index_from_ryg_resolves_green` | green RYG string → correct logical phase index |
| `test_phase_index_from_ryg_rejects_non_green` | all-red RYG raises (decisions must be read at green) |
| `test_movement_vehicle_counts_ordering_and_aggregation` | 8-vector in `[WL,WT,EL,ET,NL,NT,SL,ST]` order; per-movement total = early_queued + 3 segments |
| `test_adjacency_single_intersection_is_self` | 1x1 → row `[0]` (self) |
| `test_adjacency_line_topk_self_first` | top-k neighbors, self always first |
| `test_expand_state_for_memory_expands_cur_phase` | stored state expands cur_phase to 8-bit, leaves other features + original untouched |
| `test_combined_feature_width_is_16` | cur_phase(8) + lane_num_vehicle(8) = 16 = `agent.len_feature` |

---

## 4. Lifted-agent smoke test (`scratchpad/smoke_colight.py`) — **PASS**

Under TF 2.21 + tf-keras (legacy Keras 2): `build_network` (incl. `Adam(lr=)`, `Lambda`,
`RepeatVector3D`), `choose_action`, `prepare_Xs_Y` (Xs `[(25,1,16),(25,1,1)]`, Y
`(25,1,4)`), `train_network`, and `.h5` save/load all succeed. Confirms the two state
representations: **live** state (`cur_phase=[raw_idx]`, expanded on the fly by
`choose_action`) vs **stored** transition state (`cur_phase` pre-expanded to 8-bit, the
form `prepare_Xs_Y` consumes). The runner stores expanded `cur_phase`; mismatching this
would silently break the training feature width.

---

## 5. End-to-end SUMO run (`runner_colight.py`, hangzhou 1x1) — **PASS**

`--mode train_eval --num_rounds 3 --simulation_steps 360 --epochs 3`:
- 3 training rounds executed: SUMO episode → state_features → `choose_action` →
  transitions → `prepare_Xs_Y` → `train_network` → periodic target-net copy → `.h5` save.
- Epsilon decayed across rounds (0.760 → 0.722); round reward improved (−77.9 → −58.3
  over the visible rounds — directional only at this scale).
- Eval loaded `round_2` weights, ran one **greedy** (ε=0) episode through
  `MetricsRecorder`, wrote `final_summary.json` (travel_time, waiting_time, queue_length,
  vehicle accounting) — the same comparable-metrics path as the LLM runner.

Notes:
- `total_decisions: 0` / `hallucination_rate: null` in the summary are expected — those
  counters are LLM-specific (`record_decision`); RL eval reports the controller-agnostic
  step/trip metrics. `MetricsRecorder` is reused unmodified.
- SUMO "missing yellow phase" warnings are benign (identical to the LLM path; PhaseHandler
  inserts our own yellow/all-red). One teleport occurred under random high-ε actions —
  expected; watch for gridlock as a tuning signal, not a port bug.

---

## 6. Validated contracts (silent-mismatch risks, now closed)

- **lane_num_vehicle = 8 movements** (not source's 12 lanes); width config-driven, agent
  diffable. (§3, §4)
- **PHASE one-hot** re-keyed to 0-based ids, co-indexed with lane_num_vehicle. (§3)
- **Two state representations** (live raw vs stored expanded cur_phase). (§4)
- **Round-synchronized decisions** (`all(handler.switch_phase)`) keep multi-agent samples
  time-aligned. Trivial at 1x1; exercised for real in Phase 3.
- **Reward** = −0.25 × stopped count (SUMO halting = MIN_SPEED predicate), window-averaged.
- **Cadence** = 30s green = source MIN_ACTION_TIME = LLM cadence.
- **Env requirement**: `tensorflow-cpu==2.21.0` + `tf-keras==2.21.0`; `TF_USE_LEGACY_KERAS=1`
  (set by the agent module). numpy 2.x / torch untouched.

---

## 7. Adversarial correctness review (4 dimensions → independent verification)

A multi-agent review (reward/window, state-shape, alignment/adjacency, fidelity/reuse),
each finding independently re-verified by a skeptic. 3 candidates, 2 confirmed (both low),
1 dismissed.

| Finding | Verdict | Action |
|---|---|---|
| Terminal decision of each episode never finalized into a transition (~1/120, no corruption) | confirmed, low | **FIXED** — `run_episode` now closes the last decision at episode end using the terminal observation (cur_phase from handler). Re-run: `transitions_this_round` 9 → 10. |
| `DEFAULT_SIMULATION_CONFIG` differs from committed HEAD | confirmed, low | **Not mine** — pre-existing working-tree change (with `runner.py`'s `mock_llm_inference`), present before this work. CoLight changes are strictly additive; left untouched, flagged to the user. |
| Target-net (`q_network_bar`) cadence differs from source | dismissed (not-a-bug) | Documented, contract-sanctioned adaptation (PORTING_PLAN §10); persistent agent copies q→q_bar every UPDATE_Q_BAR_FREQ rounds. |

## 8. Phase 3 — multi-intersection (hangzhou 4x4)

Net: `hangzhou_4x4_gudang_18041610_1h` — **16 TLs, all 36-link / 3-lane uniform**, so it
reuses the existing `three_lane` config (no new config). Run:
```
runner_colight.py --mode train_eval --simulation_config \
  dataset/sumo_version/hangzhou_4x4_gudang_18041610_1h/roadnet.sumocfg \
  --intersection_config three_lane --num_rounds 100 --simulation_steps 1200
```

- **`three_lane` config matches this net.** Each of the 4 phases set on all 16 TLs, then
  `get_state` passes its approach-guard on corner / interior / far-corner intersections —
  i.e. every phase's lit links map to the expected approach. (`scratchpad/probe_4x4b.py`)
- **Adjacency is exact.** Each intersection resolves to self + its 4 orthogonal grid
  neighbors; e.g. `intersection_2_2` (idx 5) → `[5,4,6,1,9]`. `num_neighbors = min(5,16) = 5`.
- **Trains + evals end-to-end** with 16 synchronized agents and real 5-neighbor attention;
  all 16 buffers stay equal-length (10/round at 360 steps, 34/round at 1200 steps) — the
  round-synchronized scheme holds at scale.
- **Training reward improves over rounds** (check a, 25-round run): −1148 (r0) → −832 (r10)
  → −615 (r20) → −571 (r24) as epsilon decays 0.80 → 0.23.

### Attention ablation (`--ablate_attention`)

The check has two parts; I separate them because the spec's phrasing ("ablation degrades
performance") conflates correctness with training outcome.

**(1) Correctness — attention is WIRED. PASS (definitive).**
`scratchpad/attn_wiring_test.py`: one trained network, identical features, real-neighbor
vs self-only adjacency as the ONLY varied input. Result: `max|ΔQ| = 0.142`, `mean|ΔQ| =
0.032`, and the greedy action flips at 1/16 intersections. A dead/disconnected adjacency
input would give byte-identical outputs — it does not — so neighbor information genuinely
flows through `MultiHeadsAttModel` into decisions. This is the correctness guarantee the
ablation exists to provide.

**(2) Performance — does attention HELP? Inconclusive, but the gap closes with budget.**
Single-seed greedy eval, hangzhou 4x4 (`average_queue_length`, lower = better):

| Budget | with attention | ablated (self-only) | gap (attn − ablate) |
|---|---|---|---|
| 25 rounds | 53.6 | 47.5 | +6.1 (ablate better) |
| 60 rounds | 47.7 | 46.5 | +1.2 (ablate better) |

Single-seed (seed 42), ablation never *degraded* performance and the gap shrank 6.1 → 1.2 as
training increased. The ablation does NOT change the architecture or parameter count (both
networks are identical 18,628-param models; only the adjacency INPUT differs — real neighbors
vs self), so this is not "a bigger model underfitting."

**A1 — multi-seed @ 60 rounds (the decisive rung): the single-seed gap was noise.**

| seed | attention | ablated | gap (attn − ablate) |
|---|---|---|---|
| 42 | 47.73 | 46.51 | +1.22 (ablate better) |
| 1 | 45.51 | 46.08 | −0.57 (attn better) |
| 7 | 46.04 | 47.99 | −1.95 (attn better) |
| **mean ± std** | **46.43 ± 0.95** | **46.86 ± 0.82** | **−0.43 (attn better)** |

The per-seed gap **flips sign**, and averaged over seeds attention is marginally *better*
(46.43 vs 46.86) — a difference well inside the ±~0.9 seed std. So at 60 rounds attention and
self-only are at **statistical parity, attention nominally ahead**; the earlier single-seed
"self-only is better" was seed noise, not a coordination deficit. (Travel time is likewise a
wash: attn mean 335.2 vs ablate 333.8.)

**A2 — full budget @ 100 rounds (seed 42, same net):** queue 45.24 (attn) vs 44.25
(ablate); travel 331.5s (attn) vs 335.4s (ablate). The metrics disagree and both gaps are
small: attention wins on travel time, self-only wins on queue by ~1.0 — and seed 42 is
exactly the seed A1 showed is queue-biased toward self-only (seeds 1, 7 favored attention).
The seed-42 queue gap keeps shrinking with budget (6.1 → 1.2 → 1.0 at 25 → 60 → 100 rounds)
without crossing to attention-favored, while travel time flips to attention-favored by 100
rounds. So even at full budget there is **no robust margin either way on this net**.

### Ablation ladder — final synthesis

| Rung | Setup | Result |
|---|---|---|
| Wiring (deterministic) | one trained net, real vs self adjacency | **attention is wired** — Q-values change, 1 greedy action flips (§8.1) |
| Single-seed trend | seed 42, 25/60/100 rounds | gap shrinks 6.1 → 1.2 → 1.0 (queue); travel flips to attn by 100 |
| **A1 multi-seed** | 3 seeds @ 60 rounds | **parity** — gap flips sign across seeds; mean 46.43 (attn) vs 46.86 (ablate) |
| A2 full budget | seed 42 @ 100 rounds | mixed/small — attn better travel, self-only better queue (seed-42 bias) |
| A3 other net (`syn_4x4_gaussian`) | seed 42 @ 60 rounds | parity again — attn queue 294.97 / travel 247.8s vs ablate 284.28 / 245.7s; net is near-gridlock (queue ~290 vs Gudang ~46), so a poor discriminator |

**A3 — other net (`syn_4x4_gaussian`, uniform synthetic demand, seed 42 @ 60 rounds):**
attn queue 294.97 / travel 247.8s / 2871 completed vs ablate queue 284.28 / travel 245.7s /
2865 completed. Self-only is marginally better again — but (a) seed 42 is the seed A1 showed is
queue-biased toward self-only, and (b) this net is **oversaturated** (queue ~290 ≈ near
gridlock, 6× Gudang), so neither controller can do much and it barely discriminates coordination
value. Same parity conclusion as Gudang.

**Conclusion (honest):** CoLight's neighbor attention is **correctly wired** (proven), and on
both 4x4 nets tested (Gudang real traffic and `syn_4x4_gaussian` synthetic) at these budgets it
performs at **statistical parity** with isolated (self-only) control — it neither clearly helps
nor hurts; apparent single-seed gaps are seed noise. A clear *positive* coordination margin (the
paper's claim) is not demonstrated here and would likely need traffic with stronger corridor
structure (e.g. `manhattan_16x3`) and/or more extensive tuning — consistent with the project
note that exact paper numbers require CityFlow. This does not weaken the baseline: its job is to
be a faithful, runnable, correctly-wired comparator, which it is. **Status: correctness proven;
performance at parity across both tested nets, reported honestly rather than asserted.**

## 9. Still out of scope

- A clear positive coordination margin — would need corridor-structured traffic
  (`manhattan_16x3`) and/or extensive tuning; parity is the supported result on both 4x4 nets
  tested. (Multi-seed on the other nets and full-budget multi-seed not run.)
- Cross-controller ordering (CoLight > MaxPressure > FixedTime) — needs those baselines,
  which are explicitly not built yet.
