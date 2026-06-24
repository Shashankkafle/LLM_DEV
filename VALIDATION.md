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

**(2) Performance — does attention HELP? Inconclusive at low budget.**
25-round, single-seed, greedy eval: attention did NOT degrade when ablated; self-only was
marginally *better* (queue 47.5 vs 53.6; travel 336.5 vs 340.3s). Note the ablation does NOT
change the architecture or parameter count (both networks are identical 18,628-param models;
only the adjacency INPUT differs — real neighbors vs self). So this is not "a bigger model
underfitting"; it is that with real adjacency the network must learn to exploit a
higher-information but noisier neighbor signal, which is harder to optimize at a small budget
than the effectively-local self-only problem. Consistent with undertraining (source uses 100
rounds) and single-seed noise. A larger-budget run (60 rounds) is in progress; a fully
conclusive performance ablation would average over multiple seeds at the full budget.
**Status: correctness proven; performance benefit not yet demonstrated and may be
traffic/budget dependent — reported honestly rather than asserted.**

## 9. Still out of scope

- Cross-controller ordering (CoLight > MaxPressure > FixedTime) — needs those baselines,
  which are explicitly not built yet.
- Multi-seed mean±std performance ablation at the full 100-round budget.
