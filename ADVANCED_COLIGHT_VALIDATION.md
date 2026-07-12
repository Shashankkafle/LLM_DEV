# Advanced CoLight — Validation (branch `advanced_colight`)

Advanced CoLight reuses the **same lifted `CoLightAgent`** as CoLight; it differs only in
its state features. So almost all of CoLight's validation (agent fidelity, attention wiring,
adjacency, PHASE one-hot, cadence, reward, round-synchronization) carries over unchanged —
see VALIDATION.md. This doc covers only what is NEW.

## 1. Provenance

| Item | Value |
|---|---|
| Source | `usail-hkust/LLMTSCS` @ `d5d4180`, `run_advanced_colight.py` |
| Agent | `"AdvancedColight": CoLightAgent` — the SAME agent (no new agent file) |
| Source features | `cur_phase`, `traffic_movement_pressure_queue_efficient`, `lane_enter_running_part`, `adjacency_matrix` |
| Source feature fns | `cityflow_env._get_traffic_movement_pressure_efficient`, `_get_part_traffic_movement_features` |

## 2. Reuse vs new

| Component | Disposition |
|---|---|
| `CoLightAgent`, runner loop, `SumoEnv`, `PhaseHandler`, `MetricsRecorder`, adjacency, PHASE one-hot, reward, 30s cadence | **REUSED unchanged** |
| 2 new feature fns + per-movement topology helper in `utils/state_features.py` | **NEW (additive registry entries)** |
| `COLIGHT_FEATURES` / `ADVANCED_COLIGHT_FEATURES` / `ADVANCED_COLIGHT_AGENT_CONF` in `configurations.py` | **NEW (additive)** |
| `VARIANTS` registry + `--variant` flag in `runner_colight.py` | **NEW** (loop now variant-parameterized; terminal-finalize + build_state no longer hardcode `lane_num_vehicle`) |

## 3. The two new features (8-movement representation)

**`traffic_movement_pressure_queue_efficient`** — source: per entering movement,
`entering_queue - exiting_queue / 3`. Port: `entering_halting[m] - exiting_halting[m] /
num_outgoing_lanes`, over the 8 canonical movements.
- entering halting = `getLastStepHaltingNumber` summed over the movement's entering lanes.
- exiting halting = downstream edge's `getLastStepHaltingNumber`. **#1 contract detail
  (resolved):** the exiting/downstream queue is NOT in `get_state`. Resolved by deriving each
  movement's outgoing edge from the controlled-link `to_lane` (the lane after the junction) —
  reuses existing topology, no net re-parsing. Verified on the real 1x1 net: `NT→road_1_1_3`
  (South ✓), `ET→road_1_1_2` (West ✓), etc.
- Normalization by the actual outgoing-edge lane count (vs source's hardcoded /3) generalizes
  to 2-lane and 3-lane nets. Queues are halting (stopped) counts, matching source's waiting count.

**`lane_enter_running_part`** — source: running (moving) vehicles in the lane part nearest
the stopline. Port: `get_state`'s `segment_1` per movement (nearest third, moving vehicles —
`get_state` already excludes stopped vehicles into `early_queued`). Reuses `get_state`; no new
SUMO calls. (Source uses a fixed near-stopline window; port uses the net-relative nearest third.)

**Assumption:** the per-movement topology infers movement type from the phase name's 2nd char
('T'/'L'), valid because the 4 protected phases (ETWT/NTST/ELWL/NLSL) each pair two same-type
movements. (8-phase mixed-type configs would need a per-link type; out of scope.)

## 4. Per-feature unit tests (`tests/test_advanced_features.py`) — **4/4 PASS**

| Test | Asserts |
|---|---|
| `test_efficient_pressure_normalizes_by_outgoing_lanes` | pressure = entering − exiting/lanes, 8-vector in `[WL,WT,EL,ET,NL,NT,SL,ST]` order |
| `test_efficient_pressure_guards_zero_lanes_and_missing` | zero-lane guard → exit_norm 0; missing movement → 0 |
| `test_segment1_counts_picks_nearest_segment_only` | running = `segment_1` only (early_queued/seg2/seg3 are decoys); correct ordering |
| `test_advanced_feature_list_and_width` | feature list shape; adjacency_matrix last; cur_phase one-hot width 8 |

The 8 original CoLight feature tests still pass (no regression).

## 5. End-to-end (SUMO)

- **`len_feature` = 24** for advanced (8 cur_phase + 8 pressure + 8 running) vs 16 for colight —
  confirmed the agent is built with the advanced features (`_cal_len_feature` generalized
  automatically via `NUM_LANE_FEATURES`).
- **Topology helper verified on the real 1x1 net** (`scratchpad/probe_advanced.py`): every
  movement resolves entering lanes + a correct outgoing edge; pressure and running vectors are
  length-8 and sensible.
- **hangzhou 1x1** train+eval runs end-to-end via `--variant advanced_colight`. (At a 2-round
  budget the greedy eval coincided with CoLight's on this single intersection — an
  under-training degeneracy, not a wiring issue; both confirmed to use their distinct 16/24-dim
  features.)
- **hangzhou 4x4** (16 agents, 3-lane topology) train+evals end-to-end; the eval **differs**
  from CoLight's 4x4 eval (35 vs 33 completed, 184.3 vs 181.6s travel, 34.2 vs 36.2 queue),
  confirming the advanced features change behavior at scale.

Run:
```
runner_colight.py --variant advanced_colight --mode train_eval \
  --simulation_config dataset/sumo_version/hangzhou_4x4_gudang_18041610_1h/roadnet.sumocfg \
  --intersection_config three_lane --num_rounds 100 --simulation_steps 1200
```

## 6. Deferred
- Full-budget training comparison CoLight vs Advanced CoLight (does the richer feature set help?)
  — same multi-seed/full-budget caveat as the attention ablation.
- Parity note: source advanced features are 12-dim per-lane; this port is 8-movement, the same
  principled deviation made for CoLight's `lane_num_vehicle` (dataset-portable).
