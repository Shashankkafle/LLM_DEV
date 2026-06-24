# Next steps — forward plan

Roadmap after the CoLight baseline (committed `246bcd4` on branch `colight`). Two forks:
(A) finish the attention-ablation characterization, (B) implement Advanced CoLight on a
separate branch. Each fork below becomes its own commit so individual points are
roll-back-able.

---

## A. Attention-ablation ladder (current fork, on `colight`)

The deterministic wiring proof already PASSES (neighbor info provably affects decisions —
VALIDATION.md §8). Open question is only the *performance* benefit, which at 25 rounds /
1 seed did not appear. A 60-round single-seed comparison is running. Plan **(c)** — climb
the ladder regardless of the 60-round outcome, stopping early if a rung clearly shows the
benefit; if none does, that is a legitimate, documented null result.

| Rung | Tests | Note |
|---|---|---|
| **A1. 3 seeds @ 60 rounds** | single-seed noise (biggest confound) | run with-attn vs ablated for seeds {42, 1, 7}; report mean±std of eval queue/travel |
| **A2. Full budget: 100 rounds, 3600-step episodes** | undertraining / short episodes (source's actual setting) | longer; the attention input is higher-info-but-noisier and needs more training to exploit |
| **A3. Different 4x4 net** | this Gudang traffic may not reward coordination | try `syn_4x4_gaussian_500_1h` (uniform) and/or `manhattan_16x3` (corridors) |
| **A4. Accept + document** | — | if A1–A3 show no benefit: honest finding that attention doesn't beat isolated control here at these budgets; baseline still valid (attention is wired) |

Outcome handling:
- **If any rung shows attention helping** → record numbers, flip performance check to PASS,
  note the budget/dataset threshold, commit.
- **If none do** → document the null result with the full ladder evidence, commit. Does NOT
  block moving on (the engineering is correct and validated).

Then: **thoroughly run/test CoLight** (longer training run, sanity on eval metrics, replay)
and commit before forking to Advanced CoLight.

---

## B. Advanced CoLight (NEW branch `advanced_colight`, branched from `colight`)

**Gate:** start only after CoLight tests pass (they do) AND the ablation/CoLight-testing
above is wrapped. All Advanced-CoLight changes live on `advanced_colight`, NOT `colight`.

**Key insight — the agent is already done.** In source, `"AdvancedColight": CoLightAgent`
— it reuses the *same* graph-attention agent I already lifted. Advanced CoLight differs
ONLY in its state features. So this is almost entirely a `state_features.py` addition, which
is exactly what the feature-registry seam was designed for.

Source spec (`run_advanced_colight.py`, pinned `d5d4180`):
```
LIST_STATE_FEATURE = ["cur_phase",
                      "traffic_movement_pressure_queue_efficient",
                      "lane_enter_running_part",
                      "adjacency_matrix"]
CNN_layers = [[32, 32]]            # same as CoLight
agent = CoLightAgent               # same as CoLight
```

### What to lift vs reuse vs new (mirrors the CoLight approach)
| Component | Disposition |
|---|---|
| `CoLightAgent` | **REUSE as-is** (no new agent; the lifted file is unchanged) |
| `runner_colight.py` loop | **REUSE**, parameterized by a feature-set/variant (see below) |
| `SumoEnv`, `PhaseHandler`, `MetricsRecorder`, adjacency, PHASE one-hot | **REUSE unchanged** |
| 2 new feature functions in `utils/state_features.py` | **NEW (additive)** |
| `ADVANCED_COLIGHT_AGENT_CONF` + feature list in `configurations.py` | **NEW (additive)** |
| Runner variant selection | **NEW (small)** |

### The two new features (8-movement representation, like CoLight)
1. **`traffic_movement_pressure_queue_efficient`** — source
   `_get_traffic_movement_pressure_efficient`: per entering movement `m`,
   `pressure[m] = entering_queue[m] - (exiting_queue on m's outgoing road) / 3`.
   - Entering queue per movement: already available from `get_state` (movement early_queued).
   - Outgoing road per movement: `MOVEMENT_OUTGOING_ROAD` (already in configurations.py).
   - **#1 contract detail / new-data risk:** the *exiting* (downstream) queue is NOT in
     `get_state` (which only covers controlled/incoming lanes). Need a small helper to query
     the outgoing road's halting count. Resolve: map each movement → its outgoing edge via the
     env's `dic_exiting_approach`/road topology, sum `getLastStepHaltingNumber` over those lanes.
   - Output: length-8 vector over `COLIGHT_MOVEMENT_ORDER`.
2. **`lane_enter_running_part`** — source: running (moving) vehicles in the lane segment
   nearest the stopline, per entering movement (= near-stopline count minus its queued part).
   - Already derivable from `get_state`'s per-movement `segment_1` (nearest segment, moving
     vehicles only — `get_state` already excludes stopped vehicles from segment counts).
   - Decide: use `segment_1` directly, or a fixed near-stopline window matching source's
     `obs_length`. Document the choice. Output: length-8 vector.

### Feature width
`_cal_len_feature` already generalizes: `cur_phase`(8) + pressure(8) + running(8) = **24**
(it adds `NUM_LANE_FEATURES=8` per non-phase/non-adjacency feature). No agent change needed.

### Runner wiring (keep additive)
Add a variant registry: `{"colight": (FEATURES_COLIGHT, COLIGHT_AGENT_CONF),
"advanced_colight": (FEATURES_ADV, ADVANCED_COLIGHT_AGENT_CONF)}` and a `--variant` flag on
`runner_colight.py` (default `colight`). The loop, cadence, reward, memory, and eval are
identical — only the feature list + conf differ. The stored-transition `expand_state_for_memory`
still only touches `cur_phase`, so it is variant-agnostic.

### Validation (mirror CoLight; on `advanced_colight`)
- Per-feature unit tests for the 2 new features (hand-built SUMO state → expected 8-vector,
  including the outgoing-queue path).
- Parity table vs source advanced conf.
- End-to-end train+eval on hangzhou 1x1 and 4x4 through MetricsRecorder.
- Reuse the existing adjacency/attention validation (shared, already proven).

### Workflow
Plan → STOP for approval (no code) → implement features + config + runner variant + tests on
hangzhou 1x1 → STOP → extend to 4x4 → STOP. Commit at each step for rollback.

---

## Commit discipline (per user request)
- Each fork / milestone = its own commit on its branch, so any point is roll-back-able.
- Advanced CoLight strictly on `advanced_colight`; nothing advanced-specific on `colight`.
- Pre-existing non-CoLight working-tree changes (e.g. `runner.py` mock-LLM toggle) are NOT
  bundled into these commits.
