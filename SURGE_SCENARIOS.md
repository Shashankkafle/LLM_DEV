# Demand-surge scenarios

Offline route-file generation for demand surges, an extended horizon, and
per-second network- and edge-level logging. Nothing here touches
BlockageManager, the LLM prompt path, PhaseHandler or CoLight; the C3 blockage
scenario keeps producing identical results (the recorder changes were checked
on same-seed runs: every pre-existing output file is byte-identical, files and
fields are only added).

Two generations of the mechanism live side by side:

* **v1 (2026-09-24)**: clone existing base vehicles by a multiplier, one window,
  3600/7200 s horizon whose hour 2 repeats hour 1. Sections "Repo facts" to
  "Smoke-test result" below describe it; everything there still holds.
* **v2 (2026-09-29)**: a list of surges per scenario, each either a clone surge
  or an *injected* stream of new vehicles on an explicit route at an absolute
  rate (veh/h); any horizon, filled by tiling a chosen slice of hour 1; per-edge
  and per-surge recorder series; an analysis script for throughput, peak
  approach queue, oversaturation and recovery time. See "Schema v2" at the end.

Pieces:

| Piece | Where |
|---|---|
| Generator CLI | `tools/gen_surge_routes.py` |
| Family / sweep helper (v2) | `tools/make_surge_family.py` |
| Analysis (v2) | `tools/analyze_surge.py`, helpers in `utils/surge_analysis.py`, table via `tools/surge_sweep_table.py` |
| Shipped scenarios (parameters) | `dataset/llm_light/Hangzhou/4_4/scenarios/surge_*.json` (auto-registered as config aliases) |
| Generated route files + sidecars | `dataset/llm_light/Hangzhou/4_4/surge/` |
| Per-scenario sumocfgs | `dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_<name>.sumocfg` |
| Config aliases + experiment presets | `experiments.py` (`SURGE_SCENARIOS`, `surge_baselines`, `surge_family`) |
| Per-second series | `network_timeseries.csv`, `edge_timeseries.csv`, `surge_timeseries.csv` in every run dir, written by `utils/metrics_recorder.py` |
| Tests | `tests/test_surge_routes.py`, artifacts under `tests/artifacts/surge/` |

## Repo facts

Recorded 2026-09-24 before writing any code, from the files named in the task.

**Simulator and loop.** SUMO 1.27.1 (`eclipse-sumo`/`traci`/`sumolib` pinned in
`pyproject.toml`, Python 3.11). Every controller runs through
`runner_common.run_control_loop`: `for step in range(simulation_steps)`, then
`env.step()` (`traci.simulationStep()` followed by `blockage_manager.step(t)`),
then `recorder.record_step_summary(step)`, then per-intersection PhaseHandler
ticks and decisions, then early exit when `getMinExpectedNumber() <= 0`.
`SumoEnv.cmd` is `[sumo, -c, <sumocfg>] + sumo_metrics_args(output_dir)` plus
`--seed`; `sumo_metrics_args` already carries `--time-to-teleport -1`,
tripinfo/queue/statistic outputs, and the env never adds the teleport flag twice.

**Horizon under TraCI.** Verified empirically: with the base sumocfg's
`<end value="3600"/>`, a TraCI client can keep stepping past 3600 (probe reached
t=3605 with 502 vehicles still in the network and 7 pending). The `<end>` value
is therefore not what bounds a run; `--simulation_steps` is. The surge sumocfgs
still declare `<end value="7200"/>` so the files are honest when run without
TraCI.

**Aliases and identity.** `experiments.CONFIGS` maps a short alias to a sumocfg
path; `BLOCKAGES` maps an alias to a scenario JSON; `make_combo` resolves both.
`run_matrix.build_command` only forwards `--simulation_config`,
`--simulation_steps`, `--seed`, `--blockage_scenario` (plus controller flags), so
a new config alias needs no run_matrix change. Run identity
(`_identity_fields`) keys the config by `Path(simulation_config).name`, so every
new sumocfg must have a distinct basename. Steps come from the experiment preset
or `--steps`, never from the sumocfg.

**Base route file** `dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.rou.xml`
(alias `hzreal`; `hz1` is the cfphys variant): 2983 `<vehicle>` elements, ids
`0`..`2982`, integer departs 0..3599 already sorted by depart, attributes exactly
`id` and `depart`, one inline `<route edges="..."/>` child each, LF line endings,
tab indentation, inter-vehicle gap `\n\t\t`, footer `\n\t</routes>\n`. Top-level
children are only `vType` and `vehicle`. Demand is not flat: minutes 10-19 carry
64-67 departures/min against 40-57 elsewhere.

**Where the vType is defined.** Line 3 of the .rou.xml:
`<vType id="pkw" length="5.0" width="2.0" minGap="2.5" maxSpeed="11.111"
accel="2.0" decel="4.5"/>`. There is no additional file and the sumocfg
(`net-file`, `route-files`, `begin`, `end` only) defines nothing. BUT no vehicle
in the base file carries a `type` attribute, so SUMO resolves every one of them
to its built-in `DEFAULT_VEHTYPE`, not to `pkw`. Verified live through TraCI:
`getTypeID("0") == "DEFAULT_VEHTYPE"`, length 5.0, minGap 2.5, accel 2.6,
decel 4.5, tau 1.0, sigma 0.5, speedFactor drawn per vehicle (1.06 for vehicle 0).
`pkw` and `DEFAULT_VEHTYPE` agree on length 5.0 and minGap 2.5 and differ on
accel (2.0 vs 2.6). This is the "dead vType" already known from the
CityFlow reconciliation; the cfphys variant fixes it with `type="pkw"` and pinned
stochastic parameters. The generator copies vehicles verbatim, so clones resolve
exactly like their source (DEFAULT_VEHTYPE on hzreal, `pkw` on a cfphys base).

**sumocfg** `anon_4_4_hangzhou_real.sumocfg`: `net-file roadnet.net.xml`,
`route-files anon_4_4_hangzhou_real.rou.xml`, begin 0, end 3600. Paths are
relative to the sumocfg's directory; `utils/run_manifest.input_fingerprints`
resolves `net-file`/`route-files`/`additional-files` the same way, so a route
file in a subdirectory is fingerprinted correctly.

**The C3 site.** `road_2_4_3` exists in `roadnet.net.xml`: edge from
`intersection_2_4` to `intersection_2_3`, 572.8 m, three lanes, geometry running
from y=2386.4 down to y=1813.6, i.e. it enters intersection_2_3 from the north.
C3 (`scenarios/c3_through_north_2_3_1200s.json`) blocks its lane index 1
(`road_2_4_3_1`, the single N->S through lane) at 10 m from the stop line from
step 500 to 1700. 186 base vehicles route via `road_2_4_3`; 154 of them depart
inside [600, 3600].

**Blockage scenarios.** `utils/blockage_manager.load_scenario` requires
`scenario_name`, `description`, `blockages` and rejects unknown blockage keys.
A surge scenario JSON has no `blockages` key, so passing one as
`--blockage_scenario` fails at load time rather than silently doing nothing.
The obstacle vehicle id starts with `OBSTACLE_VEHICLE_PREFIX` (`obstacle_`).

**MetricsRecorder** (`utils/metrics_recorder.py`). `record_step_summary(step)`
runs once per step after `simulationStep()`, so `traci.simulation.getTime()`
inside it is `step + 1` at step length 1 s. It filters the obstacle prefix from
loaded/departed/arrived id sets, from the per-vehicle speed loop (queue and
waiting accumulation) and from `record_decision_wait`. `final_summary.json` is
assembled by `get_final_summary`: run identity, completed-trip averages
(`average_travel_time_s`, `average_waiting_time_s`), `average_queue_length`,
vehicle accounting, `average_per_decision_wait_s`, decision counters, the
CityFlow-style pair (`cityflow_style_att_s`, `cityflow_style_awt_s`),
`cityflow_clock_att_s`, then the `sumo_*` block parsed from the statistic
output (`sumo_vehicles_not_inserted` is the `waiting` attribute,
`sumo_effective_att_s` = duration + departDelay). Filenames live in
`configurations.py`.

**BLOCKAGE_PORTING_PLAN.md** relevant points: back-spill starves insertion and
pending vehicles are invisible to every lane metric (section 3.3, quote
`sumo_vehicles_not_inserted`); runs with trapped or pending vehicles never
early-stop (3.4); same-seed pairs isolate a treatment's effect because SUMO is
deterministic given the seed (3.5).

**Existing FixedTime behaviour on hzreal** (`logs/c3_campaign/ft_hzreal_none_seed1_*`):
the halted count peaks around step 1800 (301) and drains to 96 by step 3000,
so FixedTime is not saturated at base demand and recovery after a surge is
observable. A 3600 s FixedTime run takes about 300 s wall clock, so a 7200 s
run is roughly 10 min.

**Tests.** `tests/` holds plain scripts (`check()` + `failures` list, run with
`python tests/<file>.py`), not pytest suites; pytest is not in the uv lock.

## Assumptions

Decisions taken where the task left room, in the order they came up.

1. **vType identity.** The task describes the default vType as `pkw`. On the
   hzreal base every vehicle actually resolves to `DEFAULT_VEHTYPE` (see Repo
   facts). The generator does not add `type="pkw"` (that would change the base
   physics and break "copy verbatim"); the vType test asserts that every clone
   resolves in SUMO to the same (typeID, length 5.0, minGap 2.5) as the base
   vehicles, which is what "same vType as in the base file" can mean here.
2. **`road_2_4_3` is the north approach into intersection_2_3** and is used
   unchanged as the default `--via` edge; no substitution was needed.
3. **Hour-2 departs are clamped to [3600, 7199]**, not only `>= 3600`: a copy
   pushed to 7200 or later by the +30 s jitter could never be inserted inside
   the horizon and would inflate `sumo_vehicles_not_inserted`, the very
   diagnostic the smoke test reads. The upper clamp affects at most the last
   30 s of hour-1 departures.
4. **Surge clone departs are clamped to [0, horizon - 1]** for the same reason.
5. **"Hour-1 vehicle" = depart < 3600.** A base file with later departures
   would leave those vehicles uncopied (hzreal has none).
6. **The surge is applied to the extended demand** (base vehicles plus their
   hour-2 copies), so a window crossing 3600 s also surges hour-2 copies. With
   the shipped window (600-3600) nothing in hour 2 qualifies. A clone of an
   hour-2 copy is named `surge_<id>_h2_<k>`.
7. **`--hour2-scale`** is the expected number of hour-2 copies per hour-1
   vehicle: `floor(scale)` copies plus one more with probability `frac(scale)`.
   Copy 0 is `<id>_h2`, further copies `<id>_h2_<k>`; a scale below 1 thins.
8. **`--mult` below 1.0 is rejected** (base vehicles are never thinned or
   moved); `--mult 1.0` (the default) is the no-surge control.
9. **Trapezoid.** `m(t) = 1 + (mult - 1) * min(1, (t - START)/ramp, (END - t)/ramp)`
   inside `[START, END]`, 1.0 outside; so `m(START) = m(END) = 1.0`. A window
   shorter than `2 * ramp` never reaches `mult` (the generator warns). The
   default window is `600 3600` for both horizons; for a 3600 s horizon that
   leaves no drain time, so pass `--window` explicitly there.
10. **Clone count per vehicle** = `floor(m(t) - 1)` plus one more with
    probability `frac(m(t) - 1)`, drawn from `random.Random(gen_seed)` in file
    order. The RNG is consumed for every vehicle (also outside the window) so
    the draw sequence is a pure function of the seed. Neither the route file
    nor the sidecar contains timestamps, so both are byte-reproducible.
11. **`--via`** means "contains all listed edges in this order" (a subsequence,
    not necessarily contiguous); **`--od`** means first edge and last edge
    match. They are mutually exclusive and only valid with
    `--surge-type route`. For a network surge the sidecar still reports the
    "on route" counts using the default via edge, so the C3 approach's load is
    visible in every sidecar.
12. **Sidecar "base" counts are the pre-surge demand** (base file plus hour-2
    copies for a 7200 s file). The per-minute realised multiplier is
    `total / base` on the surged population (on-route vehicles for a route
    surge, everyone for a network surge), `null` where base is 0. Because
    single-minute route counts are 1-7 vehicles, the sidecar also reports the
    aggregate ratio over the whole window and over the hold
    (`realised_multiplier_window`, `realised_multiplier_hold`).
13. **Ordering.** Vehicles are stable-sorted by depart; base vehicles keep file
    order on ties. Since the base file is already sorted, `--mult 1.0
    --horizon 3600` reproduces the base file byte-for-byte.
14. **Verbatim copying is literal.** Each `<vehicle>` block is kept as raw text
    (regex-delimited, cross-checked against the XML parser's vehicle count) and
    a clone only rewrites the `id` and `depart` attributes of the opening tag.
    Named routes (`route="..."`), `<flow>`, `<trip>` and `<vTypeDistribution>`
    are unsupported and make the generator exit with an error.
15. **A third shipped scenario, `hzreal_h2` (control)**, was added next to the
    two requested ones: every surge result must be read against a same-horizon,
    same-seed control, and the smoke test's not-inserted comparison needs it.
16. **File placement.** sumocfgs sit next to the originals (so `net-file
    roadnet.net.xml` stays relative), route files and sidecars in the `surge/`
    subdirectory. The original `.sumocfg`/`.rou.xml` are untouched.
17. **An experiment preset `surge_baselines`** (FixedTime + MaxPressure, the
    three configs, seeds 1-3, steps 7200) was added so the aliases are runnable
    through `run_matrix.py` without an ad-hoc `--steps 7200`; the task only
    required the aliases.
18. **Generator seed** 20260924 for all shipped scenarios.
19. **network_timeseries.csv semantics.** `t` is the sim time after the step
    (1..N). `vehicles_in_network` counts `getIDList()` minus the obstacle
    (equals `getIDCount()` on runs without a blockage). `departed_cum` /
    `arrived_cum` accumulate the obstacle-filtered departed/arrived id lists
    (identical to accumulating `getDepartedNumber` / `getArrivedNumber` except
    for the obstacle). `halting_total` is the existing network-wide
    speed < 0.1 m/s count that feeds `average_queue_length`.
    `not_inserted_pending` is `len(getPendingVehicles())` minus the obstacle.
    `peak_vehicles_in_network_time_s` is the first time the peak was reached.
20. **Tests are pytest-style functions** but pytest is not in the uv lock, so
    they also run as a script. Run with `uv run --no-sync --with pytest -m
    pytest tests/test_surge_routes.py` or `python tests/test_surge_routes.py`.
21. **Smoke test control.** Not-inserted is compared against a FixedTime run of
    `hzreal_h2` at the same seed (1), both as SUMO's end-of-run
    `sumo_vehicles_not_inserted` and as the maximum `not_inserted_pending`
    inside the window; a rise above 10% on either prints the throttling warning
    (it does not fail the test). The recovery assertion is exactly the one
    specified (occupancy at t=7199 below t=3600 in the surge run); note that
    the hour-2 demand tail is a jittered copy of the hour-1 tail, so the control
    run's values at both times are reported alongside for context.
22. **METRICS.md** gained a short entry for the two new summary fields and the
    CSV, since it is the canonical field reference.

## How to generate

From the repo root. The shipped scenarios regenerate from their JSON (explicit
flags override the JSON):

```
python tools/gen_surge_routes.py --scenario dataset/llm_light/Hangzhou/4_4/scenarios/surge_h2.json
python tools/gen_surge_routes.py --scenario dataset/llm_light/Hangzhou/4_4/scenarios/surge_s1_net150.json
python tools/gen_surge_routes.py --scenario dataset/llm_light/Hangzhou/4_4/scenarios/surge_s2_route150_2_3.json
```

Equivalent explicit commands:

```
python tools/gen_surge_routes.py --base dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.rou.xml \
    --out dataset/llm_light/Hangzhou/4_4/surge/anon_4_4_hangzhou_real_h2.rou.xml \
    --horizon 7200 --mult 1.0 --gen-seed 20260924

python tools/gen_surge_routes.py --base dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.rou.xml \
    --out dataset/llm_light/Hangzhou/4_4/surge/anon_4_4_hangzhou_real_s1_net150.rou.xml \
    --horizon 7200 --surge-type network --mult 1.5 --window 600 3600 --ramp 300 --gen-seed 20260924

python tools/gen_surge_routes.py --base dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real.rou.xml \
    --out dataset/llm_light/Hangzhou/4_4/surge/anon_4_4_hangzhou_real_s2_route150_2_3.rou.xml \
    --horizon 7200 --surge-type route --via road_2_4_3 --mult 1.5 --window 600 3600 --ramp 300 --gen-seed 20260924
```

Each run writes `<out>` and `<out minus .rou.xml>.surge.json` and prints one
summary line, e.g.

```
wrote .../anon_4_4_hangzhou_real_s2_route150_2_3.rou.xml: 2983 base + 2983 hour-2 + 69 surge = 6035 vehicles | route surge x1.5 over 600-3600 s (ramp 300 s) | realised x1.514 on the hold, x1.448 over the window | sidecar ...
```

Flags: `--base`, `--out`, `--horizon {3600,7200}`, `--hour2-scale`,
`--surge-type {network,route}`, `--window START END`, `--ramp`, `--mult`,
`--od FROM TO` | `--via EDGE [EDGE ...]`, `--gen-seed`, `--scenario`.

To run a scenario:

```
python run_matrix.py --experiment surge_baselines --dry_run
python run_matrix.py --experiment surge_baselines --controllers fixedtime --seeds 1
python runner_baselines.py --controller fixedtime --simulation_steps 7200 \
    --simulation_config dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_s2_route150_2_3.sumocfg --seed 1
```

Always pass 7200 steps for these configs; the sumocfg's `<end>` does not bound a
TraCI run.

To run the tests:

```
uv run --no-sync --with pytest -m pytest tests/test_surge_routes.py -k "not smoke"   # ~25 s
uv run --no-sync --with pytest -m pytest tests/test_surge_routes.py -k smoke -s      # ~20 min, SUMO only
python tests/test_surge_routes.py --skip-smoke                                        # without pytest
```

## Scenario JSON schema

`dataset/llm_light/Hangzhou/4_4/scenarios/surge_<name>.json`:

| Key | Type | Meaning |
|---|---|---|
| `scenario_name` | string | Equals the config alias in `experiments.CONFIGS`. |
| `kind` | `"demand_surge"` | Distinguishes the file from blockage scenarios (which `load_scenario` would reject anyway: no `blockages` key). |
| `description` | string | Human summary of the profile and the control arm. |
| `config_alias` | string | Alias to pass via `--configs`. |
| `sumocfg` | path | The per-scenario sumocfg (repo-relative). |
| `route_file` | path | The generated `.rou.xml` the sumocfg references. |
| `sidecar` | path | The generator's `.surge.json` (counts, realised multiplier). |
| `simulation_steps` | int | Steps to run (7200). Not read by any runner; documents the intended `--steps`. |
| `generator` | object | Exactly the generator's flags, by argparse dest name: `base`, `out`, `horizon`, `hour2_scale`, `surge_type`, `window` (`[start, end]`), `ramp`, `mult`, `od` (`null` or `[from, to]`), `via` (`null` or list), `gen_seed`. `--scenario` loads these as defaults. |

Sidecar (`<name>.surge.json`) keys: `generator`, `parameters` (the same flag
set), `route_filter` (`{kind: via|od, edges}`), `surge_population`
(`route`|`all`), `base_vehicle_count`, `hour2_added_count`,
`surge_added_count`, `total_vehicle_count`, `realised_multiplier_window`,
`realised_multiplier_hold`, `vtypes_defined` (the base file's `<vType>`
attributes), `resolved_vehicle_types` (`{base, output}` histograms of the type
each vehicle resolves to), `per_minute` (`minute`, `base_all`, `total_all`,
`base_route`, `total_route`, `expected_multiplier`, `realised_multiplier`).

## Alias naming rule

`<base alias>_s<n>_<net|route><mult x 100>[_<x>_<y>]`, and `<base alias>_h2`
for the extended-horizon control:

* `hzreal_h2`: hzreal demand, 7200 s, hour 2 = jittered copy of hour 1, no surge.
* `hzreal_s1_net150`: network-wide surge, multiplier 1.5.
* `hzreal_s2_route150_2_3`: route surge, multiplier 1.5, on routes entering
  intersection `2_3` via the default via edge (`road_2_4_3`). If a route surge
  used a different approach or an OD pair, the suffix names the intersection or
  the pair instead.

A new alias needs: the scenario JSON, the generated route file (run the
generator), a sumocfg with a distinct basename, and one line in
`experiments.CONFIGS`. Nothing in `run_matrix.py` changes.

## Realised per-minute multiplier

From the shipped sidecars (`gen-seed 20260924`). Aggregates first:

| scenario | vehicles (base + hour-2 + surge) | realised over window 600-3600 s | realised over hold 900-3300 s |
|---|---|---|---|
| S1 network x1.5 | 7051 (2983 + 2983 + 1085) | x1.439 | x1.485 |
| S2 route x1.5 via road_2_4_3 | 6035 (2983 + 2983 + 69) | x1.448 | x1.514 |
| H2 control | 5966 (2983 + 2983 + 0) | x1.0 | x1.0 |

Per minute (minutes 0-7 and 63-119 are x1.00 in every scenario; hour-2 copies
make minutes 60-119 mirror 0-59 with jitter). "S1 base / total" counts every
vehicle; "S2 base / total" counts only vehicles routed via `road_2_4_3`, so its
per-minute ratio is noisy (1-7 base vehicles per minute) and the hold aggregate
above is the number to quote.

| minute | window | expected m(t) | S1 base / total (all) | S1 realised | S2 base / total (via road_2_4_3) | S2 realised |
|---|---|---|---|---|---|---|
| 8 (480-539 s) | outside | 1.00 | 57 / 57 | x1.00 | 4 / 4 | x1.00 |
| 9 (540-599 s) | outside | 1.00 | 52 / 52 | x1.00 | 3 / 3 | x1.00 |
| 10 (600-659 s) | ramp up | 1.05 | 67 / 68 | x1.01 | 4 / 4 | x1.00 |
| 11 (660-719 s) | ramp up | 1.15 | 64 / 70 | x1.09 | 5 / 6 | x1.20 |
| 12 (720-779 s) | ramp up | 1.25 | 65 / 83 | x1.28 | 6 / 7 | x1.17 |
| 13 (780-839 s) | ramp up | 1.35 | 65 / 88 | x1.35 | 3 / 4 | x1.33 |
| 14 (840-899 s) | ramp up | 1.45 | 65 / 104 | x1.60 | 7 / 11 | x1.57 |
| 15 (900-959 s) | hold | 1.50 | 67 / 104 | x1.55 | 3 / 6 | x2.00 |
| 16 (960-1019 s) | hold | 1.50 | 67 / 99 | x1.48 | 7 / 9 | x1.29 |
| 17 (1020-1079 s) | hold | 1.50 | 65 / 99 | x1.52 | 5 / 9 | x1.80 |
| 18 (1080-1139 s) | hold | 1.50 | 67 / 100 | x1.49 | 1 / 1 | x1.00 |
| 19 (1140-1199 s) | hold | 1.50 | 67 / 97 | x1.45 | 2 / 3 | x1.50 |
| 20 (1200-1259 s) | hold | 1.50 | 48 / 74 | x1.54 | 3 / 5 | x1.67 |
| 21 (1260-1319 s) | hold | 1.50 | 48 / 71 | x1.48 | 3 / 6 | x2.00 |
| 22 (1320-1379 s) | hold | 1.50 | 44 / 64 | x1.46 | 1 / 2 | x2.00 |
| 23 (1380-1439 s) | hold | 1.50 | 46 / 75 | x1.63 | 2 / 3 | x1.50 |
| 24 (1440-1499 s) | hold | 1.50 | 44 / 68 | x1.54 | 1 / 1 | x1.00 |
| 25 (1500-1559 s) | hold | 1.50 | 53 / 75 | x1.42 | 3 / 6 | x2.00 |
| 26 (1560-1619 s) | hold | 1.50 | 52 / 73 | x1.40 | 6 / 12 | x2.00 |
| 27 (1620-1679 s) | hold | 1.50 | 49 / 74 | x1.51 | 4 / 5 | x1.25 |
| 28 (1680-1739 s) | hold | 1.50 | 52 / 74 | x1.42 | 4 / 6 | x1.50 |
| 29 (1740-1799 s) | hold | 1.50 | 52 / 75 | x1.44 | 3 / 5 | x1.67 |
| 30 (1800-1859 s) | hold | 1.50 | 46 / 67 | x1.46 | 3 / 4 | x1.33 |
| 31 (1860-1919 s) | hold | 1.50 | 46 / 64 | x1.39 | 5 / 5 | x1.00 |
| 32 (1920-1979 s) | hold | 1.50 | 44 / 67 | x1.52 | 1 / 2 | x2.00 |
| 33 (1980-2039 s) | hold | 1.50 | 46 / 62 | x1.35 | 3 / 4 | x1.33 |
| 34 (2040-2099 s) | hold | 1.50 | 46 / 68 | x1.48 | 5 / 6 | x1.20 |
| 35 (2100-2159 s) | hold | 1.50 | 46 / 66 | x1.44 | 1 / 1 | x1.00 |
| 36 (2160-2219 s) | hold | 1.50 | 43 / 73 | x1.70 | 1 / 2 | x2.00 |
| 37 (2220-2279 s) | hold | 1.50 | 40 / 55 | x1.38 | 2 / 4 | x2.00 |
| 38 (2280-2339 s) | hold | 1.50 | 45 / 65 | x1.44 | 1 / 2 | x2.00 |
| 39 (2340-2399 s) | hold | 1.50 | 43 / 59 | x1.37 | 3 / 5 | x1.67 |
| 40 (2400-2459 s) | hold | 1.50 | 49 / 70 | x1.43 | 3 / 4 | x1.33 |
| 41 (2460-2519 s) | hold | 1.50 | 49 / 76 | x1.55 | 4 / 5 | x1.25 |
| 42 (2520-2579 s) | hold | 1.50 | 47 / 69 | x1.47 | 1 / 1 | x1.00 |
| 43 (2580-2639 s) | hold | 1.50 | 47 / 68 | x1.45 | 2 / 2 | x1.00 |
| 44 (2640-2699 s) | hold | 1.50 | 49 / 75 | x1.53 | 2 / 3 | x1.50 |
| 45 (2700-2759 s) | hold | 1.50 | 44 / 59 | x1.34 | 2 / 4 | x2.00 |
| 46 (2760-2819 s) | hold | 1.50 | 42 / 63 | x1.50 | 1 / 1 | x1.00 |
| 47 (2820-2879 s) | hold | 1.50 | 41 / 61 | x1.49 | 4 / 6 | x1.50 |
| 48 (2880-2939 s) | hold | 1.50 | 42 / 62 | x1.48 | 2 / 3 | x1.50 |
| 49 (2940-2999 s) | hold | 1.50 | 43 / 70 | x1.63 | 4 / 4 | x1.00 |
| 50 (3000-3059 s) | hold | 1.50 | 43 / 70 | x1.63 | 2 / 4 | x2.00 |
| 51 (3060-3119 s) | hold | 1.50 | 43 / 67 | x1.56 | 5 / 7 | x1.40 |
| 52 (3120-3179 s) | hold | 1.50 | 41 / 64 | x1.56 | 2 / 3 | x1.50 |
| 53 (3180-3239 s) | hold | 1.50 | 43 / 67 | x1.56 | 1 / 1 | x1.00 |
| 54 (3240-3299 s) | hold | 1.50 | 42 / 58 | x1.38 | 1 / 3 | x3.00 |
| 55 (3300-3359 s) | ramp down | 1.45 | 43 / 62 | x1.44 | 5 / 7 | x1.40 |
| 56 (3360-3419 s) | ramp down | 1.35 | 43 / 62 | x1.44 | 3 / 5 | x1.67 |
| 57 (3420-3479 s) | ramp down | 1.25 | 41 / 53 | x1.29 | 6 / 7 | x1.17 |
| 58 (3480-3539 s) | ramp down | 1.15 | 42 / 52 | x1.24 | 3 / 4 | x1.33 |
| 59 (3540-3599 s) | ramp down | 1.05 | 43 / 45 | x1.05 | 3 / 3 | x1.00 |
| 60 (3600-3659 s) | outside | 1.00 | 53 / 53 | x1.00 | 4 / 4 | x1.00 |
| 61 (3660-3719 s) | outside | 1.00 | 45 / 45 | x1.00 | 3 / 3 | x1.00 |
| 62 (3720-3779 s) | outside | 1.00 | 50 / 50 | x1.00 | 3 / 3 | x1.00 |

## Smoke-test result (FixedTime, seed 1, 7200 s), 2026-09-24

`test_smoke` runs FixedTime on `hzreal_s2_route150_2_3` and on the control
`hzreal_h2` (seed 1, 7200 steps, about 12 min each). Numbers from
`tests/artifacts/surge/test_smoke/smoke_report.json` and the two
`network_timeseries.csv` files:

| quantity | S2 route surge | H2 control |
|---|---|---|
| timeseries rows | 7200 | 7200 |
| `peak_vehicles_in_network` (time) | 651 (t=4781) | 647 (t=4790) |
| peak inside the window 600-3600 s | 588 | 581 |
| `vehicles_in_network` at t=3600 | 477 | 471 |
| `vehicles_in_network` at t=7199 | 533 | 508 |
| max `not_inserted_pending` inside the window | 55 | 18 |
| `sumo_vehicles_not_inserted` (end of run) | 167 | 126 |
| arrived / completion_rate | 5333 / 0.884 | 5330 / 0.893 |
| `cityflow_style_att_s` | 612.5 | 606.2 |

Occupancy every 600 s (surge / control): 600 s 371/371, 1200 s 581/573,
1800 s 509/494, 2400 s 486/485, 3000 s 511/495, 3600 s 477/471, 4200 s 541/536,
4800 s 646/643, 5400 s 562/569, 6000 s 517/518, 6600 s 535/537, 7200 s 535/510.

Outcome of the three assertions as specified:

1. 7200 rows: **passes**.
2. Peak inside the surge window: **fails** (t=4781). The control peaks at the
   same place (t=4790), so the peak is the hour-2 copy of the base demand bump
   (minutes 70-79 mirror minutes 10-19, 64-67 departures/min) landing on a
   network that still holds 471 vehicles from hour 1. The route surge adds 69
   vehicles over 50 minutes (+1.4/min network-wide) and moves the peak by +4.
3. Occupancy at t=7199 below t=3600: **fails** (533 vs 477), and the control
   fails it too (508 vs 471). The hour-2 tail carries the same demand as the
   hour-1 tail, so under FixedTime the network ends hour 2 more loaded than it
   ended hour 1. The surge's excess over the control does not decay either:
   the 69 extra vehicles end up as +3 arrived, +25 still in the network and
   +41 never inserted.

Not-inserted versus the control: **rises by more than 10%** (end of run +32.5%,
window maximum +206%), so the test prints the throttling warning: the extra
vehicles on the `road_2_4_3` routes queue at their entry edges instead of
entering, and the route multiplier is only partly realised in the network.
(In the base 3600 s FixedTime run only 4 vehicles were never inserted; the
control's 126 shows hour 2 alone already saturates the entries under
FixedTime.)

What this means for using the scenarios: with FixedTime, a network-level peak
or a "t=7199 below t=3600" criterion cannot isolate a surge as small as S2; the
control must be subtracted (the report carries the control values and a
smoothed surge-minus-control excess for that), and a recovery window needs
hour 2 lighter than hour 1 (`--hour2-scale`, e.g. 0.7) or a window that ends
well before 3600 s. The assertions are left exactly as specified so the test
keeps reporting this until the scenario or the criterion is changed.

Additional diagnostics run outside the test (seed 1, 7200 steps, runs kept
in the session scratchpad, not in the repo):

| run | peak (time) | t=3600 | t=7199 | not inserted (end) | max pending in window | smoothed excess over its control: peak (time), at end |
|---|---|---|---|---|---|---|
| FixedTime, H2 control | 647 (t=4790) | 471 | 508 | 126 | 18 | - |
| FixedTime, S2 route x1.5 | 651 (t=4781) | 477 | 533 | 167 | 55 | +24 (t=7200), +24 |
| FixedTime, S1 network x1.5 | 770 (t=1334) | 628 | 549 | 377 | 256 | +231 (t=2460), +47 |
| MaxPressure, H2 control | 480 (t=4700) | 304 | 292 | 0 | 1 | - |
| MaxPressure, S2 route x1.5 | 481 (t=1153) | 315 | 302 | 0 | 2 | +28 (t=1896), 0 |

Reading: the network surge S1 satisfies both dynamic assertions even under
FixedTime (peak inside the window, t=7199 below t=3600), but 256 vehicles were
pending insertion at its worst point against 18 in the control, so a good part
of the x1.5 never entered the network under FixedTime. Under MaxPressure the
S2 route surge satisfies all three assertions with nothing throttled: its
excess occupancy over the control rises to about 28 vehicles around t=1900
and is back to zero by the end of the run, the surge-then-recovery shape the
smoke test is looking for. The specified FixedTime + S2 combination is the one
pairing where the controller cannot drain hour 2 and the surge is too small to
dominate the network-level peak.

## Schema v2 (2026-09-29): injected surges, any horizon, oversaturation probe

Motivation (user decisions 2026-09-29): specify surge routes explicitly, run
two surges at once that cross at one intersection, use any horizon, keep the
onset fixed (no controller observes time: CoLight's 16-dim observation and the
LLM prompt carry no step or clock), and find the rate at which an approach
oversaturates before sweeping the surge duration.

### Scenario JSON

A v2 file has `surges` instead of `generator`; v1 files stay valid (the loader
turns a `generator` block into one clone surge with the default tiling).

```json
{
  "scenario_name": "hzreal_cross_2_3", "kind": "demand_surge", "schema_version": 2,
  "config_alias": "hzreal_cross_2_3", "control_alias": "hzreal_h2_steady",
  "sumocfg": ".../anon_4_4_hangzhou_real_cross_2_3.sumocfg",
  "route_file": ".../surge/anon_4_4_hangzhou_real_cross_2_3.rou.xml",
  "sidecar":    ".../surge/anon_4_4_hangzhou_real_cross_2_3.surge.json",
  "base": ".../anon_4_4_hangzhou_real.rou.xml", "net": ".../roadnet.net.xml",
  "horizon": 9000, "simulation_steps": 9000,
  "tiling": {"source": [1200, 3600], "scale": 1.0},
  "gen_seed": 20260924, "intersection": "intersection_2_3",
  "surges": [
    {"name": "we3", "mode": "inject", "route": ["road_0_3_0", "road_1_3_0", "road_2_3_0", "road_3_3_0", "road_4_3_0"],
     "rate_veh_h": 600, "window": [3600, 4800], "ramp": 0, "arrivals": "uniform"},
    {"name": "ns2", "mode": "inject", "od": ["road_2_5_3", "road_2_1_3"],
     "rate_veh_h": 600, "window": [3600, 4800], "ramp": 0, "arrivals": "uniform"}
  ]
}
```

Per surge: `name` (`[A-Za-z][A-Za-z0-9-]*`, unique), `mode` (`inject` |
`clone`), `window`, `ramp` (0 = step). Inject: `route` (edge list, validated
edge by edge against the net) xor `od` (resolved by sumolib's shortest path
for `passenger`), `rate_veh_h`, `arrivals` (`uniform` default | `poisson`),
optional `depart_attrs`, optional per-surge `intersection`. Clone: `mult`,
`scope` (`network` | `route`), `via` | `od`. `control_alias` names the
same-seed control every run of this scenario is read against; `experiments`
and `build_results` use it. Top-level `intersection` is the crossing point
the analysis reports on; each inject surge's `approach_edge` into it is
resolved at generation time and stored in the sidecar.

### What the generator does

1. Read the base file (vehicles copied verbatim as before).
2. **Tiling**: for a horizon past 3600 s, the base vehicles departing inside
   `tiling.source` `[S, E)` are repeated tile by tile (`+3600`, `+3600+L`, ...,
   `L = E - S`), ids `<id>_h2`, `<id>_h3`, ..., each with +-30 s jitter clamped
   into its tile; a partial last tile keeps only the departs that fit.
   `tiling.scale` is the old `--hour2-scale`. The default source `[0, 3600)`
   with a 7200 s horizon reproduces the v1 hour-2 copy byte for byte (the
   shared RNG is consumed in exactly the old order); the steady source
   `[1200, 3600)` (41-49 departures/min, no bump) is what the v2 scenarios
   use so surge and recovery sit on a flat baseline.
3. **Surges**, in list order. A clone surge draws from the shared RNG as in
   v1 (a v1-style unnamed clone surge keeps ids `surge_<baseid>_<k>`; a named
   one numbers its clones `surge_<name>_<k>`). An inject surge writes NEW
   vehicles `surge_<name>_<k>` with the attribute set and block layout of the
   base file's first vehicle (no `type` on hzreal, so they resolve to
   DEFAULT_VEHTYPE like every base vehicle; `type="pkw" departLane="best"
   departSpeed="max"` on a cfphys base). `uniform` arrivals use an integer
   accumulator on `rate_veh_h/3600 * shape(t)`, so the added load is exactly
   the nominal rate (600 veh/h over 1200 s = 200 vehicles) and the throughput
   knee of a rate sweep is attributable to capacity, not to arrival
   burstiness; `poisson` draws exponential gaps at the peak rate, thinned by
   the trapezoid, from a per-surge RNG seeded `"<gen_seed>/<name>"` so one
   surge's departs never move when another surge changes.
4. Stable sort by depart, write the route file, the sidecar and (with
   `--write-sumocfg`) the per-scenario sumocfg with `<end value="horizon">`.

Sidecar v2 keeps every v1 key (the legacy multiplier fields describe the
first clone surge, or the v1 defaults) and adds `schema_version`, `tiling`
(`source`, `scale`, `tile_count`, `added_count`), `surges` (per surge:
route, `entry_edge`, `approach_edge`, `intersection`, `added_count`,
`realised_rate_hold_veh_h`, `per_minute_added`) and `sumolib_version`.

CLI additions: `--horizon` any int, `--tile-source START END`, `--net`,
`--write-sumocfg`. Multi-surge scenarios are JSON-only (`--scenario`, with
`--out` as the only override). sumolib is imported only when an inject surge
or an `od` exists; it resolves from SUMO_HOME (1.27.1 here).

### Shipped v2 scenarios

| alias | what | control |
|---|---|---|
| `hzreal_h2_steady` | hour 1 as recorded, then the steady slice tiled to 9000 s, no surge (2983 base + 4108 tiled) | - |
| `hzreal_cross_2_3` | template: `we3` W->E along row 3 (approach `road_1_3_0`) and `ns2` N->S along column 2 (approach `road_2_4_3`, the C3 approach), both 600 veh/h step over 3600-4800 s | `hzreal_h2_steady` |
| `hzreal_cross_2_3_r{300,600,900,1200}` | the rate family derived from the template | `hzreal_h2_steady` |

Neither entry edge (`road_0_3_0`, `road_2_5_3`) carries base demand; the
straight movement at every intersection uses a single lane (`fromLane="1"`),
so per-approach through capacity is one lane times green share.

```
python tools/make_surge_family.py --template dataset/llm_light/Hangzhou/4_4/scenarios/surge_cross_2_3.json --sweep rate 300 600 900 1200
python tools/make_surge_family.py --template ... --sweep hold 600 1200 1800 2700 [--surges ns2]
python run_matrix.py --experiment surge_family --configs hzreal_h2_steady hzreal_cross_2_3_r600 ... --seeds 1
python tools/analyze_surge.py --run logs/surge_family/ft_hzreal_cross_2_3_r600_none_seed1_<ts> --control logs/surge_family/ft_hzreal_h2_steady_none_seed1_<ts>
python tools/surge_sweep_table.py --analyses "logs/surge_family/**/surge_analysis.json"
python tools/analyze_surge.py estimate --capacity C --base-rate d0 --surge-rate d1 --duration T
python build_results.py --experiment surge_family
```

Every `scenarios/surge_*.json` with `kind: demand_surge` is discovered by
`experiments.SURGE_SCENARIOS` and registered in `CONFIGS` under its
`config_alias`, so a generated scenario is runnable without editing
`experiments.py`; the `surge_family` preset takes `steps: "from_scenario"`.
Alias naming: `<template alias>_r<rate>` / `_hold<seconds>`.

### Recorder and analysis

`edge_timeseries.csv` (per-edge vehicles and halting, every step, from the
same loop as `halting_total`), `surge_timeseries.csv` (per injected surge:
`departed_cum`, `arrived_cum`, `pending`) and `surge_vehicles` in
`final_summary.json` are described in METRICS.md. `build_results.py` now keys
configs by alias (surge variants no longer pool into one `hangzhou_4_4/real`
row) and adds seed-paired `delta_att_vs_control` /
`delta_not_inserted_vs_control` against each scenario's `control_alias`.

`tools/analyze_surge.py` writes `surge_analysis.json` into the run dir with,
per surge: stream departed / arrived by window end / arrived by run end,
peak approach halting inside the window (and the control's on the same
edge), an **oversaturation flag** (the 300 s-smoothed approach queue, or the
surge's own insertion backlog, still growing by more than 2 vehicles over the
last third of the hold), **recovery time** (seconds after the window end
until the smoothed approach queue is within 10% + 1 vehicle of the control's
and stays there for 300 s; `null` if never), and the surge's max pending;
plus network-level throughput in the window, smoothed excess occupancy over
the control and max pending. `estimate` prints the deterministic queueing
numbers (`Q = (d1 - C) T`, `t_clear = Q / (C - d0)`, recovery window
`2 t_clear`) for sizing a horizon once a capacity is measured.

Horizon guidance: warm-up + baseline (hour 1 as recorded, its minutes-10-19
bump included) -> surge (hour 2, fixed onset) -> recovery at least twice the
queueing estimate. Insertion caveat for high rates: base vehicles carry no
`departLane`, so SUMO inserts on lane 0; the entry edge can then be the
bottleneck rather than the intersection, which `pending` per surge exposes
and `depart_attrs: {"departLane": "best"}` works around.
