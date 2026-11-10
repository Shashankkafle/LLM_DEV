# Shadow batch v4: rerun of v3 with the evaluation logs

The same 24 runs as `logs/shadow_tuning_v3`: {FixedTime, MaxPressure} × seeds {1, 2} ×
6 scenario rows, with the same args. The difference is that each run now writes the
evaluation logs from `LOGGING_CHECKLIST.md`, including the two files that cannot be
recovered from v3 by replay:

- `lane_halting.csv`, which `side_street_halted` in `network_metrics.csv` needs
- `lane_entries.csv`, which interior-lane arrival rates (blocked lanes and their feeders) need

Each run also checks its own live CSVs against an offline replay at exit and writes the
result to `run_manifest.json` under `verification`. v3 could only be checked replay against replay.

| row          | sumocfg                                          | steps | extra                                                           |
|--------------|--------------------------------------------------|-------|-----------------------------------------------------------------|
| cross_r1200  | `anon_4_4_hangzhou_real_cross_2_3_r1200.sumocfg` | 9000  |                                                                 |
| h2_steady    | `anon_4_4_hangzhou_real_h2_steady.sumocfg`       | 7200  |                                                                 |
| cross_1h     | `anon_4_4_hangzhou_real_cross_2_3_1h.sumocfg`    | 3600  |                                                                 |
| hzreal_none  | `anon_4_4_hangzhou_real.sumocfg`                 | 3600  |                                                                 |
| hzreal_c3    | `anon_4_4_hangzhou_real.sumocfg`                 | 3600  | `--blockage_scenario scenarios/c3_through_north_2_3_1200s.json` |
| hzreal_c4    | `anon_4_4_hangzhou_real.sumocfg`                 | 3600  | `--blockage_scenario scenarios/c4_two_lane_west_2_2_1200s.json` |

`tools\run_shadow.ps1` builds these 24 commands (longest runs first) and launches them.

## Cost

On the local 14-core machine at 8 in parallel, v3 took 3–18 min per run (cross_r1200 is
the longest) and about 30 min of wall-clock in total. The new per-step `lane_entries`
logging adds a little on top. Keep `-MaxParallel` at or below cores − 2; the launcher
warns if it is higher.

## Setup (PowerShell, repo root)

```
git fetch origin; git checkout logging-eval
uv sync
.venv\Scripts\Activate.ps1
sumo --version        # v3 ran on SUMO 1.27.1; use the same version (see below)
```

The launcher's commands call plain `python`, so activate the venv before launching it.
The run windows inherit that environment.

## Run

Use `-Command`, not `-File`. With `-File`, `-Seeds 1,2` arrives as the single seed 12.

```
powershell -ExecutionPolicy Bypass -Command "& tools\run_shadow.ps1 -RunGroup shadow_tuning_v4 -Seeds 1,2 -DryRun"
powershell -ExecutionPolicy Bypass -Command "& tools\run_shadow.ps1 -RunGroup shadow_tuning_v4 -Seeds 1,2 -MaxParallel 8"
```

The dry run should list 24 runs. Each run opens its own window, which stays open after
the run ends. Output also goes to `logs\shadow_tuning_v4\<test_name>.console.log`, and a
`<test_name>.done` file holds the exit code and seconds. When every run has finished, the
launcher prints a table of exit codes. Run folders land in
`logs\shadow_tuning_v4\<test_name>_<timestamp>\`.

## After the runs

1. Check that every run verified itself and see its scenario gate:

   ```
   python -c "import json,pathlib; [print(p.parent.name, (m:=json.loads(p.read_text()))['verification'], m['scenario_gate']['passed']) for p in sorted(pathlib.Path('logs/shadow_tuning_v4').glob('*/run_manifest.json'))]"
   ```

   Every run should show `{'passed': True, 'first_diff': None}`. The gate should be
   true for the cross_* rows and false for all the others.

2. Compute the metrics. This writes `eval_metrics.json` into each run and prints one line per run:

   ```
   python tools\eval_metrics.py (Get-ChildItem -Directory logs\shadow_tuning_v4).FullName
   ```

3. Copy `logs\shadow_tuning_v4\` back to the analysis machine, for example with
   `Compress-Archive logs\shadow_tuning_v4 shadow_tuning_v4.zip`. Each run folder is
   self-contained (manifest, inputs, all CSVs), so `tools\verify_flags.py` and
   `tools\eval_metrics.py` can be rerun there.

## Cross-check against v3 (optional)

On the same SUMO version, v4 traffic should match v3 bit for bit, because cfphys is
deterministic. In that case, the first 22 columns of v4 `movement_flags.csv` (the v3
columns plus `branch`) should equal the v3 files, and the replay-derived files
(`regime.csv`, `extents.csv`, `ground_truth.csv`, and `network_metrics.csv` except
`side_street_halted`) should equal `logs\shadow_tuning_v3_replay\`. If the other machine
runs a different SUMO version, expect drift on the saturated cross_* rows; the v4 runs
are still valid on their own.
