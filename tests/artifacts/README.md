# Test artifacts

Everything under this directory except this README is written by tests and
git-ignored. Each test wipes and recreates its own subfolder on every run, so
what you see is always the last run's output.

## surge/  (tests/test_surge_routes.py)

| Subfolder | Contents |
|---|---|
| `test_parity/` | `parity.rou.xml` + `parity.surge.json`: `--mult 1.0 --horizon 3600`, must equal the base file. |
| `test_extension/` | `h2.rou.xml` + sidecar: `--horizon 7200 --mult 1.0`, base demand plus its hour-2 copy. |
| `test_route_filter/` | `s2.rou.xml` + sidecar: route surge x1.5 through `road_2_4_3`. |
| `test_determinism/` | `run_a.rou.xml`, `run_b.rou.xml` (same `--gen-seed`, must be byte-identical) and `run_c.rou.xml` (different seed). |
| `test_vtype/` | `s2.rou.xml` + sidecar, plus `vtype_resolution.json`: the vType every sampled vehicle resolved to in a live SUMO session. |
| `test_smoke/` | `surge/` and `control/`: the `.sumocfg` each FixedTime 7200 s run used, its `network_timeseries.csv` and `final_summary.json`, the full run directory under `runs/`, and `smoke_report.json` with the headline numbers (peak, t=3600 / t=7199 occupancy, not-inserted vs control). |
