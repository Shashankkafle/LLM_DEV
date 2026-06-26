# SUMO Controller Evaluation — Fair Comparison

## Why this exists

The LightGPT-13B LLM run on the SUMO Hangzhou 4×4 scenario looked far worse than
the LLMLight paper (ATT 390s vs 330s; 26% of vehicles never departed). The
question was whether the **controller** was bad or the **SUMO port** made the
numbers uninterpretable. The paper runs CityFlow; this repo runs SUMO, which is
microscopic (insertion refusal + spillback gridlock) where CityFlow's flow model
is not. Three things had to be fixed before any judgement was possible:

1. **Metrics measured the wrong population** — travel time counted from insertion
   over completed trips only; queue length ignored un-inserted vehicles. The 26%
   never-departed and the wait-to-insert were invisible.
2. **No in-SUMO baselines** — without FixedTime / MaxPressure on the *same*
   scenario, "bad numbers" couldn't be attributed to the controller vs. the
   scenario.
3. **Horizon boundary** — the route file injects demand up to the last second of
   the 3600s horizon, so ~16% of vehicles (1087) can't finish even at free-flow;
   completed-only ATT silently dropped them.

## What changed

- `MetricsRecorder` now emits **population-faithful metrics** from SUMO's own
  `--statistic-output`/`--tripinfo-output` (mean duration, depart delay, time
  loss, inserted/running/not-inserted/teleported), plus **CityFlow-style ATT/AWT**
  that count in-flight vehicles' time-so-far at the horizon (matching how CityFlow
  averages, so the boundary doesn't drop late departures).
- `runner_baselines.py` adds **FixedTime** and **MaxPressure** on the same
  `SumoEnv` / `PhaseHandler` / `MetricsRecorder` seam as the LLM and CoLight
  runners — identical scenario, transitions, and metrics; only the decision rule
  differs.
- `replay_runner.py` re-scores the existing LLM run from its recorded phase
  timeline (no 8-hour re-inference) under the new metrics.
- `compare_controllers.py` tabulates several runs side by side.

## Results — Hangzhou 4×4 (`anon_4_4_hangzhou_real_5816`, 3600 steps)

| Controller   | CityFlow-ATT | CityFlow-AWT | finished | not-inserted | teleports | mean-queue | eff-ATT |
|--------------|-------------:|-------------:|---------:|-------------:|----------:|-----------:|--------:|
| FixedTime    | 509.9 | 225.5 | 3241 | 2408 | 0  | 286.6 | 537.7 |
| MaxPressure  | 374.9 |  80.0 | 4013 | 1853 | 65 | 114.0 | 492.0 |
| **LLM (LightGPT-13B)** | **369.8** | **74.0** | **4070** | 1853 | 45 | 105.4 | 487.8 |

(`eff-ATT` = mean in-network duration + mean wait-to-insert, completed vehicles
only; the "full gridlock pain". `not-inserted` = scheduled but never admitted —
genuine SUMO source-edge spillback, reported separately since it has no
in-network time.)

## Interpretation

- **The LLM controller works.** It beats FixedTime decisively (370 vs 510 ATT;
  4070 vs 3241 finished) and edges MaxPressure — the **same ordering the LLMLight
  paper reports** (FixedTime ≫ MaxPressure ≈ LLM, LLM slightly ahead).
- **The gridlock is the scenario/simulator, not the controller.** FixedTime
  *also* gridlocks (2408 never inserted, queue 287), so the 26%-never-departed is
  a property of SUMO under this rush-hour ramp, not an LLM failure.
- **Absolute values run ~10–15% above CityFlow** uniformly across controllers —
  the expected SUMO-vs-CityFlow simulator gap. SUMO will not reproduce the paper's
  exact numbers by design; the *relative* ranking is what transfers.

## Reproduce

```
SC=dataset/llm_light/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.sumocfg
# baselines
python runner_baselines.py --controller fixedtime   --simulation_config $SC --intersection_config three_lane --simulation_steps 3600 --test_name fixedtime_hz4x4
python runner_baselines.py --controller maxpressure --simulation_config $SC --intersection_config three_lane --simulation_steps 3600 --test_name maxpressure_hz4x4
# re-score the existing LLM run (no re-inference)
python replay_runner.py /path/to/remote_data_new/replay_record.jsonl --no-gui
# compare
python compare_controllers.py LLM=<llm_rerun_dir> FixedTime=<ft_dir> MaxPressure=<mp_dir>
```
(`PYTHONPATH` must include the repo root and `%SUMO_HOME%\tools`.)

## Not yet done — CoLight at full load

CoLight is the RL comparator but is **not in this table**: the venv has no
TensorFlow (CoLight needs `tensorflow-cpu` + `tf-keras`), and the existing
weights are short training on 1200-step episodes. Running it fairly needs TF
installed and training at the full 3600-step load. The LLM-vs-baseline conclusion
above does not depend on it.
