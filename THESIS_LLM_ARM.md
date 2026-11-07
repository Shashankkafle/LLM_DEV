# Thesis LLM arm: paper 2 prompt, no incident report, demand surges

Row `llm_zeroshot_noreport_paper2prompt`: the paper 2 model and prompt on three
demand-surge scenarios of the Hangzhou 4x4, seeds 1 and 2, horizon 7200 s. There is
no blockage, so the incident-report section never appears. The system prompt
(including the generic blockage-guidance block), the prompt builder, the parser and
fallback rules, `max_new_tokens` 1024 and greedy decoding all match paper 2. The
prompt hash `553b90b8...` is the same at the paper 2 commit and here.

| Run order | Scenario    | sumocfg                                          | Seed |
|-----------|-------------|--------------------------------------------------|------|
| 1         | cross_r1200 | `anon_4_4_hangzhou_real_cross_2_3_r1200.sumocfg` | 1    |
| 2         | cross_1h    | `anon_4_4_hangzhou_real_cross_2_3_1h.sumocfg`    | 1    |
| 3         | hzreal_none | `anon_4_4_hangzhou_real.sumocfg`                 | 1    |
| 4-6       | same three  |                                                  | 2    |

## How this differs from paper 2

Paper 2 (`logs/llm_real_*/*_202607*`) ran on commit `15473b8` with in-process HF
transformers in fp16, one run at a time. This arm runs the same weights
(`~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b`, served as `qwen2.5_14b`
with `--dtype float16`) through the vLLM server, so several runs can share the GPU.
The server applies the chat template, and continuous batching makes runs
non-bit-reproducible (see `LLM_REMOTE_RUNBOOK.md` section 4b). Each manifest records
this under `thesis_arm.notes`.

hzreal_none and cross_1h have no demand after 3599 s. Their second hour only drains
the network: it costs less wall time, because empty intersections skip the LLM
call, and the runner stops early if the network fully empties. Residual gridlock
can keep a few vehicles in the network to 7200, since teleport is off. In an
offline test with a phase-cycling stand-in controller, both ran the full 7200
steps. cross_r1200 has demand through 9000 s and always runs the full 7200.

## Commands (on the GPU box, repo `~/LLM_DEV`)

```bash
cd ~/LLM_DEV && git fetch origin && git checkout thesis-llm-arm && git pull
uv sync --locked
tmux new -s thesis                               # or attach to it

bash tools/thesis_llm_arm.sh smoke               # starts vLLM if needed; about 5 min
bash tools/thesis_llm_arm.sh full 2 --dry_run    # print the 6 runner commands
bash tools/thesis_llm_arm.sh full 2 2>&1 | tee logs/thesis_llm_arm_launch.log
```

Progress, one line per run:

```bash
python tools/thesis_progress.py            # or: watch -n 60 .venv/bin/python tools/thesis_progress.py
python monitor_llm.py                      # server side: waiting, kv %, preemptions
```

Re-running `full` after a crash skips the runs that completed and redoes the rest.

## Outputs

- Runs: `logs/thesis_llm_arm/llm_<scenario>_seed<N>_<timestamp>/`, with the
  `thesis_arm` block in `run_manifest.json`
- Console logs: `logs/thesis_llm_arm/llm_<scenario>_seed<N>.console.log`
- Smoke: `logs/thesis_llm_arm_smoke/.../smoke_report.txt`
- Wall-clock per run: appended to `~/run_times.txt` in the paper 2 format
- Server log, when the launcher started the server: `logs/vllm_server_thesis.log`
