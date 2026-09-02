# LLM real-routes runs on the GPU box — runbook

The LLM grid on the base "real" (stochastic) Hangzhou routes, run on a
**custom-fine-tuned Qwen2.5-14B** (merged model at
`~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b`) on `cngpu-vm001`.
Four arms × seeds 1–3 (12 runs):

| Experiment            | Blockage | Who hears the incident? | Runs |
|-----------------------|----------|-------------------------|------|
| `llm_real_normal`     | none     | —                       | s1 s2 s3 |
| `llm_real_c3_text`    | C3       | both intersections (`2_3` approach + `2_4` exit) | s1 s2 s3 |
| `llm_real_c3_notext`  | C3       | neither (`--hide_blockage_info`) | s1 s2 s3 |
| `llm_real_c3_approach_only` | C3 | only downstream `2_3`; withheld from upstream `2_4` (`--blockage_info_scope approach`) | s1 s2 s3 |

All four are defined in `experiments.py` and drive through `run_matrix.py`, so
they get manifest identity, skip/reuse, and `build_results.py` aggregation for
free. Decoding is greedy (`temperature 0.0`, `do_sample False`), so run-to-run
variation is purely the SUMO `--seed` — the same seed seam the FT/MP baselines
use (`setup_run(..., seed=args.seed)` in `runner.py`).

Assumes `cngpu-vm001` already has the repo + `uv` env from the Jul 17 LLM run.

---

## 1. Sync the code

```bash
cd LLMTSC_SUMO
git fetch origin
git checkout blockage_staging
git pull
```

The LLM experiment defs travel with this branch. `models/LLMs/` and `logs/` are
gitignored, so weights and results do **not** come over git.

## 2. Refresh the env (cheap no-op if unchanged)

```bash
uv sync --locked
source .venv/bin/activate
```

## 3. Confirm the model

The runs point `--llm_path` at the merged fine-tuned model, wired in
`experiments.LLM_MODEL_PATH`:

```
~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
```

It's a merged model directory (not an HF cache folder), already present on the
box. Confirm it has both weights and a tokenizer:

```bash
ls ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
# expect: config.json, *.safetensors (or pytorch_model*.bin), and tokenizer
# files (tokenizer.json / tokenizer_config.json)
```

If the tokenizer files are missing the load fails — copy them from the base
Qwen2.5-14B, or re-export the merge with the tokenizer included.

> **Custom fine-tune caveat.** If it was trained on a different prompt format
> than `configurations.LLM_SYSTEM_PROMPT` + `utils/prompt_builder`, decisions
> may parse poorly. Watch `valid_resp_rate` in the seed-1 smoke (§5a); the
> manifest records the exact formatted prompt for inspection.

> Note: the model **is** part of run identity (`experiments._identity_fields`),
> so two models over the same config/seed/blockage combos are distinct results
> and share a `logs/` tree safely.

## 4. GPU + LLM smoke (before committing to full runs)

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python runner.py --test_name remote_smoke --simulation_steps 300 \
    --llm_path ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
```

Confirms the 14B loads on GPU (fp16 ≈ 28 GB, `device_map=auto`) and produces
parseable `<signal>` decisions. Optional CPU firewall sanity (toy net, stub
LLM, no GPU):

```bash
PYTHONPATH=. python tests/smoke_blockage_prompt_leakage.py
```

## 4b. Batched inference — mechanism check + acceptance (run once, on this box)

`runner.py` batches every intersection whose green window ends on the same step
into one `generate()` call by default — that's what gets a full run to ~6 h
instead of ~40 h. Two separate things must hold before trusting it.

**(1) The mechanism is sound.** On order-stable arithmetic, batching is
byte-identical to per-intersection inference — verified locally on CPU/fp32:
`tests/verify_batch_equivalence.py` (raw text + token usage 5/5) and
`tests/verify_batch_loop_equivalence.py` (decisions.jsonl identical, batch-size
invariant). The GPU run of that test is a *mechanism* check, **not** a
signal-parity pass/fail:

```bash
python tests/verify_batch_equivalence.py \
    --llm_path ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b \
    --max_new_tokens 1024
```
Require `size-1 batch == single: True`. **Expect a few `<signal>` mismatches on
fp16** — a near-tie logit flipping a token is inherent fp16 noise (the same test
is bit-exact on CPU/fp32), not a bug. Whether it *matters* is decided in (2).

**(2) It doesn't move the results.** Run a short sequential-vs-batched pair on the
same seed and compare the decisions and the bottom-line metrics:

```bash
python runner.py --simulation_steps 400 --seed 1 --sequential \
    --llm_path ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
python runner.py --simulation_steps 400 --seed 1 \
    --llm_path ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
python tests/compare_runs.py logs/<sequential_dir> logs/<batched_dir>
```
Read `compare_runs.py` as: *"decisions identical before first flip"* is the clean
fp16 sensitivity, and the **aggregate deltas** (ATT/AWT/throughput) are the
verdict. Accept batching when those deltas are **≪ the seed-to-seed spread** —
get that yardstick from two batched runs at different seeds (`--seed 1` vs
`--seed 2`). **Observed on this 14B (150-step pair): ~3% of LLM-queried decisions
flipped, benign — ATT identical, AWT within ~0.35 s, far below seed noise → batch
the grid.**

If the batched-vs-sequential deltas ever rival the seed spread, run sequentially:
set `"sequential": true` in the experiment's `extra` (run_matrix passes it
through) or call `runner.py --sequential` directly. Greedy decoding
(`temperature 0.0`) keeps every run deterministic for a given seed regardless.

If a wide batch OOMs, cap it with `"max_batch_size": 8` in `extra` (or
`--max_batch_size 8`); the runner also auto-falls-back to per-prompt on any batch
failure, so an OOM degrades instead of crashing the step.

## 4c. Alternative: run an arm on a hosted model (OpenRouter)

No GPU box required. Give `--llm_path` an `openrouter:<provider>/<model>` value
and `runner.build_llm` swaps the local HuggingFace backend for
`models_inference/LLM/openrouter_llm.py`, which talks to OpenRouter's
OpenAI-compatible API over stdlib urllib (no extra dependency, nothing to
`uv sync`). Everything downstream — prompts, parsing, batching, the decision
records — is unchanged.

```bash
set +o history                 # keep the key out of ~/.bash_history
export OPENROUTER_API_KEY=sk-or-...
set -o history
python runner.py --test_name openrouter_smoke --simulation_steps 300 \
    --llm_path openrouter:google/gemma-3-27b-it
```

Better for repeat use, put it in a gitignored `.env` (`.env` and `.env.*` are in
`.gitignore`) and source it, so the key is never typed into a shell or a script:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env && chmod 600 .env
set -a; . ./.env; set +a
```

The key is read **only** from the environment — there is no CLI flag for it, so it
cannot land in `run_manifest.json`'s `argv`/`args`, and `describe()` never
includes it (pinned by a check in `tests/smoke_openrouter_backend.py`). Never
paste it into `experiments.py` or a run script.

For a full arm, put the same string in the experiment's `extra`:
`{"llm_path": "openrouter:google/gemma-3-27b-it"}`. No `run_matrix.py` change is
needed, and because the model is part of run identity these results never pool
with the local 14B's.

### Testing several models, one after another

`--llm_paths` makes the model a sweep dimension like seeds and blockages. The
matrix runs them sequentially, one run per model:

```bash
python run_matrix.py --experiment llm_real_normal --seeds 1 --steps 300 \
    --llm_paths openrouter:google/gemma-3-27b-it \
                openrouter:google/gemma-3-12b-it \
                ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
```

Local paths and hosted models can be mixed freely. Because the model is part of
run identity:

* re-running the same list **skips** what is already done — adding a model to the
  list only spends on the new one;
* `build_results.py` reports **one row per model** (a `model` column in both
  sheets) instead of averaging different models into a single "llm" result;
* a blockage arm's ΔATT is measured against **that same model's** clean run.

Run dirs carry the model tag: `llm_hzreal_c3_seed1_gemma-3-27b-it_<timestamp>`.

Always `--dry_run` a multi-model sweep first — the combo count is
models × seeds × blockages, and every combo costs credit. A failing model (bad
id, exhausted quota) is tallied as `failed` and the sweep continues to the next.

`initialize_llm()` sends a 1-token preflight so a bad key, an unknown model, or a
dead network **crashes before SUMO starts** — the runner treats an inference
failure as "hold the current phase", so a persistent fault would otherwise burn a
whole run producing silently degraded control. The manifest's `llm` block records
`resolved_model` / `resolved_provider` (OpenRouter routes across providers, so a
hosted run is not reproducible the way a local one is) and never the API key.

Batching here is one concurrent request per intersection rather than one padded
`generate()`, so the §4b equivalence question does not arise — each prompt is an
independent call. `--max_batch_size` doubles as the concurrency cap; lower it if
you hit rate limits. `OPENROUTER_BASE_URL` overrides the endpoint.

Offline check of the whole backend (fake server, no key, no credit):

```bash
python tests/smoke_openrouter_backend.py
```

**Cost.** A 3600-step hzreal run queries the LLM ~1,630 times at ~759 input
tokens each (≈1.24 M input tokens). On `google/gemma-3-27b-it` that is roughly
**$0.30–0.55 per run** (~$10 for the whole 6-arm × 3-seed grid); `gemma-3-12b-it`
is about a third of that. Completion length dominates the spread. Avoid
`gemma-2-27b-it` — 8× the price of `gemma-3-27b-it`. The `:free` variants cost
nothing but are rate-limited, which collides with the concurrent batch: fine for
a smoke run, risky for a full arm. Shake out with `--simulation_steps 300` first.

## 4d. Fitting a model that's too big — `--quantization`

`--quantization {none,8bit,4bit}` loads a local model through bitsandbytes so one
larger than the GPU still fits. The A40 has ~44.7 GB free, which rules out fp16
for anything past ~20 B params:

| Model | fp16 | 8bit | 4bit |
|---|---|---|---|
| Qwen2.5-14B (fine-tuned) | ~28 GB ✅ | — | — |
| Gemma 4 26B A4B | ~50 GB ❌ | ~25 GB ✅ | ~13 GB ✅ |
| Gemma 4 31B | ~61 GB ❌ | ~31 GB ✅ | ~17 GB ✅ |

A Mixture-of-Experts model gets **no** memory relief from being sparse: Gemma 4
26B A4B activates only 3.8 B params per token but keeps all 128 experts resident.
Sparsity buys speed, not capacity. Without quantization `device_map="auto"` spills
to CPU and streams expert weights over PCIe every token — days per seed, not hours.

Quantization is part of run identity (`experiments._identity_fields`), so an 8bit
run never pools with a full-precision one, `build_results.py` gives it its own
row, and the run dir is tagged (`..._gemma-4-26B-A4B-it_think-off_8bit`). It is
recorded twice in `run_manifest.json`: under `args` (what was asked for, and what
skip/reuse matches on) and under `llm` (what the backend did, beside `torch_dtype`
and `device_map`). Default `none` keys identically to runs predating the flag, so
nothing already completed is invalidated.

`--quantization` is local-only; the OpenRouter backend warns and ignores it, since
the hosted server owns the precision it serves. `bitsandbytes` is a Linux-marked
dependency in `pyproject.toml` — it arrives via `uv sync --locked`. **Never
`uv pip install`** it or anything else here: that resolves outside the lock and
will pull PyPI's CUDA-13 torch over the pinned cu126 build, which breaks CUDA
outright on this box's 550 driver.

Offline check of the whole flag (no GPU, no weights):

```bash
python tests/smoke_local_quantization.py
```

### Gemma 4 specifics

- Use an **`-it`** repo. The base `google/gemma-4-26B-A4B` ships no chat template,
  and `_format_prompt` then falls back to Alpaca — wrong format *and* a model
  never trained to follow instructions.
- `gemma-4-12B-it` (the "Unified" encoder-free variant) **will not load** on
  transformers 5.9.0: its `gemma4_unified` model type is unregistered. The 31B,
  26B A4B and E4B all declare plain `gemma4` and work.
- Run with **`--reasoning off`**. Gemma 4 defaults to thinking on, reasoning draws
  from the same `LLM_MAX_NEW_TOKENS` budget, and a long think truncates before the
  `<signal>` — an empty answer the runner can only turn into a held phase.
- **`--reasoning on` is not usable on this family yet.** Gemma delimits reasoning
  as `<|channel>thought … <channel|>`, but `split_reasoning` only recognizes
  `<think>`/`</think>`, so chain-of-thought would flow into the signal parser. A
  thinking arm needs those delimiters added first.

```bash
python run_matrix.py --experiment llm_real_normal --seeds 1 2 3 \
    --llm_paths google/gemma-4-26B-A4B-it --reasoning off --quantization 8bit
```

## 5. Run in order — seed 1 first, verify, then the rest

Long runs: wrap each in `tmux`/`nohup` and tee a log. A full 3600-step 14B run
is the wall-clock unknown — time the first one before assuming the schedule.

### 5a. Normal, seed 1 — the pipeline validator

```bash
python run_matrix.py --experiment llm_real_normal --seeds 1
```

Then read the run's `final_summary.json`. This validates the whole real-routes
pipeline end-to-end: confirm `valid_resp_rate` ≈ 1.0 (the fine-tune emits
parseable `<signal>` decisions — the main risk with a custom-trained model) and
that the run completed.

**No pre-registered ATT anchor exists for this fine-tune** — the ~320 s figure
was the stock 7B's number, now void. To get a falsifiable gate, run this model
once on cfphys `hz1` first (seed-invariant, so one deterministic run fixes the
anchor), then judge `hzreal` against it:

```bash
python run_matrix.py --experiment llm_real_normal --configs hz1 --seeds 1
```

### 5b. Informed C3, seed 1 — then audit the prompts

```bash
python run_matrix.py --experiment llm_real_c3_text --seeds 1
```

Audit the logged prompts (this is the Jul 24 prompt audit, done off the real
decision log). The incident text must appear **only** at `intersection_2_3`
(approach) and `intersection_2_4` (exit), and **only** during steps 500–1700:

```bash
python - <<'PY'
import json, glob
run = sorted(glob.glob("logs/llm_real_c3_text/*seed1_*"))[-1]
hits = set()
for f in glob.glob(f"{run}/intersection_*/decisions.jsonl"):
    for line in open(f):
        d = json.loads(line)
        if d.get("blockage_info_in_prompt"):
            hits.add((d["intersection_id"], d["step"]))
iids  = {i for i, _ in hits}
steps = [s for _, s in hits]
print("run:", run)
print("intersections with incident text:", sorted(iids))
print("step range:", min(steps), "-", max(steps), f"({len(hits)} decisions)")
assert iids <= {"intersection_2_3", "intersection_2_4"}, iids
assert all(500 <= s <= 1700 for s in steps), (min(steps), max(steps))
print("PASS: incident text confined to 2_3/2_4, steps 500-1700")
PY
```

If the audit finds a defect, you've burned one run, not two — fix before 5c.

### 5c. Uninformed C3, seed 1

```bash
python run_matrix.py --experiment llm_real_c3_notext --seeds 1
```

### 5d. Approach-only C3, seed 1 (incident withheld from upstream 2_4)

```bash
python run_matrix.py --experiment llm_real_c3_approach_only --seeds 1
```

Re-run the 5b audit snippet against `logs/llm_real_c3_approach_only/*seed1_*`
but assert the incident text reaches **only** `intersection_2_3` this time —
i.e. change the assertion to `iids <= {"intersection_2_3"}`. If `2_4` shows up,
the scope flag isn't taking effect.

### 5e. Fill seeds 2–3 (pure throughput; the matrix skips the done seed-1 runs)

```bash
python run_matrix.py --experiment llm_real_normal
python run_matrix.py --experiment llm_real_c3_text
python run_matrix.py --experiment llm_real_c3_notext
python run_matrix.py --experiment llm_real_c3_approach_only
```

Re-running a script here is safe: a completed identity is skipped, only a
missing/crashed one re-runs.

## 6. Aggregate + bring results back

```bash
python build_results.py --experiment llm_real_normal
python build_results.py --experiment llm_real_c3_text
python build_results.py --experiment llm_real_c3_notext
python build_results.py --experiment llm_real_c3_approach_only
```

Copy the `logs/llm_real_*` trees back to the analysis machine (rsync/scp) —
they're gitignored, so they won't come back via git.

---

## 7. The C2 grid — one trigger

C2 blocks the West through lane into `intersection_2_2` (`road_1_2_0_1`, 10 m
upstream, steps 500–1700), so the incident is reported at `intersection_2_2`
(approach) and `intersection_1_2` (downstream exit). Two arms × seeds 1–3, plus
the clean baseline the C2 delta is measured against:

| Experiment            | Blockage | Who hears the incident? | Runs |
|-----------------------|----------|-------------------------|------|
| `llm_real_normal`     | none     | —                       | s1 s2 s3 |
| `llm_real_c2_text`    | C2       | both intersections (`2_2` approach + `1_2` exit) | s1 s2 s3 |
| `llm_real_c2_notext`  | C2       | neither (`--hide_blockage_info`) | s1 s2 s3 |

```bash
python run_llm_c2.py                      # all three arms, seeds 1-3
python run_llm_c2.py --dry_run            # print the plan, run nothing
python run_llm_c2.py --seeds 1            # seed 1 first, verify, then rerun
python run_llm_c2.py --arms text notext   # skip the clean baseline
```

Arms run one after the other (never concurrently — one 14B on one GPU) and each
lands in its own `logs/<experiment>/`. The clean phase costs nothing if
`llm_real_normal` seeds 1–3 already ran on this box: the matrix skips a
completed identity. Re-running after a crash resumes.

Wrap it: `tmux new -s c2 'python run_llm_c2.py 2>&1 | tee c2_grid.log'`.

Aggregate over **all** of `logs/` — the C2 arms live in different groups from
their clean baseline, so a per-experiment scope can't compute the ATT delta:

```bash
python build_results.py
```

> **Prompt audit.** `audit_prompts.py` hard-codes the C3 goldens
> (`intersection_2_3` / `intersection_2_4`, the "563 m … 573 m link" claim), so
> it does **not** audit these C2 runs. The §5b inline snippet still works —
> point it at `logs/llm_real_c2_text/*seed1_*` and assert
> `iids <= {"intersection_2_2", "intersection_1_2"}` with the same 500–1700
> window.
