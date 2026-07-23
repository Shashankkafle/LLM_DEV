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

> Note: the model is **not** part of run identity. Don't run a different model
> over these same config/seed/blockage combos in this `logs/` tree — the matrix
> would treat them as the same result.

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

## 4b. Batched-inference equivalence gate (run once, on this box)

`runner.py` batches every intersection whose green window ends on the same step
into one `generate()` call by default — that's what gets a full run to ~6 h
instead of ~40 h. Batching is provably equivalent to per-intersection inference
on order-stable arithmetic (CPU/fp32: byte-identical outputs, verified in CI),
but on **fp16 GPU a near-tie logit can rarely flip a token**, so certify it once
against the real model at the production token budget before trusting the grid:

```bash
python tests/verify_batch_equivalence.py \
    --llm_path ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b \
    --max_new_tokens 1024
```

Acceptance: **`<signal> mismatches: 0`** and `size-1 batch == single: True`.
A nonzero raw-text mismatch with zero signal mismatches is the benign fp16 case
(the decision is unchanged). If any `<signal>` flips, run the grid with
`--sequential` (add it in `experiments.py` `extra`) or investigate before pooling
results. Note: greedy decoding (`temperature 0.0`) → runs are still deterministic
run-to-run for a given seed; batched vs sequential is the only comparison at issue.

If a wide batch ever OOMs, cap it with `--max_batch_size 8` (the runner also
auto-falls-back to per-prompt on any batch failure, so an OOM degrades instead
of crashing the step).

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
