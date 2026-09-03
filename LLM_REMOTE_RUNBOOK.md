# LLM real-routes runs on the GPU box — runbook

The LLM grid on the base "real" (stochastic) Hangzhou routes, run on a
**custom-fine-tuned Qwen2.5-14B** (merged model at
`~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b`, served by vLLM as
`qwen2.5_14b`) on `cngpu-vm001`.
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

Assumes `cngpu-vm001` already has the repo + `uv` env. vLLM installs into a
separate venv on first `serve_vllm.sh` — see §3.

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

## 3. Start the vLLM server

Runs decide against a vLLM server over HTTP; nothing loads a model in-process
any more. **vLLM lives in its own virtualenv** — `uv add vllm` into this project
would resolve its own torch over the cu126 build pinned in `uv.lock` and break
CUDA outright on this box's 550 driver. `serve_vllm.sh` builds that isolated env
on first use (`VLLM_VENV` overrides where, `VLLM_PORT` overrides the port):

```bash
tmux new -s vllm
bash serve_vllm.sh ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b qwen2.5_14b
```

The merged model dir still needs both weights and a tokenizer — vLLM applies the
chat template server-side, so a missing tokenizer fails the server, not the run:

```bash
ls ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
# expect: config.json, *.safetensors (or pytorch_model*.bin), and tokenizer
# files (tokenizer.json / tokenizer_config.json)
```

Then, in the shell that runs the grid:

```bash
export VLLM_SERVE_CMD="$(cat .vllm_serve_cmd)"   # recorded in every manifest
export VLLM_BASE_URL=http://localhost:8000/v1    # only if not the default
```

`--llm_path` names the **served model**, never a URL: `vllm:qwen2.5_14b`. Keeping
the endpoint in `VLLM_BASE_URL` is what lets the server move between ports
without invalidating a grid — the URL is not part of run identity, the served
name is. Encode the serving precision in that name
(`--served-model-name qwen2.5_14b-awq`) and it rides into identity, the run-dir
tag and the results' `model` column for free.

> **Custom fine-tune caveat.** If it was trained on a different prompt format
> than `configurations.LLM_SYSTEM_PROMPT` + `utils/prompt_builder`, decisions
> may parse poorly. Watch `valid_resp_rate` in the seed-1 smoke (§5a); the
> manifest records the exact messages sent for inspection.

> Note: the model **is** part of run identity (`experiments._identity_fields`),
> so two models over the same config/seed/blockage combos are distinct results
> and share a `logs/` tree safely.

> **Identity broke at the migration, deliberately.** Runs made against the
> retired in-process backend recorded a filesystem path in `args.llm_path`;
> these record `vllm:<served-name>`, so they do not pool and completed local runs
> will re-run. That is honest, not a bug: the stacks differ in attention
> implementation, KV dtype and reduction order, and the prompt itself now goes
> through the server's chat template rather than the client's. §4b below measured
> ~3% of decisions flipping from a mere left-padding change *on the same weights
> in the same stack*; a stack swap moves at least that. Do not add an alias.

### What the client can no longer see

The chat template, the dtype and the device map are the server's business now, so
the manifest records what it can observe instead: `llm.server.model_entry`
(`id`, `root` — the weights behind the served name — and `max_model_len`),
`llm.server.version` from `GET /version`, `llm.generation.extra_payload_sent`
(what `--reasoning` actually sent), and `llm.serve_cmd` if you exported it above.

## 4. Smoke it before committing to full runs

```bash
python runner.py --test_name remote_smoke --simulation_steps 300     --llm_path vllm:qwen2.5_14b
```

Preflight runs `GET /v1/models` and `GET /version` before SUMO starts, so a dead
server, a wrong port or a misspelled served name crashes immediately and names
what the server *does* serve — the runner treats an inference failure as "hold
the current phase", so a persistent fault would otherwise burn a whole run
producing silently degraded control.

Offline checks (no GPU, no server, no network):

```bash
python tests/smoke_http_backends.py            # the vLLM backend end to end
python tests/smoke_openrouter_backend.py       # the hosted backend
python tests/smoke_run_identity_flags.py       # identity did not shift
PYTHONPATH=. python tests/smoke_blockage_prompt_leakage.py
```

## 4b. Batched inference — no equivalence question any more

`runner.py` still batches every intersection whose green window ends on the same
step, but a batch is now **N independent HTTP requests** (one per intersection),
which vLLM fuses into one continuous batch server-side. There is no shared
padded `generate()`, so the outputs are the single-call outputs by construction.
`tests/verify_batch_equivalence.py` — which existed to prove the left-padding was
sound — is deleted along with the padding. `--max_batch_size` now caps
concurrency rather than tensor width, and the runner still falls back to
per-prompt on any batch failure.

`tests/verify_batch_loop_equivalence.py` still pins the loop itself (sequential
vs batched vs capped-batch record identically, on a stub, no GPU):

```bash
python tests/verify_batch_loop_equivalence.py
```

**Run the acceptance pair once on the new stack**, because the mechanism changed:

```bash
python runner.py --simulation_steps 400 --seed 1 --sequential --llm_path vllm:qwen2.5_14b
python runner.py --simulation_steps 400 --seed 1 --llm_path vllm:qwen2.5_14b
python tests/compare_runs.py logs/<sequential_dir> logs/<batched_dir>
```

The **aggregate deltas** (ATT/AWT/throughput) are the verdict; accept when they
are ≪ the seed-to-seed spread, which you get from two batched runs at different
seeds. For reference, the retired HF stack measured ~3% of LLM-queried decisions
flipping with ATT identical and AWT within ~0.35 s — far below seed noise.

Greedy decoding (`temperature 0.0`) keeps a run deterministic given a seed *in
principle*, but continuous batching varies batch composition with arrival timing,
so a vLLM run is not bit-reproducible even against itself. Pin the server version
(it lands in the manifest) and treat the acceptance pair, not bit-equality, as
the standard.

Time the first full run before assuming a schedule — the HF stack budgeted ~6 h
per 3600-step run; vLLM should be well under that, but it is unmeasured here.

## 4c. Alternative: run an arm on a hosted model (OpenRouter)

No GPU box and no vLLM server required. Give `--llm_path` an
`openrouter:<provider>/<model>` value and `runner.build_llm` routes to the
OpenRouter subclass in `models_inference/LLM/http_llm.py` instead of the vLLM
one. Both speak the same OpenAI-compatible API over stdlib urllib (no extra
dependency, nothing to `uv sync`), and everything downstream — prompts, parsing,
batching, the decision records — is identical.

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
with the self-hosted 14B's.

### Testing several models, one after another

`--llm_paths` makes the model a sweep dimension like seeds and blockages. The
matrix runs them sequentially, one run per model:

```bash
python run_matrix.py --experiment llm_real_normal --seeds 1 --steps 300 \
    --llm_paths openrouter:google/gemma-3-27b-it \
                openrouter:google/gemma-3-12b-it \
                vllm:qwen2.5_14b
```

Self-hosted and hosted models can be mixed freely. Because the model is part of
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

## 4d. Fitting a model that's too big — quantization is the server's job

`--quantization` no longer loads anything: the server owns its precision. Launch
vLLM with the scheme you want and *declare* it, so identity keeps the results
apart. The A40 has ~44.7 GB free, which rules out fp16 past ~20 B params:

| Model | fp16 | 8-bit | 4-bit |
|---|---|---|---|
| Qwen2.5-14B (fine-tuned) | ~28 GB ✅ | — | — |
| Gemma 4 26B A4B | ~50 GB ❌ | ~25 GB ✅ | ~13 GB ✅ |
| Gemma 4 31B | ~61 GB ❌ | ~31 GB ✅ | ~17 GB ✅ |

A Mixture-of-Experts model gets **no** memory relief from being sparse: Gemma 4
26B A4B activates only 3.8 B params per token but keeps all 128 experts
resident. Sparsity buys speed, not capacity.

```bash
bash serve_vllm.sh google/gemma-4-26B-A4B-it gemma-4-26B-A4B-it-awq     --quantization awq --max-model-len 8192
python run_matrix.py --experiment llm_real_normal --seeds 1 2 3     --llm_paths vllm:gemma-4-26B-A4B-it-awq --reasoning off
```

Two ways to keep an AWQ-served run from pooling with an fp16-served one, and you
want the first:

1. **Encode it in the served name** (`-awq` above). It then rides in `--llm_path`,
   which is identity, the run-dir tag *and* the results' `model` column.
2. `--quantization awq` as a bare label. It is part of identity, but it is
   **unverifiable** — `/v1/models` does not report the scheme, so nothing checks
   that it is true. It exists as the backstop for when someone forgets (1).

`--quantization` is warned-and-ignored on the OpenRouter arm, which owns the
precision it serves.

### Gemma 4 specifics

- Use an **`-it`** repo. The base `google/gemma-4-26B-A4B` ships no chat
  template, and the server has nothing to apply.
- Run with **`--reasoning off`**. Gemma 4 defaults to thinking on, reasoning
  draws from the same `--max_new_tokens` budget, and a long think truncates
  before the `<signal>` — an empty answer the runner can only turn into a held
  phase. Startup now *proves* the flag took effect: two free `POST /tokenize`
  calls render the same messages with `enable_thinking` true and false, and
  identical token counts mean the template ignores the variable, which raises
  rather than thinking anyway.
- **`--reasoning on` is now usable on this family** — the old blocker was that
  Gemma delimits reasoning as `<|channel>thought … <channel|>` while the client
  only knew `<think>`. Launch the server with the matching `--reasoning-parser`
  and it returns `reasoning_content` separately, leaving clean `content` for the
  signal parser. If you forget, the client warns on the first 3 decisions that a
  reasoning marker survived into the answer — never silently parse a chain of
  thought.

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
