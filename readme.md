# LLMTSC_SUMO

LLM-based traffic signal control on SUMO, with fixed-time / max-pressure and
CoLight (RL) baselines running through the same environment seam.

## Setup (any machine, Linux or Windows)

Dependencies are managed with [uv](https://docs.astral.sh/uv/):
`pyproject.toml` holds the loose direct dependencies, `uv.lock` (committed)
holds the exact cross-platform resolution. The same lock installs the right
wheels on every machine — CUDA torch on Linux, CPU torch on Windows.

On a Linux GPU server, one command (no sudo):

```bash
git clone <repo-url> && cd LLMTSC_SUMO
bash setup_server.sh
```

The script is idempotent and self-verifying: `uv sync` (installs pinned
Python 3.11.15, all locked packages, and the SUMO 1.27.1 simulator via the
`eclipse-sumo` wheel — binary lands in `.venv/bin`), downloads the default
LLM (Qwen2.5-0.5B-Instruct, pinned revision) into `models/LLMs/`, then
verifies SUMO, torch-on-GPU, TF/traci imports, and a 120-step headless run.

On any other machine, the environment alone is just:

```bash
uv sync
```

Then:

```bash
source .venv/bin/activate
python runner.py --test_name server_smoke --simulation_steps 300
```

## Upgrading dependencies (deliberately, never by accident)

- `uv sync` never changes versions — it installs exactly what `uv.lock` says.
- The ML core (torch, transformers, tokenizers, numpy, TF pair, SUMO trio) is
  held at known-good versions by exact pins in `[project.dependencies]` and
  `[tool.uv] constraint-dependencies` in `pyproject.toml`.
- To upgrade one package: relax its pin/constraint, run
  `uv lock --upgrade-package <name>`, test, commit the new lock.
- Every known-good state is a git commit of `uv.lock`. If a machine breaks
  after an upgrade: `git checkout <good-sha> -- uv.lock && uv sync`.
- Every run also records its actual torch/transformers/TF/traci/numpy
  versions in `run_manifest.json` under `logs/`, so results stay traceable to
  their dependency set forever.

### Why torch comes from the PyTorch indexes, not PyPI

PyPI's default Linux torch wheel is currently a CUDA-13 build that **fails on
drivers < 580** (the A40 server runs driver 550). `pyproject.toml` routes
torch to the cu126 index on Linux (works on any driver >= 525) and the cpu
index on Windows. `uv sync` applies this automatically — never install torch
manually.

### Dependency files

- `pyproject.toml` + `uv.lock` — **authoritative**; all installs go through
  `uv sync`
- `requirements.lock` — frozen snapshot of the 2026-07-14 known-good venv,
  kept as a historical record only
- `uv-packages.txt`, `req.txt` — legacy, superseded

## Running other LLMs

Download any HF model into the local cache and point a run at it:

```bash
hf download Qwen/Qwen2.5-7B-Instruct --cache-dir models/LLMs
python runner.py --llm_path models/LLMs/models--Qwen--Qwen2.5-7B-Instruct ...
```

(`--llm_path` accepts the `models--Org--Name` cache folder directly; the code
descends into `snapshots/<revision>/` itself. The `snapshot_download` Python
API works too — it's what `setup_server.sh` uses.)

A40 headroom (46 GB): fp16 comfortably fits models up to ~14B; for ~70B-class
use 4-bit quantization (`accelerate` is already installed).

## CoLight note

CoLight's TensorFlow is deliberately the CPU build (`tensorflow-cpu==2.21.0`
+ `tf-keras==2.21.0`); its networks are tiny and the GPU is reserved for
torch/LLM inference. `TF_USE_LEGACY_KERAS=1` is set in code before TF import.
