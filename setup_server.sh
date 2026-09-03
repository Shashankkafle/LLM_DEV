#!/usr/bin/env bash
# One-shot, idempotent setup for any Linux machine (tested target: A40,
# driver 550.54.14); also runs under Git Bash on Windows (CPU-only torch
# there, per uv.lock). Safe to re-run; every step skips work already done.
#
#   bash setup_server.sh
#
# All dependency logic lives in pyproject.toml + uv.lock (committed):
# `uv sync` installs the pinned Python, the exact locked package set, the
# SUMO simulator (eclipse-sumo wheel, binary in .venv/bin), and the right
# torch build per platform (cu126 CUDA wheel on Linux -- never PyPI's
# default, whose Linux wheel is a CUDA-13 build that breaks on driver < 580).
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n=== %s ===\n' "$*"; }

step "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "environment (Python + all deps from uv.lock)"
uv sync --locked
if [ -e .venv/bin/python ]; then
    VENV_PY=".venv/bin/python"
    SUMO_BIN=".venv/bin/sumo"
    ON_WINDOWS=0
else
    VENV_PY=".venv/Scripts/python.exe"
    SUMO_BIN=".venv/Scripts/sumo.exe"
    ON_WINDOWS=1
fi

step "LLM serving (vLLM, separate environment)"
# No weights are downloaded here any more: the in-process HuggingFace backend
# was retired, and runs decide against a vLLM server over HTTP. vLLM resolves
# its own torch, so it must NOT be installed into this project's environment --
# it would pull PyPI's CUDA-13 wheel over the cu126 build pinned in uv.lock and
# break CUDA on driver 550. serve_vllm.sh builds an isolated venv for it:
#
#   bash serve_vllm.sh <model-path-or-repo-id> qwen2.5_14b   # in its own shell
#   export VLLM_SERVE_CMD="$(cat .vllm_serve_cmd)"
#   python runner.py --llm_path vllm:qwen2.5_14b
#
# Set VLLM_BASE_URL if the server is not on http://localhost:8000/v1.
echo "skipped (see serve_vllm.sh -- vLLM installs into its own venv)"

step "verify: SUMO"
"$SUMO_BIN" --version | head -n 2

step "verify: torch sees the GPU"
if [ "$ON_WINDOWS" = 1 ]; then
    # uv.lock pins the CPU-only torch wheel on win32, so no CUDA check here.
    "$VENV_PY" -c "import torch; print(torch.__version__, '| CPU build (Windows)')"
else
    "$VENV_PY" -c "
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False -- check driver / wheel'
print(torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))
"
fi

step "verify: TF / traci / transformers import"
TF_USE_LEGACY_KERAS=1 "$VENV_PY" -c "
import tensorflow, tf_keras, traci, sumolib, transformers
print('tensorflow', tensorflow.__version__, '| tf_keras', tf_keras.__version__)
print('traci', traci.__version__, '| transformers', transformers.__version__)
"

step "verify: 120-step headless end-to-end SUMO run (fixedtime baseline)"
"$VENV_PY" runner_baselines.py --controller fixedtime \
    --simulation_steps 120 --test_name setup_smoke

step "done"
if [ "$ON_WINDOWS" = 1 ]; then
    echo "Setup complete. Activate with: .venv\\Scripts\\activate (or 'source .venv/Scripts/activate' in Git Bash)"
else
    echo "Setup complete. Activate with: source .venv/bin/activate"
fi
echo "LLM smoke test:  python runner.py --test_name server_smoke --simulation_steps 300"
