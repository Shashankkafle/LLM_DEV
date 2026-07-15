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

QWEN_REPO="Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"

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

step "default LLM weights ($QWEN_REPO @ ${QWEN_REVISION:0:12})"
# snapshot_download API rather than the `hf` CLI: the API surface is stable
# across hub versions. Other models: same call with a different repo id / no
# revision pin, then pass the path via --llm_path at run time.
QWEN_REPO="$QWEN_REPO" QWEN_REVISION="$QWEN_REVISION" "$VENV_PY" -c "
import os
from huggingface_hub import snapshot_download
path = snapshot_download(os.environ['QWEN_REPO'],
                         revision=os.environ['QWEN_REVISION'],
                         cache_dir='models/LLMs')
print('model at:', path)
"

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
