#!/usr/bin/env bash
# Start the vLLM server the LLM runs decide against.
#
#   bash serve_vllm.sh ~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b qwen2.5_14b
#   python runner.py --llm_path vllm:qwen2.5_14b        # in another shell
#
# vLLM lives in its OWN virtualenv, never the repo's. It resolves its own torch,
# and `uv add vllm` would pull that over the cu126 build pinned in uv.lock --
# which breaks CUDA outright on this box's 550 driver. The two environments
# share nothing but the HTTP port.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH="${1:?usage: serve_vllm.sh <model-path-or-repo-id> <served-name> [vllm args...]}"
SERVED_NAME="${2:?usage: serve_vllm.sh <model-path-or-repo-id> <served-name> [vllm args...]}"
shift 2

VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm}"
PORT="${VLLM_PORT:-8000}"

if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
    echo "=== creating the vLLM environment at $VLLM_VENV ==="
    # Deliberately outside this repo's uv project: see the header.
    uv venv "$VLLM_VENV"
    VIRTUAL_ENV="$VLLM_VENV" uv pip install vllm
fi

# --served-model-name is what --llm_path names, and it is part of run identity.
# Encode the serving precision in it (qwen2.5_14b-awq) so the scheme rides in
# the manifest, the run-dir tag and the results' model column for free.
CMD=("$VLLM_VENV/bin/vllm" serve "$MODEL_PATH"
     --served-model-name "$SERVED_NAME"
     --port "$PORT"
     "$@")

# A served run's reproducibility lives in these launch flags, which are outside
# the repo, so drop them where the client can pick them up. The client is a
# different process (usually a different shell), so exporting here would not
# reach it -- in the shell that runs the grid:
#
#     export VLLM_SERVE_CMD="$(cat .vllm_serve_cmd)"
#
# runner.py then records it verbatim in run_manifest.json under llm.serve_cmd.
printf '%s
' "${CMD[*]}" > .vllm_serve_cmd
echo "=== ${CMD[*]} ==="
echo "=== client: export VLLM_SERVE_CMD=\"\$(cat .vllm_serve_cmd)\" ==="
exec "${CMD[@]}"
