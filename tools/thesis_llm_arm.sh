#!/usr/bin/env bash
# Thesis LLM arm: the paper 2 prompt with no incident report (no event exists),
# on three demand-surge scenarios of the Hangzhou 4x4, seeds 1-2, 7200 s.
#
#   bash tools/thesis_llm_arm.sh smoke              # cross_r1200 s1 for 120 sim-s, then the report
#   bash tools/thesis_llm_arm.sh full               # the 6 runs, 2 in parallel
#   bash tools/thesis_llm_arm.sh full 3             # ... 3 in parallel
#   bash tools/thesis_llm_arm.sh full 2 --dry_run   # print the plan, run nothing
#   bash tools/thesis_llm_arm.sh repair             # fill in missing run_times / thesis_arm records
#   VLLM_PORT=8010 bash tools/thesis_llm_arm.sh ... # if something else holds 8000
#   THESIS_ROWS="cross_r1500 cross_r1800" bash tools/thesis_llm_arm.sh full 2
#                                                   # other scenario rows, seeds 1-2 each
#
# Both modes first make sure the vLLM server is up and serving the paper 2
# weights, and start it if nothing is listening. Runs are the paper 2 runner
# argv (runner.py, three_lane, --seed) with only the scenario, the horizon and
# the vllm: model path changed. A run that already completed is skipped, so
# re-running `full` after a crash resumes.
#
# Logs:      logs/thesis_llm_arm/llm_<scenario>_seed<N>_<timestamp>/
# Console:   logs/thesis_llm_arm/llm_<scenario>_seed<N>.console.log
# Wall-clock per run is appended to ~/run_times.txt in the paper 2 format.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH=~/LLMTSCS-custom_prompts/ft_models/merged/qwen2.5_14b
SERVED_NAME=qwen2.5_14b
# Paper 2 ran this model in torch.float16; pin it rather than let vLLM pick
# from config.json. The rest are the runbook's explicit limits (section 4e).
GPU_MEMORY_PERCENT=90
SERVE_ARGS=(--dtype float16 --max-model-len 4096 --max-num-seqs 64
            --gpu-memory-utilization "0.$GPU_MEMORY_PERCENT")
# One port drives both the server serve_vllm.sh starts and the URL the runs
# call. The GPU box is shared, and JupyterHub holds 8000 and 8001 there:
#   VLLM_PORT=8010 bash tools/thesis_llm_arm.sh smoke
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:$VLLM_PORT/v1}"
# The box's /usr/bin/nvcc is CUDA 11, too old for flashinfer to JIT-build its
# sampler, so use vLLM's PyTorch sampler (greedy = argmax either way). vLLM's
# other JIT builds call `ninja`, which lives only in the vLLM venv.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.venvs/vllm/bin:$PATH"
PYTHON="${PYTHON:-.venv/bin/python}"

# Overridable only so the launcher itself can be tested quickly; the arm is 7200.
HORIZON="${THESIS_HORIZON:-7200}"
SMOKE_STEPS=120
RUN_GROUP=thesis_llm_arm
SMOKE_GROUP=thesis_llm_arm_smoke
RUN_TIMES="${RUN_TIMES:-$HOME/run_times.txt}"
SERVER_LOG=logs/vllm_server_thesis.log
SERVER_WAIT_S=1200
# Not part of run identity. Raised from 120 s because parallel runs queue at
# the server, and a timeout is retried, which repeats the whole generation.
REQUEST_TIMEOUT_S=300

DATASET=dataset/llm_light/Hangzhou/4_4
declare -A SUMOCFG=(
    [hzreal_none]="$DATASET/anon_4_4_hangzhou_real.sumocfg"
    [cross_1h]="$DATASET/anon_4_4_hangzhou_real_cross_2_3_1h.sumocfg"
    [cross_r1200]="$DATASET/anon_4_4_hangzhou_real_cross_2_3_r1200.sumocfg"
    [cross_r1500]="$DATASET/anon_4_4_hangzhou_real_cross_2_3_r1500.sumocfg"
    [cross_r1800]="$DATASET/anon_4_4_hangzhou_real_cross_2_3_r1800.sumocfg"
)
DEFAULT_ROWS="cross_r1200 cross_1h hzreal_none"
SEEDS=(1 2)

# Every row for seed 1, then every row for seed 2.
RUN_ORDER=()
for seed in "${SEEDS[@]}"; do
    for row in ${THESIS_ROWS:-$DEFAULT_ROWS}; do
        RUN_ORDER+=("$row $seed")
    done
done

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "ERROR: $*"; exit 1; }

for row in ${THESIS_ROWS:-$DEFAULT_ROWS}; do
    [ -n "${SUMOCFG[$row]:-}" ] || die "unknown row '$row'. Known: ${!SUMOCFG[*]}"
done

# --- server ---------------------------------------------------------------

served_models() {
    curl -sf --max-time 5 "$VLLM_BASE_URL/models" 2>/dev/null \
        | "$PYTHON" -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' \
        2>/dev/null || true
}

# The GPU is shared with other users. vLLM claims GPU_MEMORY_FRACTION of the
# card's TOTAL memory at start-up, so fail now with the reason rather than
# after SERVER_WAIT_S of waiting for a server that cannot come up.
check_gpu_memory() {
    command -v nvidia-smi > /dev/null || return 0
    local total used free needed
    read -r total used <<< "$(nvidia-smi --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits | head -1 | tr -d ',')"
    free=$(( total - used ))
    needed=$(( total * GPU_MEMORY_PERCENT / 100 ))
    if (( free < needed )); then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
        die "GPU has ${free} MiB free, vLLM needs ${needed} MiB (${GPU_MEMORY_PERCENT}% of ${total}). Wait for the processes above to exit."
    fi
}

port_in_use() {
    (exec 3<> "/dev/tcp/127.0.0.1/$VLLM_PORT") 2> /dev/null
}

# Something that is not a vLLM server can hold the port (on the GPU box,
# JupyterHub listens on 8000 and 8001); ours would then die at start-up.
check_port_free() {
    if port_in_use; then
        die "port $VLLM_PORT is already in use by something that is not a vLLM server. Pick a free one, e.g. VLLM_PORT=8010 bash tools/thesis_llm_arm.sh ..."
    fi
}

start_server() {
    check_gpu_memory
    check_port_free
    log "no server at $VLLM_BASE_URL; starting it (log: $SERVER_LOG)"
    mkdir -p logs
    # setsid -w stays alive as long as the server does, so $! tracks it.
    setsid -w nohup bash serve_vllm.sh "$MODEL_PATH" "$SERVED_NAME" "${SERVE_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1 < /dev/null &
    local server_pid=$! waited=0
    until [ -n "$(served_models)" ]; do
        if ! kill -0 "$server_pid" 2> /dev/null; then
            tail -5 "$SERVER_LOG"
            die "the server exited during start-up; see $SERVER_LOG"
        fi
        (( waited >= SERVER_WAIT_S )) && die "server not up after ${SERVER_WAIT_S}s; see $SERVER_LOG"
        sleep 10
        waited=$(( waited + 10 ))
    done
    log "server up after ${waited}s"
}

ensure_server() {
    local served
    served="$(served_models)"
    if [ -z "$served" ]; then
        start_server
        served="$(served_models)"
    fi
    # Never stop a server someone else started: say what it serves and quit.
    [[ " $served " == *" $SERVED_NAME "* ]] \
        || die "the server at $VLLM_BASE_URL serves '$served', not '$SERVED_NAME'. Stop it first."
    # serve_vllm.sh drops its launch command here; the runner records it in
    # every manifest under llm.serve_cmd.
    if [ -f .vllm_serve_cmd ]; then
        export VLLM_SERVE_CMD="$(cat .vllm_serve_cmd)"
    fi
    log "server OK: $SERVED_NAME at $VLLM_BASE_URL"
}

# --- one run --------------------------------------------------------------

test_name_of() { echo "llm_${1}_seed${2}"; }

newest_run_dir() {
    local group="$1" test_name="$2"
    { ls -td "logs/$group/${test_name}_"2*/ 2>/dev/null || true; } | head -1 | sed 's:/$::'
}

is_completed() {
    local run_dir="$1"
    [ -n "$run_dir" ] && [ -f "$run_dir/run_manifest.json" ] || return 1
    "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "completed")' \
        "$run_dir/run_manifest.json"
}

# Paper 2 format: "== <group>/<run>", then "<mtime> <path>" for the run's first
# file (start) and for run_manifest.json, which is finalized last (end).
append_run_time() {
    local rel="$1"
    (
        cd logs
        echo "== $rel"
        # sed, not head: head exits early, sort then dies of SIGPIPE, and
        # pipefail + set -e silently killed the whole run job.
        find "$rel" -type f -printf '%T@ %TY-%Tm-%Td %TT %p\n' | sort -n | sed -n 1p | cut -d' ' -f2-
        find "$rel/run_manifest.json" -printf '%TY-%Tm-%Td %TT %p\n'
    ) >> "$RUN_TIMES"
}

# Sets RUNNER_CMD to the argv of one run.
build_runner_cmd() {
    local scenario="$1" seed="$2" steps="$3" group="$4"
    RUNNER_CMD=("$PYTHON" runner.py
        --test_name "$(test_name_of "$scenario" "$seed")" --run_group "$group"
        --simulation_config "${SUMOCFG[$scenario]}" --simulation_steps "$steps"
        --intersection_config three_lane --seed "$seed"
        --llm_path "vllm:$SERVED_NAME" --request_timeout "$REQUEST_TIMEOUT_S")
}

run_one() {
    local scenario="$1" seed="$2" steps="$3" group="$4" record_time="$5"
    local test_name console exit_code=0 run_dir
    test_name="$(test_name_of "$scenario" "$seed")"
    console="logs/$group/$test_name.console.log"
    mkdir -p "logs/$group"

    log "start $test_name ($steps steps) -> $console"
    build_runner_cmd "$scenario" "$seed" "$steps" "$group"
    "${RUNNER_CMD[@]}" > "$console" 2>&1 || exit_code=$?

    run_dir="$(newest_run_dir "$group" "$test_name")"
    if [ -z "$run_dir" ] || [ ! -f "$run_dir/run_manifest.json" ]; then
        log "FAILED $test_name before creating a run dir (exit $exit_code); see $console"
        return 0
    fi
    # Record the wall-clock first: the end time is run_manifest.json's mtime,
    # which the identity block below would otherwise move.
    if [ "$record_time" = yes ]; then
        append_run_time "$group/$(basename "$run_dir")" \
            || log "WARN: could not append $test_name to $RUN_TIMES; run: bash tools/thesis_llm_arm.sh repair"
    fi
    "$PYTHON" tools/thesis_arm_manifest.py "$run_dir" \
        --scenario "$scenario" --seed "$seed" --horizon "$steps" \
        || log "WARN: thesis_arm block not written for $test_name; run: bash tools/thesis_llm_arm.sh repair"
    log "end $test_name exit=$exit_code run_dir=$run_dir"
}

has_thesis_block() {
    "$PYTHON" -c 'import json,sys; sys.exit("thesis_arm" not in json.load(open(sys.argv[1])))' \
        "$1/run_manifest.json"
}

# A complete entry is the "==" line followed by the run_manifest.json line.
has_complete_run_time() {
    local rel="$1"
    grep -A2 -x "== $rel" "$RUN_TIMES" 2> /dev/null | grep -q "$rel/run_manifest.json"
}

# Fills in what a killed record step left out, for every completed run. Order
# matters: the run_times end time is run_manifest.json's mtime, which writing
# the thesis_arm block moves.
repair() {
    local entry scenario seed run_dir rel
    for entry in "${RUN_ORDER[@]}"; do
        read -r scenario seed <<< "$entry"
        run_dir="$(newest_run_dir "$RUN_GROUP" "$(test_name_of "$scenario" "$seed")")"
        is_completed "$run_dir" || continue
        rel="$RUN_GROUP/$(basename "$run_dir")"
        if ! has_thesis_block "$run_dir"; then
            if ! has_complete_run_time "$rel"; then
                append_run_time "$rel"
                log "appended a complete $RUN_TIMES entry for $rel (delete any older partial one)"
            fi
            "$PYTHON" tools/thesis_arm_manifest.py "$run_dir" \
                --scenario "$scenario" --seed "$seed" --horizon "$HORIZON"
        elif ! has_complete_run_time "$rel"; then
            log "WARN: $rel has no complete $RUN_TIMES entry, and its manifest mtime has moved; take the end time from thesis_arm.ended_at"
        fi
    done
    log "repair done"
}

# --- modes ----------------------------------------------------------------

smoke() {
    ensure_server
    run_one cross_r1200 1 "$SMOKE_STEPS" "$SMOKE_GROUP" no
    local run_dir
    run_dir="$(newest_run_dir "$SMOKE_GROUP" "$(test_name_of cross_r1200 1)")"
    [ -n "$run_dir" ] || die "smoke run left no run dir"
    local report_ok=yes
    "$PYTHON" tools/thesis_smoke_report.py "$run_dir" | tee "$run_dir/smoke_report.txt" || report_ok=no
    log "smoke report saved to $run_dir/smoke_report.txt (passed: $report_ok)"
    [ "$report_ok" = yes ]
}

full() {
    local parallel="$1" dry_run="$2" entry scenario seed run_dir
    (( parallel >= 1 )) || die "parallel must be >= 1"
    if [ "$dry_run" = yes ]; then
        for entry in "${RUN_ORDER[@]}"; do
            read -r scenario seed <<< "$entry"
            build_runner_cmd "$scenario" "$seed" "$HORIZON" "$RUN_GROUP"
            echo "${RUNNER_CMD[*]}"
        done
        return
    fi

    ensure_server
    log "${#RUN_ORDER[@]} runs, $parallel in parallel, horizon $HORIZON"
    for entry in "${RUN_ORDER[@]}"; do
        read -r scenario seed <<< "$entry"
        run_dir="$(newest_run_dir "$RUN_GROUP" "$(test_name_of "$scenario" "$seed")")"
        if is_completed "$run_dir"; then
            log "skip $(basename "$run_dir") (completed)"
            continue
        fi
        while (( $(jobs -rp | wc -l) >= parallel )); do
            wait -n || true
        done
        run_one "$scenario" "$seed" "$HORIZON" "$RUN_GROUP" yes &
        # Stagger starts so two runs never share a SUMO start-up second.
        sleep 5
    done
    wait
    log "all runs finished"
    "$PYTHON" tools/thesis_progress.py "logs/$RUN_GROUP"
}

case "${1:-}" in
    smoke) smoke ;;
    repair) repair ;;
    full)
        parallel="${2:-2}"
        dry_run=no
        [ "${3:-}" = "--dry_run" ] && dry_run=yes
        full "$parallel" "$dry_run"
        ;;
    *) echo "usage: bash tools/thesis_llm_arm.sh smoke | full [N_PARALLEL] [--dry_run] | repair"; exit 2 ;;
esac
