#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Baselines sweep, one algorithm per GPU (parallel)
# ssh -i ~/.ssh/ContinuumFLPaper cc@192.5.86.200
#
#  One dataset, each GPU runs a different baseline algorithm.
#  GPU 0 → ClusterFL   GPU 1 → IFCA   GPU 2 → APCfl
#  GPU 3 → GeoFL       (SnapCFL shares a GPU by default)
#
#  Usage:
#    bash scripts/run_baselines_sweep_4gpu.sh femnist
#    bash scripts/run_baselines_sweep_4gpu.sh cifar100
#    DRY_RUN=true bash scripts/run_baselines_sweep_4gpu.sh femnist
#    RESUME_FROM=results/... bash scripts/run_baselines_sweep_4gpu.sh femnist
#
#  Notifications:
#    NOTIFY_WEBHOOK=https://ntfy.sh/your-topic bash scripts/run_baselines_sweep_4gpu.sh femnist
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Raise open-file limit up to the hard cap — never exceed it
_soft_limit=$(ulimit -Sn 2>/dev/null || echo 1024)
_hard_limit=$(ulimit -Hn 2>/dev/null || echo 1024)
_target=65536
(( _target > _hard_limit )) && _target=$_hard_limit
if (( _target > _soft_limit )); then
    ulimit -Sn "$_target" 2>/dev/null || true
fi
echo "  Open-file limit: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
unset _soft_limit _hard_limit _target

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DATASET  (first argument, required)
# └─────────────────────────────────────────────────────────────────────────────
DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "Usage: bash scripts/run_baselines_sweep_4gpu.sh <dataset>" >&2
    echo "  datasets: femnist cifar100 shakespeare speechcommands" >&2
    exit 1
fi

# ┌─────────────────────────────────────────────────────────────────────────────
# │ METHOD → GPU assignment
# │   Override via env vars, e.g.: GPU_IFCA=2 bash ... femnist
# └─────────────────────────────────────────────────────────────────────────────
GPU_ClusterFL="${GPU_ClusterFL:-0}"
GPU_IFCA="${GPU_IFCA:-1}"
GPU_GeoFL="${GPU_GeoFL:-2}"
GPU_SnapCFL="${GPU_SnapCFL:-3}"

METHOD_GPU_MAP=(
    "ClusterFL:${GPU_ClusterFL}"
    "IFCA:${GPU_IFCA}"
    "GeoFL:${GPU_GeoFL}"
    "SnapCFL:${GPU_SnapCFL}"
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ NOTIFICATIONS
# └─────────────────────────────────────────────────────────────────────────────
NOTIFY_WEBHOOK="${NOTIFY_WEBHOOK:-}"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ GPU MEMORY THRESHOLD
# └─────────────────────────────────────────────────────────────────────────────
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-4000}"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ FIXED PARAMETERS
# └─────────────────────────────────────────────────────────────────────────────
NUM_ROUNDS="${NUM_ROUNDS:-200}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LEARNING_RATE=0.001
BATCH_SIZE=32
INTRA_ZONE_ALPHA=100
INTER_ZONE_ALPHA=5.0
COMPRESSION_RATE=0.10
SPATIAL_WEIGHT=0.4
DATA_WEIGHT=0.4
NETWORK_WEIGHT=0.2
SPATIAL_REGULARIZATION=0.05
CORRELATION_THRESHOLD=0.05
RANDOM_SEED=42

declare -A DATASET_MAX_SAMPLES=(
    [femnist]=50000
    [cifar100]=50000
    [shakespeare]=50000
    [speechcommands]=50000
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ ZONE CONFIGS   "NUM_DEVICES:NUM_ZONES:MIN_ZONE_SIZE:MAX_ZONE_SIZE:LABEL"
# └─────────────────────────────────────────────────────────────────────────────
ZONE_CONFIGS=(
    "10:2:3:6:10c_2z"
    "10:5:1:4:10c_5z"
    "50:5:4:15:50c_5z"
    "50:10:3:8:50c_10z"
    # "50:25:1:4:50c_25z"
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ FAULT CONFIGS   "DEVICE_FAIL:ZONE_FAIL:LABEL"
# └─────────────────────────────────────────────────────────────────────────────
FAULT_CONFIGS=(
    "0.00:0.00:fault_free"
    "0.05:0.00:dev_low"
    # "0.10:0.00:dev_moderate"
    "0.20:0.00:dev_high"
    "0.05:0.02:dev_low__zone_low"
    # "0.10:0.05:dev_moderate__zone_moderate"
    "0.20:0.10:dev_high__zone_high"
    # "0.30:0.15:severe"
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PATHS
# └─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$PROJECT_ROOT/logs/baselines_sweep_4gpu_${TIMESTAMP}"
mkdir -p "$LOG_ROOT"

PROGRESS_FILE="$LOG_ROOT/progress.txt"
{
    echo "ContinuumFL — Baselines sweep progress"
    echo "Dataset  : $DATASET"
    echo "Started  : $(date)"
    echo "─────────────────────────────────────────"
} > "$PROGRESS_FILE"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PYTHON
# └─────────────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    command -v python3 &>/dev/null && PYTHON_BIN="python3" || {
        echo "❌ Neither 'python' nor 'python3' found." >&2; exit 1
    }
fi
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

# ┌─────────────────────────────────────────────────────────────────────────────
# │ HELPER: check GPU free memory
# └─────────────────────────────────────────────────────────────────────────────
check_gpu_memory() {
    local gpu_id="$1"
    if ! command -v nvidia-smi &>/dev/null; then
        echo "  ⚠️  nvidia-smi not found — skipping GPU memory check"
        return 0
    fi
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
              --id="$gpu_id" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$free_mb" ]]; then
        echo "  ⚠️  Could not read memory for GPU $gpu_id"; return 0
    fi
    if (( free_mb < MIN_FREE_MEM_MB )); then
        echo "  ⚠️  GPU $gpu_id: only ${free_mb} MB free (threshold: ${MIN_FREE_MEM_MB} MB) — may OOM"
    else
        echo "  ✅ GPU $gpu_id: ${free_mb} MB free — OK"
    fi
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ HELPER: send notification
# └─────────────────────────────────────────────────────────────────────────────
notify() {
    local subject="$1" body="$2"
    if [[ -n "$NOTIFY_WEBHOOK" ]] && command -v curl &>/dev/null; then
        curl -s -X POST "$NOTIFY_WEBHOOK" -H "Title: $subject" -d "$body" > /dev/null \
            && echo "  🔔 Webhook notified" || echo "  ⚠️  Webhook POST failed"
    fi
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ HELPER: record progress
# └─────────────────────────────────────────────────────────────────────────────
log_progress() {
    local status="$1" label="$2" method="$3" gpu_id="$4"
    echo "$(date '+%H:%M:%S')  [$status]  GPU$gpu_id  $method  $label" >> "$PROGRESS_FILE"
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ HELPER: check if already completed
# └─────────────────────────────────────────────────────────────────────────────
already_done() {
    local results_dir="$1" run_log="$1/run.log"
    [[ -f "$run_log" ]] && ! grep -q "Exit code [^0]" "$run_log" 2>/dev/null
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ WORKER: runs full zone + fault sweep for one method on one GPU
# └─────────────────────────────────────────────────────────────────────────────
run_method_on_gpu() {
    local method="$1"
    local gpu_id="$2"

    local results_root
    if [[ -n "${RESUME_FROM:-}" ]]; then
        local candidate
        candidate=$(find "$PROJECT_ROOT/results" -maxdepth 1 -type d \
                    -name "baselines_sweep_${DATASET}_*" 2>/dev/null | sort | tail -1)
        results_root="${candidate:-$PROJECT_ROOT/results/baselines_sweep_${DATASET}_${TIMESTAMP}}"
        echo "  [GPU $gpu_id | $method] Resuming from: $results_root"
    else
        results_root="$PROJECT_ROOT/results/baselines_sweep_${DATASET}_${TIMESTAMP}"
    fi

    local log_root="$PROJECT_ROOT/logs/baselines_sweep_${DATASET}_${TIMESTAMP}"
    local checkpoint_root="$PROJECT_ROOT/checkpoints/baselines_sweep_${DATASET}_${TIMESTAMP}"
    mkdir -p "$results_root" "$log_root" "$checkpoint_root"

    local max_samples="${DATASET_MAX_SAMPLES[$DATASET]:-50000}"
    local total_zone=${#ZONE_CONFIGS[@]}
    local total_fault=${#FAULT_CONFIGS[@]}
    local total=$(( total_zone + total_fault ))
    local idx=0 done_count=0 skip_count=0 fail_count=0

    echo "════════════════════════════════════════════════════════════"
    echo "  GPU $gpu_id — Method: $method  Dataset: $DATASET"
    echo "  Zone configs  : $total_zone"
    echo "  Fault configs : $total_fault"
    echo "  Total runs    : $total"
    echo "  Results       : $results_root"
    echo "════════════════════════════════════════════════════════════"

    _run_one() {
        local results_dir="$1/method_${method}"
        local num_devices="$2" num_zones="$3" min_zone="$4" max_zone="$5"
        local dev_fail="$6" zone_fail="$7" enable_failure="$8"
        local label
        label="$(basename "$1")__${method}"

        if already_done "$results_dir"; then
            echo "  [GPU $gpu_id | $method] ⏭  SKIP: $label"
            log_progress "SKIPPED" "$label" "$method" "$gpu_id"
            skip_count=$(( skip_count + 1 ))
            return 0
        fi

        local exp_log_dir="$log_root/$(basename "$1")/${method}"
        local checkpoint_dir="$checkpoint_root/$(basename "$1")/${method}"
        mkdir -p "$results_dir" "$exp_log_dir" "$checkpoint_dir"

        local cmd=(
            "$PYTHON_BIN" "$PROJECT_ROOT/main.py"
            --dataset                "$DATASET"
            --max_samples            "$max_samples"
            --num_devices            "$num_devices"
            --num_zones              "$num_zones"
            --min_zone_size          "$min_zone"
            --max_zone_size          "$max_zone"
            --num_rounds             "$NUM_ROUNDS"
            --local_epochs           "$LOCAL_EPOCHS"
            --eval_every             "$EVAL_EVERY"
            --learning_rate          "$LEARNING_RATE"
            --batch_size             "$BATCH_SIZE"
            --intra_zone_alpha       "$INTRA_ZONE_ALPHA"
            --inter_zone_alpha       "$INTER_ZONE_ALPHA"
            --compression_rate       "$COMPRESSION_RATE"
            --spatial_weight         "$SPATIAL_WEIGHT"
            --data_weight            "$DATA_WEIGHT"
            --network_weight         "$NETWORK_WEIGHT"
            --spatial_regularization "$SPATIAL_REGULARIZATION"
            --correlation_threshold  "$CORRELATION_THRESHOLD"
            --device                 "cuda"
            --random_seed            "$RANDOM_SEED"
            --log_dir                "$exp_log_dir"
            --results_dir            "$results_dir"
            --checkpoint_dir         "$checkpoint_dir"
            --baselines_only
            --run_baselines
            --baseline_methods       "$method"
            --enable_early_stopping
            --early_stopping_patience 10
            --save_results
            --ifca_k                 "$num_zones"
        )

        if [[ "$enable_failure" == "true" ]]; then
            cmd+=(
                --enable_failure
                --device_failure_probability "$dev_fail"
                --zone_failure_probability   "$zone_fail"
            )
        fi

        echo "  [GPU $gpu_id | $method] ▶  $label"
        if [[ "${DRY_RUN:-false}" == "true" ]]; then
            echo "    CMD: CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"
            log_progress "DRY_RUN" "$label" "$method" "$gpu_id"
            return 0
        fi

        CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee "$results_dir/run.log"
        local exit_code=${PIPESTATUS[0]}
        if [[ $exit_code -ne 0 ]]; then
            echo "  ⚠️  [GPU $gpu_id | $method] exit $exit_code — $results_dir/run.log"
            log_progress "FAILED" "$label" "$method" "$gpu_id"
            fail_count=$(( fail_count + 1 ))
        else
            log_progress "DONE" "$label" "$method" "$gpu_id"
            done_count=$(( done_count + 1 ))
        fi
        return $exit_code
    }

    # ── zone sweep ───────────────────────────────────────────────────────────
    echo ""
    echo "  ── Zone sweep ($total_zone configs) ──"
    for cfg in "${ZONE_CONFIGS[@]}"; do
        IFS=':' read -r NUM_DEV NUM_Z MIN_Z MAX_Z LABEL <<< "$cfg"
        idx=$(( idx + 1 ))
        echo ""
        echo "  [$idx/$total] zones_${LABEL}  (${NUM_DEV} devices, ${NUM_Z} zones)"
        _run_one "$results_root/zones_${LABEL}" \
                 "$NUM_DEV" "$NUM_Z" "$MIN_Z" "$MAX_Z" \
                 "0.00" "0.00" "false" || true
    done

    # ── fault sweep ──────────────────────────────────────────────────────────
    echo ""
    echo "  ── Fault sweep ($total_fault configs) ──"
    for cfg in "${FAULT_CONFIGS[@]}"; do
        IFS=':' read -r DEV_F ZONE_F LABEL <<< "$cfg"
        idx=$(( idx + 1 ))
        local enable_fail="true"
        [[ "$DEV_F" == "0.00" && "$ZONE_F" == "0.00" ]] && enable_fail="false"
        echo ""
        echo "  [$idx/$total] fault_${LABEL}  (dev=${DEV_F}, zone=${ZONE_F})"
        _run_one "$results_root/fault_${LABEL}" \
                 "50" "5" "4" "15" \
                 "$DEV_F" "$ZONE_F" "$enable_fail" || true
    done

    echo ""
    echo "  ✅ GPU $gpu_id ($method) complete — done=$done_count skipped=$skip_count failed=$fail_count"
    [[ $fail_count -eq 0 ]]
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ GPU MEMORY CHECK
# └─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  GPU memory check (threshold: ${MIN_FREE_MEM_MB} MB free)"
echo "════════════════════════════════════════════════════════════"
for entry in "${METHOD_GPU_MAP[@]}"; do
    IFS=':' read -r method gpu <<< "$entry"
    check_gpu_memory "$gpu"
done
echo ""

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SUMMARY
# └─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  ContinuumFL — Baselines sweep (one method per GPU)"
echo "  Dataset  : $DATASET"
for entry in "${METHOD_GPU_MAP[@]}"; do
    IFS=':' read -r method gpu <<< "$entry"
    echo "  GPU $gpu → $method"
done
[[ -n "${RESUME_FROM:-}" ]] && echo "  RESUME   : ON"
[[ -n "$NOTIFY_WEBHOOK" ]]  && echo "  Webhook  : $NOTIFY_WEBHOOK"
echo "  Progress : $PROGRESS_FILE"
echo "  Logs     : $LOG_ROOT/"
echo "════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────────
# │ LAUNCH ONE BACKGROUND WORKER PER METHOD/GPU
# └─────────────────────────────────────────────────────────────────────────────
PIDS=()
for entry in "${METHOD_GPU_MAP[@]}"; do
    IFS=':' read -r method gpu <<< "$entry"
    run_method_on_gpu "$method" "$gpu" \
        > >(tee "$LOG_ROOT/gpu${gpu}_${method}.log") 2>&1 &
    PIDS+=($!)
    echo "  Launched GPU $gpu → $method  (PID ${PIDS[-1]})"
done

echo ""
echo "  All workers running in parallel."
echo "  Live logs  : tail -f $LOG_ROOT/gpu*_*.log"
echo "  Progress   : watch cat $PROGRESS_FILE"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────────
# │ WAIT & REPORT
# └─────────────────────────────────────────────────────────────────────────────
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  ⚠️  PID $pid failed"; FAILED=$(( FAILED + 1 )); }
done

FINISH_TIME="$(date)"
{
    echo "─────────────────────────────────────────"
    echo "Finished : $FINISH_TIME"
    echo "Status   : $( [[ $FAILED -eq 0 ]] && echo 'ALL DONE ✅' || echo "$FAILED WORKER(S) FAILED ⚠️" )"
} >> "$PROGRESS_FILE"

echo ""
echo "════════════════════════════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
    SUMMARY="✅ All methods finished on $DATASET at $FINISH_TIME"
    echo "  $SUMMARY"
    notify "ContinuumFL sweep done ✅" "$SUMMARY"$'\n\n'"$(cat "$PROGRESS_FILE")"
else
    SUMMARY="⚠️ $FAILED worker(s) failed — check $LOG_ROOT/"
    echo "  $SUMMARY"
    notify "ContinuumFL sweep FAILED ⚠️" "$SUMMARY"$'\n\n'"$(cat "$PROGRESS_FILE")"
fi
echo "  Progress → $PROGRESS_FILE"
echo "════════════════════════════════════════════════════════════"
