#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Retry failed baseline runs
#
#  Re-runs only the experiments that were marked [FAILED] in a prior sweep.
#  Automatically detects the most-recent baselines_sweep results dir for the
#  given dataset, or you can point it at one explicitly:
#
#  Usage:
#    bash scripts/retry_failed_baselines.sh cifar100
#    RESULTS_DIR=results/baselines_sweep_cifar100_20260611_092143 \
#        bash scripts/retry_failed_baselines.sh cifar100
#    GPU_ClusterFL=1 bash scripts/retry_failed_baselines.sh cifar100
#    DRY_RUN=true bash scripts/retry_failed_baselines.sh cifar100
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── open-file limit ──────────────────────────────────────────────────────────
_soft=$(ulimit -Sn 2>/dev/null || echo 1024)
_hard=$(ulimit -Hn 2>/dev/null || echo 1024)
_tgt=65536
(( _tgt > _hard )) && _tgt=$_hard
(( _tgt > _soft )) && ulimit -Sn "$_tgt" 2>/dev/null || true
unset _soft _hard _tgt

# ── dataset ──────────────────────────────────────────────────────────────────
DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "Usage: bash scripts/retry_failed_baselines.sh <dataset>" >&2
    echo "  datasets: femnist cifar100 shakespeare speechcommands" >&2
    exit 1
fi

# ── GPU assignment (override via env) ────────────────────────────────────────
GPU_ClusterFL="${GPU_ClusterFL:-0}"
GPU_IFCA="${GPU_IFCA:-1}"
GPU_GeoFL="${GPU_GeoFL:-2}"
GPU_SnapCFL="${GPU_SnapCFL:-3}"

declare -A METHOD_GPU=(
    [ClusterFL]="$GPU_ClusterFL"
    [IFCA]="$GPU_IFCA"
    [GeoFL]="$GPU_GeoFL"
    [SnapCFL]="$GPU_SnapCFL"
)

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Find the results dir to retry (most recent matching sweep, or explicit)
if [[ -n "${RESULTS_DIR:-}" ]]; then
    SWEEP_RESULTS="$RESULTS_DIR"
else
    SWEEP_RESULTS=$(find "$PROJECT_ROOT/results" -maxdepth 1 -type d \
        -name "baselines_sweep_${DATASET}_*" 2>/dev/null | sort | tail -1)
fi

if [[ -z "$SWEEP_RESULTS" || ! -d "$SWEEP_RESULTS" ]]; then
    echo "❌  No results dir found for dataset '$DATASET'." >&2
    echo "    Set RESULTS_DIR=<path> to point at the sweep results." >&2
    exit 1
fi

LOG_ROOT="$PROJECT_ROOT/logs/retry_${DATASET}_${TIMESTAMP}"
mkdir -p "$LOG_ROOT"

PROGRESS_FILE="$LOG_ROOT/progress.txt"
{
    echo "ContinuumFL — Retry-failed progress"
    echo "Dataset      : $DATASET"
    echo "Sweep results: $SWEEP_RESULTS"
    echo "Started      : $(date)"
    echo "─────────────────────────────────────────"
} > "$PROGRESS_FILE"

# ── fixed parameters (must match original sweep) ─────────────────────────────
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

# ── zone + fault config tables (same as original) ────────────────────────────
declare -A ZONE_CONFIG_MAP=(
    [zones_10c_2z]="10:2:3:6"
    [zones_10c_5z]="10:5:1:4"
    [zones_50c_5z]="50:5:4:15"
    [zones_50c_10z]="50:10:3:8"
)

declare -A FAULT_CONFIG_MAP=(
    [fault_fault_free]="0.00:0.00:false"
    [fault_dev_low]="0.05:0.00:true"
    [fault_dev_high]="0.20:0.00:true"
    [fault_dev_low__zone_low]="0.05:0.02:true"
    [fault_dev_high__zone_high]="0.20:0.10:true"
)

# ── python ────────────────────────────────────────────────────────────────────
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

# ── helpers ───────────────────────────────────────────────────────────────────
log_progress() {
    echo "$(date '+%H:%M:%S')  [$1]  GPU${4}  ${3}  ${2}" >> "$PROGRESS_FILE"
}

run_one() {
    local method="$1" config_label="$2"   # e.g. ClusterFL  zones_50c_5z
    local gpu_id="${METHOD_GPU[$method]}"
    local max_samples="${DATASET_MAX_SAMPLES[$DATASET]:-50000}"
    local results_dir="$SWEEP_RESULTS/${config_label}/method_${method}"

    local checkpoint_dir="$PROJECT_ROOT/checkpoints/retry_${DATASET}_${TIMESTAMP}/${config_label}/${method}"
    local exp_log_dir="$LOG_ROOT/${config_label}/${method}"
    mkdir -p "$results_dir" "$exp_log_dir" "$checkpoint_dir"

    # ── decode zone or fault config ──────────────────────────────────────────
    local num_devices num_zones min_zone max_zone dev_fail zone_fail enable_fail

    if [[ -v ZONE_CONFIG_MAP[$config_label] ]]; then
        IFS=':' read -r num_devices num_zones min_zone max_zone \
            <<< "${ZONE_CONFIG_MAP[$config_label]}"
        dev_fail="0.00"; zone_fail="0.00"; enable_fail="false"
    elif [[ -v FAULT_CONFIG_MAP[$config_label] ]]; then
        IFS=':' read -r dev_fail zone_fail enable_fail \
            <<< "${FAULT_CONFIG_MAP[$config_label]}"
        # fault sweep always uses the 50c_5z topology
        num_devices=50; num_zones=5; min_zone=4; max_zone=15
    else
        echo "  ⚠️  Unknown config label: $config_label — skipping" >&2
        return 1
    fi

    local label="${config_label}__${method}"
    echo ""
    echo "  ▶  GPU $gpu_id | $method | $label"

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

    if [[ "$enable_fail" == "true" ]]; then
        cmd+=(
            --enable_failure
            --device_failure_probability "$dev_fail"
            --zone_failure_probability   "$zone_fail"
        )
    fi

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "    CMD: CUDA_VISIBLE_DEVICES=$gpu_id ${cmd[*]}"
        log_progress "DRY_RUN" "$label" "$method" "$gpu_id"
        return 0
    fi

    # Remove stale run.log so the completion-check doesn't skip this run
    rm -f "$results_dir/run.log"

    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee "$results_dir/run.log"
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -ne 0 ]]; then
        echo "  ⚠️  FAILED (exit $exit_code) — see $results_dir/run.log"
        log_progress "FAILED" "$label" "$method" "$gpu_id"
    else
        log_progress "DONE" "$label" "$method" "$gpu_id"
    fi
    return $exit_code
}

# ═══════════════════════════════════════════════════════════════════════════════
#  FAILED RUNS TO RETRY
#  Derived from the progress.txt of the June 11–15 sweep (cifar100, ClusterFL).
#  Edit this list if you're retrying a different dataset or different failures.
# ═══════════════════════════════════════════════════════════════════════════════
declare -a RETRY_JOBS=(
    # format: "METHOD:CONFIG_LABEL"
    "ClusterFL:zones_50c_5z"
    "ClusterFL:zones_50c_10z"
    "ClusterFL:fault_fault_free"
    "ClusterFL:fault_dev_low"
    "ClusterFL:fault_dev_high"
    "ClusterFL:fault_dev_low__zone_low"
    "ClusterFL:fault_dev_high__zone_high"
)

# ── summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  ContinuumFL — Retry failed baselines"
echo "  Dataset      : $DATASET"
echo "  Sweep results: $SWEEP_RESULTS"
echo "  Jobs to retry: ${#RETRY_JOBS[@]}"
for job in "${RETRY_JOBS[@]}"; do
    IFS=':' read -r m c <<< "$job"
    echo "    GPU ${METHOD_GPU[$m]} → $m  $c"
done
[[ "${DRY_RUN:-false}" == "true" ]] && echo "  DRY_RUN      : ON"
echo "  Progress     : $PROGRESS_FILE"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── group jobs by GPU so each GPU's runs stay sequential ─────────────────────
declare -A GPU_JOBS=()
for job in "${RETRY_JOBS[@]}"; do
    IFS=':' read -r m c <<< "$job"
    g="${METHOD_GPU[$m]}"
    GPU_JOBS[$g]="${GPU_JOBS[$g]:-} $job"
done

run_gpu_queue() {
    local gpu_id="$1"; shift
    local jobs=("$@")
    local done_count=0 fail_count=0

    for job in "${jobs[@]}"; do
        [[ -z "$job" ]] && continue
        IFS=':' read -r m c <<< "$job"
        run_one "$m" "$c" \
            > >(tee "$LOG_ROOT/gpu${gpu_id}_${m}_${c}.log") 2>&1 \
            && (( done_count++ )) || (( fail_count++ )) || true
    done

    echo ""
    echo "  GPU $gpu_id queue finished — done=$done_count failed=$fail_count"
    [[ $fail_count -eq 0 ]]
}

PIDS=()
for gpu_id in "${!GPU_JOBS[@]}"; do
    # shellcheck disable=SC2206
    jobs_arr=(${GPU_JOBS[$gpu_id]})
    run_gpu_queue "$gpu_id" "${jobs_arr[@]}" &
    PIDS+=($!)
    echo "  Launched GPU $gpu_id queue (PID ${PIDS[-1]})"
done

echo ""
echo "  Live logs : tail -f $LOG_ROOT/gpu*.log"
echo "  Progress  : watch cat $PROGRESS_FILE"
echo ""

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  ⚠️  PID $pid failed"; FAILED=$(( FAILED + 1 )); }
done

FINISH_TIME="$(date)"
{
    echo "─────────────────────────────────────────"
    echo "Finished : $FINISH_TIME"
    echo "Status   : $( [[ $FAILED -eq 0 ]] && echo 'ALL DONE ✅' || echo "$FAILED GPU QUEUE(S) FAILED ⚠️" )"
} >> "$PROGRESS_FILE"

echo ""
echo "════════════════════════════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
    echo "  ✅ All retries finished for $DATASET at $FINISH_TIME"
else
    echo "  ⚠️  $FAILED queue(s) failed — check $LOG_ROOT/"
fi
echo "  Progress → $PROGRESS_FILE"
echo "════════════════════════════════════════════════════════════"
