#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — APCfl sweep (no SLURM)
#
#  Runs a single baseline method across:
#    • 4 zone configs  (clients × zones)
#    • 4 fault scenarios
#
#  Usage:
#    bash scripts/run_apcfl_sweep.sh                        # run APCfl (default)
#    METHOD=IFCA bash scripts/run_apcfl_sweep.sh            # run a different method
#    DRY_RUN=true bash scripts/run_apcfl_sweep.sh           # print commands only
#    DATASET=femnist bash scripts/run_apcfl_sweep.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DATASET — one line to change:
# │   ucihar | femnist | cifar100 | shakespeare | speechcommands
# └─────────────────────────────────────────────────────────────────────────────
DATASET="${DATASET:-speechcommands}"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ BASELINE METHOD — override with METHOD=<name>
# │   IFCA | APCfl | GeoFL
# └─────────────────────────────────────────────────────────────────────────────
METHOD="${METHOD:-APCfl}"
BASELINE_METHODS=("$METHOD")

# ┌─────────────────────────────────────────────────────────────────────────────
# │ FIXED PARAMETERS (shared across all runs)
# └─────────────────────────────────────────────────────────────────────────────
NUM_ROUNDS="${NUM_ROUNDS:-200}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LEARNING_RATE=0.001
BATCH_SIZE=16
INTRA_ZONE_ALPHA=100
INTER_ZONE_ALPHA=5.0
COMPRESSION_RATE=0.10
SPATIAL_WEIGHT=0.4
DATA_WEIGHT=0.4
NETWORK_WEIGHT=0.2
SPATIAL_REGULARIZATION=0.05
CORRELATION_THRESHOLD=0.05
RANDOM_SEED=42
DEVICE="cuda"
MAX_SAMPLES=70000

# ┌─────────────────────────────────────────────────────────────────────────────
# │ ZONE CONFIGS
# │   Format: "NUM_DEVICES:NUM_ZONES:MIN_ZONE_SIZE:MAX_ZONE_SIZE:LABEL"
# └─────────────────────────────────────────────────────────────────────────────
ZONE_CONFIGS=(
    "10:2:3:6:10c_2z"
    "10:5:1:4:10c_5z"
    "50:5:4:15:50c_5z"
    "50:10:3:8:50c_10z"
    # "50:25:1:4:50c_25z"
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ FAULT SCENARIOS
# │   Format: "DEVICE_FAIL:ZONE_FAIL:LABEL"
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
RUN_NAME="${METHOD}_sweep_${DATASET}_$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="$PROJECT_ROOT/results/$RUN_NAME"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_NAME"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT"

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
# │ HELPER: run one baseline experiment
# │   run_baselines <results_dir> <devices> <zones> <min_z> <max_z>
# │                 <dev_fail> <zone_fail> [enable_failure]
# └─────────────────────────────────────────────────────────────────────────────
run_baselines() {
    local results_dir="$1"
    local num_devices="$2"
    local num_zones="$3"
    local min_zone="$4"
    local max_zone="$5"
    local dev_fail="$6"
    local zone_fail="$7"
    local enable_failure="$8"

    local log_dir="$LOG_ROOT/$(basename "$results_dir")"
    local checkpoint_dir="$CHECKPOINT_ROOT/$(basename "$results_dir")"
    mkdir -p "$results_dir" "$log_dir" "$checkpoint_dir"

    local cmd=(
        "$PYTHON_BIN" "$PROJECT_ROOT/main.py"
        --dataset              "$DATASET"
        --max_samples          "$MAX_SAMPLES"
        --num_devices          "$num_devices"
        --num_zones            "$num_zones"
        --min_zone_size        "$min_zone"
        --max_zone_size        "$max_zone"
        --num_rounds           "$NUM_ROUNDS"
        --local_epochs         "$LOCAL_EPOCHS"
        --eval_every           "$EVAL_EVERY"
        --learning_rate        "$LEARNING_RATE"
        --batch_size           "$BATCH_SIZE"
        --intra_zone_alpha     "$INTRA_ZONE_ALPHA"
        --inter_zone_alpha     "$INTER_ZONE_ALPHA"
        --compression_rate     "$COMPRESSION_RATE"
        --spatial_weight       "$SPATIAL_WEIGHT"
        --data_weight          "$DATA_WEIGHT"
        --network_weight       "$NETWORK_WEIGHT"
        --spatial_regularization "$SPATIAL_REGULARIZATION"
        --correlation_threshold  "$CORRELATION_THRESHOLD"
        --device               "$DEVICE"
        --random_seed          "$RANDOM_SEED"
        --log_dir              "$log_dir"
        --results_dir          "$results_dir"
        --checkpoint_dir       "$checkpoint_dir"
        --baselines_only
        --run_baselines
        --baseline_methods     "${BASELINE_METHODS[@]}"
        --enable_early_stopping
        --early_stopping_patience 10
        --save_results
        --ifca_k               "$num_zones"
    )

    if [[ "$enable_failure" == "true" ]]; then
        cmd+=(
            --enable_failure
            --device_failure_probability "$dev_fail"
            --zone_failure_probability   "$zone_fail"
        )
    fi

    echo "    CMD: ${cmd[*]}"
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "    [DRY_RUN] skipping"
        return 0
    fi

    "${cmd[@]}" 2>&1 | tee "$results_dir/run.log"
    local exit_code=${PIPESTATUS[0]}
    [[ $exit_code -ne 0 ]] && echo "  ⚠️  Exit code $exit_code — check $results_dir/run.log"
    return $exit_code
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ COUNTS
# └─────────────────────────────────────────────────────────────────────────────
TOTAL_ZONE=${#ZONE_CONFIGS[@]}
TOTAL_FAULT=${#FAULT_CONFIGS[@]}
TOTAL=$(( TOTAL_ZONE + TOTAL_FAULT ))
IDX=0

echo "════════════════════════════════════════════════════════════"
echo "  ContinuumFL — ${METHOD} sweep (no SLURM)"
echo "  Dataset  : $DATASET"
echo "  Method   : $METHOD"
echo "  Zone configs  : $TOTAL_ZONE"
echo "  Fault configs : $TOTAL_FAULT"
echo "  Total runs    : $TOTAL"
echo "  Results  : $RESULTS_ROOT"
echo "════════════════════════════════════════════════════════════"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PART 1 — ZONE SWEEP (fault-free)
# └─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART 1: Zone sweep (fault-free, $TOTAL_ZONE configs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for cfg in "${ZONE_CONFIGS[@]}"; do
    IFS=':' read -r NUM_DEV NUM_Z MIN_Z MAX_Z LABEL <<< "$cfg"
    IDX=$(( IDX + 1 ))
    RUN_DIR="$RESULTS_ROOT/zones_${LABEL}"
    echo ""
    echo "  [$IDX/$TOTAL] zones_${LABEL}  (${NUM_DEV} devices, ${NUM_Z} zones)"
    echo "  → $RUN_DIR"
    run_baselines "$RUN_DIR" "$NUM_DEV" "$NUM_Z" "$MIN_Z" "$MAX_Z" \
                  "0.00" "0.00" "false" || true
done

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PART 2 — FAULT SWEEP (fixed: 50 devices, 5 zones — best config)
# └─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART 2: Fault sweep (50 devices, 5 zones, $TOTAL_FAULT configs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for cfg in "${FAULT_CONFIGS[@]}"; do
    IFS=':' read -r DEV_F ZONE_F LABEL <<< "$cfg"
    IDX=$(( IDX + 1 ))
    RUN_DIR="$RESULTS_ROOT/fault_${LABEL}"

    if [[ "$DEV_F" == "0.00" && "$ZONE_F" == "0.00" ]]; then
        ENABLE_FAIL="false"
    else
        ENABLE_FAIL="true"
    fi

    echo ""
    echo "  [$IDX/$TOTAL] fault_${LABEL}  (dev_fail=${DEV_F}, zone_fail=${ZONE_F})"
    echo "  → $RUN_DIR"
    run_baselines "$RUN_DIR" "50" "5" "4" "15" \
                  "$DEV_F" "$ZONE_F" "$ENABLE_FAIL" || true
done

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DONE
# └─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Sweep complete."
echo "  Results → $RESULTS_ROOT"
echo "════════════════════════════════════════════════════════════"
