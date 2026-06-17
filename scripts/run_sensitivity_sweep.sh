#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Sensitivity Analysis Sweep (no SLURM)
#
#  One-at-a-time (OAT) sweep: varies one hyperparameter at a time while
#  holding all others at the baseline default.
#
#  Each run → its own folder:
#    results/<RUN_NAME>/sensitivity_<PARAM>/<VALUE>/
#
#  Usage:
#    bash scripts/run_sensitivity_sweep.sh                        # all params
#    PARAM=learning_rate bash scripts/run_sensitivity_sweep.sh    # one param
#    DRY_RUN=true bash scripts/run_sensitivity_sweep.sh           # dry run
#    DATASET=femnist bash scripts/run_sensitivity_sweep.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DATASET
# │   ucihar | femnist | cifar100 | shakespeare | speechcommands
# └─────────────────────────────────────────────────────────────────────────────
DATASET="${DATASET:-ucihar}"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ BASELINE (default) CONFIG
# │   All sweeps hold these fixed except the one parameter being varied.
# └─────────────────────────────────────────────────────────────────────────────
BASE_NUM_DEVICES=50
BASE_NUM_ZONES=5
BASE_MIN_ZONE=4
BASE_MAX_ZONE=15
BASE_NUM_ROUNDS=200
BASE_LOCAL_EPOCHS=5
BASE_EVAL_EVERY=5
BASE_LEARNING_RATE=0.001
BASE_BATCH_SIZE=16
BASE_INTRA_ZONE_ALPHA=100
BASE_INTER_ZONE_ALPHA=5.0
BASE_COMPRESSION_RATE=0.10
BASE_SPATIAL_WEIGHT=0.4
BASE_DATA_WEIGHT=0.4
BASE_NETWORK_WEIGHT=0.2
BASE_SPATIAL_REG=0.05
BASE_CORRELATION_THRESHOLD=0.05
BASE_MAX_SAMPLES=70000
BASE_RANDOM_SEED=42
DEVICE="cuda"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ HYPERPARAMETER SWEEP GRIDS
# │   Format: "PARAM_KEY:val1,val2,val3,..."
# │
# │   learning_rate           — optimiser step size
# │   batch_size              — local training batch size
# │   local_epochs            — local SGD epochs per round
# │   spatial_regularization  — spatial penalty weight (regularisation)
# │   compression_rate        — model compression fraction sent to server
# │   spatial_weight          — spatial similarity weight in aggregation
# │   data_weight             — data similarity weight in aggregation
# │   network_weight          — network similarity weight in aggregation
# │   intra_zone_alpha        — Dirichlet α controlling intra-zone non-IID
# │   inter_zone_alpha        — Dirichlet α controlling inter-zone non-IID
# │   correlation_threshold   — minimum correlation to form a spatial edge
# └─────────────────────────────────────────────────────────────────────────────
SWEEP_PARAMS=(
    "learning_rate:0.0001,0.0005,0.001,0.005,0.01"
    "batch_size:8,16,32,64,128"
    "local_epochs:1,3,5,10,20"
    "spatial_regularization:0.0,0.01,0.05,0.1,0.5"
    "compression_rate:0.05,0.10,0.20,0.40,0.60"
    "spatial_weight:0.1,0.2,0.4,0.6,0.8"
    "data_weight:0.1,0.2,0.4,0.6,0.8"
    "network_weight:0.1,0.2,0.4,0.6,0.8"
    "intra_zone_alpha:10,50,100,500,1000"
    "inter_zone_alpha:1.0,2.0,5.0,10.0,50.0"
    "correlation_threshold:0.01,0.05,0.10,0.20,0.50"
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ OPTIONAL: restrict to a single parameter
# │   e.g.  PARAM=learning_rate bash scripts/run_sensitivity_sweep.sh
# └─────────────────────────────────────────────────────────────────────────────
PARAM_FILTER="${PARAM:-}"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PATHS
# └─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="sensitivity_${DATASET}_$(date +%Y%m%d_%H%M%S)"
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
# │ HELPER: run one sensitivity experiment
# │   run_sensitivity <results_dir> <param_name> <param_value>
# └─────────────────────────────────────────────────────────────────────────────
run_sensitivity() {
    local results_dir="$1"
    local param_name="$2"
    local param_value="$3"

    local log_dir="$LOG_ROOT/sensitivity_${param_name}/$(basename "$results_dir")"
    local checkpoint_dir="$CHECKPOINT_ROOT/sensitivity_${param_name}/$(basename "$results_dir")"
    mkdir -p "$results_dir" "$log_dir" "$checkpoint_dir"

    # Start with baseline defaults, then override the swept parameter
    local lr="$BASE_LEARNING_RATE"
    local bs="$BASE_BATCH_SIZE"
    local le="$BASE_LOCAL_EPOCHS"
    local sreg="$BASE_SPATIAL_REG"
    local cr="$BASE_COMPRESSION_RATE"
    local sw="$BASE_SPATIAL_WEIGHT"
    local dw="$BASE_DATA_WEIGHT"
    local nw="$BASE_NETWORK_WEIGHT"
    local iza="$BASE_INTRA_ZONE_ALPHA"
    local ieza="$BASE_INTER_ZONE_ALPHA"
    local ct="$BASE_CORRELATION_THRESHOLD"

    case "$param_name" in
        learning_rate)           lr="$param_value"   ;;
        batch_size)              bs="$param_value"   ;;
        local_epochs)            le="$param_value"   ;;
        spatial_regularization)  sreg="$param_value" ;;
        compression_rate)        cr="$param_value"   ;;
        spatial_weight)          sw="$param_value"   ;;
        data_weight)             dw="$param_value"   ;;
        network_weight)          nw="$param_value"   ;;
        intra_zone_alpha)        iza="$param_value"  ;;
        inter_zone_alpha)        ieza="$param_value" ;;
        correlation_threshold)   ct="$param_value"   ;;
        *)
            echo "  ⚠️  Unknown param '$param_name' — skipping" >&2
            return 1
            ;;
    esac

    local cmd=(
        "$PYTHON_BIN" "$PROJECT_ROOT/main.py"
        --dataset                  "$DATASET"
        --max_samples              "$BASE_MAX_SAMPLES"
        --num_devices              "$BASE_NUM_DEVICES"
        --num_zones                "$BASE_NUM_ZONES"
        --min_zone_size            "$BASE_MIN_ZONE"
        --max_zone_size            "$BASE_MAX_ZONE"
        --num_rounds               "$BASE_NUM_ROUNDS"
        --local_epochs             "$le"
        --eval_every               "$BASE_EVAL_EVERY"
        --learning_rate            "$lr"
        --batch_size               "$bs"
        --intra_zone_alpha         "$iza"
        --inter_zone_alpha         "$ieza"
        --compression_rate         "$cr"
        --spatial_weight           "$sw"
        --data_weight              "$dw"
        --network_weight           "$nw"
        --spatial_regularization   "$sreg"
        --correlation_threshold    "$ct"
        --device                   "$DEVICE"
        --random_seed              "$BASE_RANDOM_SEED"
        --log_dir                  "$log_dir"
        --results_dir              "$results_dir"
        --checkpoint_dir           "$checkpoint_dir"
        --enable_early_stopping
        --early_stopping_patience  20
        --save_results
    )

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
# │ COUNT TOTAL RUNS
# └─────────────────────────────────────────────────────────────────────────────
TOTAL=0
for entry in "${SWEEP_PARAMS[@]}"; do
    param="${entry%%:*}"
    [[ -n "$PARAM_FILTER" && "$param" != "$PARAM_FILTER" ]] && continue
    values="${entry#*:}"
    IFS=',' read -ra vals <<< "$values"
    TOTAL=$(( TOTAL + ${#vals[@]} ))
done

IDX=0

echo "════════════════════════════════════════════════════════════"
echo "  ContinuumFL — Sensitivity Analysis (OAT)"
echo "  Dataset   : $DATASET"
echo "  Baseline  : ${BASE_NUM_DEVICES} devices | ${BASE_NUM_ZONES} zones"
echo "              lr=${BASE_LEARNING_RATE} | bs=${BASE_BATCH_SIZE} | epochs=${BASE_LOCAL_EPOCHS}"
echo "              spatial_reg=${BASE_SPATIAL_REG} | compression=${BASE_COMPRESSION_RATE}"
echo "  Param filter: ${PARAM_FILTER:-none (all)}"
echo "  Total runs: $TOTAL"
echo "  Results   : $RESULTS_ROOT"
echo "════════════════════════════════════════════════════════════"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ MAIN SWEEP LOOP
# └─────────────────────────────────────────────────────────────────────────────
for entry in "${SWEEP_PARAMS[@]}"; do
    param="${entry%%:*}"
    values="${entry#*:}"

    [[ -n "$PARAM_FILTER" && "$param" != "$PARAM_FILTER" ]] && continue

    IFS=',' read -ra vals <<< "$values"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PARAM: --${param}   values: ${vals[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for val in "${vals[@]}"; do
        IDX=$(( IDX + 1 ))
        # Sanitise value for directory name (dots → underscore, minus → neg)
        val_safe="${val//\./_}"
        val_safe="${val_safe//-/neg}"
        RUN_DIR="$RESULTS_ROOT/sensitivity_${param}/${val_safe}"

        echo ""
        echo "  [$IDX/$TOTAL]  --${param} ${val}"
        echo "  → $RUN_DIR"

        run_sensitivity "$RUN_DIR" "$param" "$val" || true
    done
done

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DONE
# └─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Sensitivity sweep complete."
echo "  Results → $RESULTS_ROOT"
echo ""
echo "  Folder structure:"
echo "    $RESULTS_ROOT/"
echo "    ├── sensitivity_learning_rate/0_0001/"
echo "    │                            0_0005/"
echo "    │                            0_001/   ← baseline value"
echo "    │                            ..."
echo "    ├── sensitivity_batch_size/8/"
echo "    │                         16/         ← baseline value"
echo "    │                         ..."
echo "    ├── sensitivity_spatial_regularization/..."
echo "    ├── sensitivity_compression_rate/..."
echo "    ├── sensitivity_spatial_weight/..."
echo "    ├── sensitivity_data_weight/..."
echo "    ├── sensitivity_network_weight/..."
echo "    ├── sensitivity_intra_zone_alpha/..."
echo "    ├── sensitivity_inter_zone_alpha/..."
echo "    └── sensitivity_correlation_threshold/..."
echo "════════════════════════════════════════════════════════════"
