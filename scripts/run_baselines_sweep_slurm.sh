#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Baselines sweep (Slurm)
#
#  Runs all baseline methods across:
#    • 5 zone configs  (clients × zones)
#    • 8 fault scenarios
#
#  Submit:    sbatch scripts/run_baselines_sweep_slurm.sh [DATASET]
#  Dry-run:   DRY_RUN=true bash scripts/run_baselines_sweep_slurm.sh [DATASET]
#  Local run: bash scripts/run_baselines_sweep_slurm.sh [DATASET]
#
#  Results layout:
#    results/<RUN_NAME>/zones_<label>/run.log
#    results/<RUN_NAME>/fault_<label>/run.log
# ═══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 1 — SLURM DIRECTIVES
# │ Change these to match your cluster partition and resource needs.
# └─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=CFL_baselines
#SBATCH --partition=IFItitan          # IFIall | IFIgpu2070 | IFIgpu2070S | IFItitan | IFIgpuL40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00            # baseline sweeps run long; adjust as needed
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.Younesi@uibk.ac.at
#SBATCH --account=DPS
#SBATCH --output=logs/slurm.%x.%j.out
#SBATCH --error=logs/slurm.%x.%j.err

set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 2 — DATASET
# │   ucihar | femnist | cifar100 | shakespeare | speechcommands
# └─────────────────────────────────────────────────────────────────────────────
DATASET="${1:-${DATASET:-speechcommands}}"
shift || true

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 3 — BASELINE METHODS & FIXED PARAMETERS
# └─────────────────────────────────────────────────────────────────────────────
BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL IFCA APCfl GeoFL SnapCFL)

NUM_ROUNDS=200
LOCAL_EPOCHS=5
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
DEVICE="cuda"
MAX_SAMPLES=50000

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 4 — SWEEP CONFIGS
# └─────────────────────────────────────────────────────────────────────────────
# Format: "NUM_DEVICES:NUM_ZONES:MIN_ZONE_SIZE:MAX_ZONE_SIZE:LABEL"
ZONE_CONFIGS=(
    "10:2:3:6:10c_2z"
    "10:5:1:4:10c_5z"
    "50:5:4:15:50c_5z"
    "50:10:3:8:50c_10z"
    # "50:25:1:4:50c_25z"
)

# Format: "DEVICE_FAIL:ZONE_FAIL:LABEL"
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
# │ SECTION 5 — PATHS
# └─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_NAME="${RUN_NAME:-baselines_sweep_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="$PROJECT_ROOT/results/$RUN_NAME"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_NAME"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 6 — ENVIRONMENT SETUP (Python + PyTorch)
# └─────────────────────────────────────────────────────────────────────────────
# Uncomment if your HPC requires module loads:
# module load python/3.11
# module load cuda/11.8

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    command -v python3 &>/dev/null && PYTHON_BIN="python3" || {
        echo "❌ Neither 'python' nor 'python3' found. Load the Python module first."
        exit 1
    }
fi

VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🔨 Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" --upgrade-deps
fi
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | tail -3

echo "🔍 Detecting GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU_TYPE="NVIDIA"
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5
elif command -v rocm-smi &>/dev/null; then
    GPU_TYPE="AMD"
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm5.7 2>&1 | tail -5
else
    GPU_TYPE="CPU"
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio 2>&1 | tail -5
fi

grep -v "^torch\|^#.*torch\|cuda\|rocm" "$PROJECT_ROOT/requirements.txt" \
    | "$PYTHON_BIN" -m pip install --quiet -r /dev/stdin 2>&1 || true

"$PYTHON_BIN" -c "import torch; print('   PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "✅ Environment ready (GPU: $GPU_TYPE)"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 7 — HELPER: run one baseline experiment
# │   run_baselines <results_dir> <devices> <zones> <min_z> <max_z>
# │                 <dev_fail> <zone_fail> <enable_failure>
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
        --dataset                "$DATASET"
        --max_samples            "$MAX_SAMPLES"
        --num_devices            "$num_devices"
        --num_zones              "$num_zones"
        --min_zone_size          "$min_zone"
        --max_zone_size          "$max_zone"
        --num_rounds             "$NUM_ROUNDS"
        --local_epochs           "$LOCAL_EPOCHS"
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
        --device                 "$DEVICE"
        --random_seed            "$RANDOM_SEED"
        --log_dir                "$log_dir"
        --results_dir            "$results_dir"
        --checkpoint_dir         "$checkpoint_dir"
        --baselines_only
        --run_baselines
        --baseline_methods       "${BASELINE_METHODS[@]}"
        --ifca_k                 "$num_zones"
        --save_results
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

    local run_log="$results_dir/run.log"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun "${cmd[@]}" 2>&1 | tee "$run_log"
    else
        "${cmd[@]}" 2>&1 | tee "$run_log"
    fi

    local exit_code=${PIPESTATUS[0]}
    [[ $exit_code -ne 0 ]] && echo "  ⚠️  Exit code $exit_code — check $run_log" || true
    return $exit_code
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 8 — RUN
# └─────────────────────────────────────────────────────────────────────────────
TOTAL_ZONE=${#ZONE_CONFIGS[@]}
TOTAL_FAULT=${#FAULT_CONFIGS[@]}
TOTAL=$(( TOTAL_ZONE + TOTAL_FAULT ))
IDX=0

echo "════════════════════════════════════════════════════════════════"
echo "  ContinuumFL — Baselines sweep (Slurm)"
echo "  Dataset       : $DATASET"
echo "  Methods       : ${BASELINE_METHODS[*]}"
echo "  Zone configs  : $TOTAL_ZONE"
echo "  Fault configs : $TOTAL_FAULT"
echo "  Total runs    : $TOTAL"
echo "  Slurm ID      : ${SLURM_JOB_ID:-local}"
echo "  Results       : $RESULTS_ROOT"
echo "════════════════════════════════════════════════════════════════"

# ── Part 1: Zone sweep (fault-free) ──────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART 1: Zone sweep (fault-free, $TOTAL_ZONE configs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

# ── Part 2: Fault sweep (50 devices, 5 zones — best config) ──────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART 2: Fault sweep (50 devices, 5 zones, $TOTAL_FAULT configs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Sweep complete."
echo "  Results → $RESULTS_ROOT"
echo ""
echo "  Folder structure:"
echo "    $RESULTS_ROOT/"
echo "    ├── zones_10c_2z/run.log"
echo "    ├── zones_10c_5z/run.log"
echo "    ├── zones_50c_5z/run.log"
echo "    ├── zones_50c_10z/run.log"
echo "    ├── zones_50c_25z/run.log"
echo "    ├── fault_fault_free/run.log"
echo "    ├── fault_dev_low/run.log"
echo "    ├── fault_dev_moderate/run.log"
echo "    ├── fault_dev_high/run.log"
echo "    ├── fault_dev_low__zone_low/run.log"
echo "    ├── fault_dev_moderate__zone_moderate/run.log"
echo "    ├── fault_dev_high__zone_high/run.log"
echo "    └── fault_severe/run.log"
echo "════════════════════════════════════════════════════════════════"
