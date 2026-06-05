#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Baselines-only on UCI-HAR
#
#  Submit:   sbatch scripts/run_baselines_ucihar.sh
#  Dry-run:  DRY_RUN=true bash scripts/run_baselines_ucihar.sh
#
#  To change dataset: set DATASET below to one of:
#    ucihar | femnist | cifar100 | shakespeare | speechcommands
# ═══════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=CFL_baselines_ucihar
#SBATCH --partition=IFItitan          # IFIall | IFIgpu2070 | IFIgpu2070S | IFItitan | IFIgpuL40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.Younesi@uibk.ac.at
#SBATCH --account=DPS
#SBATCH --output=logs/slurm.%x.%j.out
#SBATCH --error=logs/slurm.%x.%j.err

set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DATASET — change this one line to switch datasets:
# │   ucihar | femnist | cifar100 | shakespeare | speechcommands
# └─────────────────────────────────────────────────────────────────────────────
DATASET="ucihar"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ EXPERIMENT PARAMETERS
# └─────────────────────────────────────────────────────────────────────────────
MAX_SAMPLES=-1
NUM_DEVICES=50
NUM_ZONES=5
MIN_ZONE_SIZE=4
MAX_ZONE_SIZE=15
NUM_ROUNDS=200
LOCAL_EPOCHS=5
LEARNING_RATE=0.001
BATCH_SIZE=64
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

# ── Baseline methods to run ────────────────────────────────────────────────────
# Remove or add methods as needed:
#   FedAvg | FedProx | HierFL | ClusterFL | IFCA | APCfl | GeoFL | SnapCFL
BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PATHS
# └─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_NAME="baselines_${DATASET}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
RESULTS_DIR="$PROJECT_ROOT/results/$RUN_NAME"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHECKPOINT_DIR"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ ENVIRONMENT SETUP
# └─────────────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    command -v python3 &>/dev/null && PYTHON_BIN="python3" || {
        echo "❌ Neither 'python' nor 'python3' found."
        exit 1
    }
fi

VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" --upgrade-deps
fi
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | tail -3

if command -v nvidia-smi &>/dev/null; then
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -3
elif command -v rocm-smi &>/dev/null; then
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm5.7 2>&1 | tail -3
else
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio 2>&1 | tail -3
fi

grep -v "^torch\|^#.*torch\|cuda\|rocm" "$PROJECT_ROOT/requirements.txt" \
    | "$PYTHON_BIN" -m pip install --quiet -r /dev/stdin 2>&1 || true

# ┌─────────────────────────────────────────────────────────────────────────────
# │ RUN
# └─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  ContinuumFL — Baselines only"
echo "  Dataset  : $DATASET"
echo "  Methods  : ${BASELINE_METHODS[*]}"
echo "  Run name : $RUN_NAME"
echo "  Results  : $RESULTS_DIR"
echo "════════════════════════════════════════════════════════"

CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/main.py"
    --dataset          "$DATASET"
    --max_samples      "$MAX_SAMPLES"
    --num_devices      "$NUM_DEVICES"
    --num_zones        "$NUM_ZONES"
    --min_zone_size    "$MIN_ZONE_SIZE"
    --max_zone_size    "$MAX_ZONE_SIZE"
    --num_rounds       "$NUM_ROUNDS"
    --local_epochs     "$LOCAL_EPOCHS"
    --learning_rate    "$LEARNING_RATE"
    --batch_size       "$BATCH_SIZE"
    --intra_zone_alpha "$INTRA_ZONE_ALPHA"
    --inter_zone_alpha "$INTER_ZONE_ALPHA"
    --compression_rate "$COMPRESSION_RATE"
    --spatial_weight   "$SPATIAL_WEIGHT"
    --data_weight      "$DATA_WEIGHT"
    --network_weight   "$NETWORK_WEIGHT"
    --spatial_regularization "$SPATIAL_REGULARIZATION"
    --correlation_threshold  "$CORRELATION_THRESHOLD"
    --device           "$DEVICE"
    --random_seed      "$RANDOM_SEED"
    --log_dir          "$LOG_DIR"
    --results_dir      "$RESULTS_DIR"
    --checkpoint_dir   "$CHECKPOINT_DIR"
    --baselines_only                          # skip ContinuumFL, run baselines only
    --run_baselines
    --baseline_methods "${BASELINE_METHODS[@]}"
    --save_results
)

echo "CMD: ${CMD[*]}"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "[DRY_RUN] skipping execution"
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun "${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"
else
    "${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"
fi

echo "✅ Done. Results → $RESULTS_DIR"
