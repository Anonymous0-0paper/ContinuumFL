#!/usr/bin/env bash
#SBATCH --job-name=CFL
#SBATCH --partition=IFItitan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=abolfazl.Younesi@uibk.ac.at 
#SBATCH --account=DPS
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/slurm.%x.%j.out
#SBATCH --error=logs/slurm.%x.%j.err

# Slurm launcher for ContinuumFL.
#
# Partition guidance from the request:
# - IFIall: nodes gc1-gc19
# - IFIgpu2070: nodes gc1-gc7, max time 12:00:00, default 00:30:00
# - IFIgpu2070S: nodes gc8-gc16
# - IFItitan: node gc17
# - IFIgpu4090: node gc18, restricted to group TCS
# - IFIgpuL40S: node gc19, 8xL40 GPUs, max time 02:00:00, default 00:30:00
#
# Edit the SBATCH directives above to match the target partition or resource
# request. Request only the GPUs you need, especially on IFIgpuL40S.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/continuumfl_common.sh"

DEFAULT_PRESET="standard"
PRESET="${1:-$DEFAULT_PRESET}"
shift || true

# User-editable defaults for the Slurm job.
DATASET="femnist"
MAX_SAMPLES=-1
NUM_DEVICES=100
NUM_ZONES=20
MIN_ZONE_SIZE=4
MAX_ZONE_SIZE=15
NUM_ROUNDS=200
LOCAL_EPOCHS=5
LEARNING_RATE=0.01
BATCH_SIZE=32
SPATIAL_WEIGHT=0.4
DATA_WEIGHT=0.4
NETWORK_WEIGHT=0.2
SPATIAL_REGULARIZATION=0.1
CORRELATION_THRESHOLD=0.05
COMPRESSION_RATE=0.1
ENABLE_COMPRESSION=false
INTRA_ZONE_ALPHA=100
INTER_ZONE_ALPHA=10
ASYNC_AGGREGATION=false
ENABLE_FAILURE=false
DEVICE_FAILURE_PROBABILITY=0.05
ZONE_FAILURE_PROBABILITY=0.02
SHAKESPEARE_NUM_SPEAKERS=35
RUN_BASELINES=true
BASELINES_ONLY=false
BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
SAVE_RESULTS=true
CREATE_VISUALIZATIONS=true
DEVICE="cuda"
RANDOM_SEED=42
LOG_DIR="$PROJECT_ROOT/logs"
RESULTS_ROOT="$PROJECT_ROOT/results"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"

apply_preset_defaults "$PRESET"

RUN_NAME="${RUN_NAME:-${PRESET}_slurm_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_DIR/$RUN_NAME"
RESULTS_DIR="$RESULTS_ROOT/$RUN_NAME"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHECKPOINT_DIR"

build_main_args

CMD=("$PYTHON_BIN" "$PROJECT_ROOT/main.py" "${MAIN_ARGS[@]}" "$@")

echo "Slurm job: ${SLURM_JOB_ID:-local}"
echo "Preset: $PRESET"
echo "Run name: $RUN_NAME"
echo "Results: $RESULTS_DIR"
echo "Log: $LOG_DIR/run.log"
echo "Command: ${CMD[*]}"

if bool_true "${DRY_RUN:-false}"; then
    exit 0
fi

srun "${CMD[@]}" 2>&1 | tee "$LOG_DIR/run.log"
