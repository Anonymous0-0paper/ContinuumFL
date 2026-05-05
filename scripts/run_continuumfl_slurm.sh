#!/usr/bin/env bash
#SBATCH --job-name=CFL-femnist
#SBATCH --partition=IFItitan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.Younesi@uibk.ac.at
#SBATCH --account=DPS
#SBATCH --mem=48G
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
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$PROJECT_ROOT/scripts/continuumfl_common.sh"

DEFAULT_PRESET="femnist"
PRESET="${1:-$DEFAULT_PRESET}"
shift || true

# User-editable defaults for the Slurm job.
DATASET="femnist"
MAX_SAMPLES=-1
NUM_DEVICES=100
NUM_ZONES=20
MIN_ZONE_SIZE=4
MAX_ZONE_SIZE=15
NUM_ROUNDS=100
LOCAL_EPOCHS=5
LEARNING_RATE=0.01
BATCH_SIZE=64
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

# Load Python module on HPC cluster
# Uncomment and adjust the module name as needed for your cluster
# module load python/3.11  # or whatever version is available
# module load pytorch      # optional, if pytorch module exists

# Find python if not in PATH
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    # Try python3
    if command -v python3 &> /dev/null; then
        PYTHON_BIN="python3"
        echo "🔧 python not found, using python3 instead"
    else
        echo "❌ Error: Neither 'python' nor 'python3' found in PATH"
        echo "💡 Please load the Python module on your HPC cluster, e.g.:"
        echo "   module load python/3.11"
        exit 1
    fi
fi

# Install requirements if not already installed
echo "📦 Checking and installing requirements..."

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🔨 Creating virtual environment..."
    echo "   Location: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" --upgrade-deps
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

# Upgrade pip
echo "📦 Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | tail -5

# Detect GPU and install appropriate PyTorch
echo "🔍 Detecting GPU type for PyTorch installation..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ NVIDIA GPU detected"
    echo "📦 Installing CUDA-enabled PyTorch..."
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -10
    GPU_TYPE="NVIDIA"
elif command -v rocm-smi &> /dev/null; then
    echo "   ✅ AMD GPU detected"
    echo "📦 Installing ROCm-enabled PyTorch..."
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 2>&1 | tail -10
    GPU_TYPE="AMD"
else
    echo "   ⚠️  No GPU detected, installing CPU PyTorch"
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio 2>&1 | tail -10
    GPU_TYPE="CPU"
fi

# Install other requirements (excluding torch since we already installed it)
echo "📦 Installing other requirements from $PROJECT_ROOT/requirements.txt..."
cat "$PROJECT_ROOT/requirements.txt" | grep -v "^torch\|^#.*torch\|cuda\|rocm" | "$PYTHON_BIN" -m pip install --quiet -r /dev/stdin 2>&1 || true

# Verify torch is installed
echo "✓ Verifying PyTorch installation..."
if ! "$PYTHON_BIN" -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "❌ Error: PyTorch installation failed"
    exit 1
fi

echo "✅ All requirements verified (GPU Type: $GPU_TYPE)"

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
