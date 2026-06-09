#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Baselines-only on UCI-HAR
#  Adapted for: VSC-5 (vsc5.vsc.ac.at) | Project p73209 | User abolfazlyoun
#
#  ── ONE-TIME SETUP (run once on login node before first sbatch) ─────────────
#    module purge
#    module load cuda/11.8.0-gcc-12.2.0-bplw5nu
#    module load python/3.11.3-gcc-12.2.0-hn7p65z
#    python -m venv --upgrade-deps $HOME/cfl_venv
#    source $HOME/cfl_venv/bin/activate
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    pip install -r requirements.txt
#    deactivate
#
#  Submit:   sbatch scripts/runBS.sh
#  Dry-run:  DRY_RUN=true bash scripts/runBS.sh
#
#  To change dataset set DATASET below to one of:
#    ucihar | femnist | cifar100 | shakespeare | speechcommands
# ═══════════════════════════════════════════════════════════════════════════════

# ── SLURM directives ──────────────────────────────────────────────────────────
#
#  zen3_0512_a100x2 partition rules (enforced by VSC-5):
#    Half-node (1 GPU): only --gres + --time allowed; NO --nodes, NO --mem,
#                       NO --ntasks, NO --cpus-per-task
#    Full-node (2 GPU): -N <n> + --gres=gpu:2, NO --mem
#
#SBATCH --job-name=CFL_baselines_ucihar
#SBATCH --account=p73209
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.younesi@uibk.ac.at
#SBATCH --output=logs/slurm.%x.%j.out
#SBATCH --error=logs/slurm.%x.%j.err

set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DATASET
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

# ── Baseline methods ──────────────────────────────────────────────────────────
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
# │ VSC-5 ENVIRONMENT
# └─────────────────────────────────────────────────────────────────────────────
module purge
module load cuda/11.8.0-gcc-12.2.0-bplw5nu
module load python/3.11.3-gcc-12.2.0-hn7p65z

VENV_DIR="$HOME/cfl_venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "   Run the ONE-TIME SETUP steps in this script's header first, then resubmit."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ── Info dump ─────────────────────────────────────────────────────────────────
echo "====== Node   : $(hostname) ======"
echo "====== Python : $(which python) — $(python --version) ======"
echo "====== CUDA   : $(nvcc --version | tail -1) ======"
nvidia-smi

python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("❌ CUDA not available to PyTorch — check module + venv setup.")
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"✅ GPU {i}: {p.name} | VRAM: {p.total_memory / 1e9:.1f} GB")
EOF

# ┌─────────────────────────────────────────────────────────────────────────────
# │ RUN
# └─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "  ContinuumFL — Baselines only"
echo "  Dataset  : $DATASET"
echo "  Methods  : ${BASELINE_METHODS[*]}"
echo "  Run name : $RUN_NAME"
echo "  Results  : $RESULTS_DIR"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "════════════════════════════════════════════════════════"

CMD=(
    python "$PROJECT_ROOT/main.py"
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
    --baselines_only
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