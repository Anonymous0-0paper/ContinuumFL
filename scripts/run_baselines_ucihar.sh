#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Baselines-only on UCI-HAR
#  Adapted for: VSC-5 (vsc5.vsc.ac.at) | Project p73209 | User abolfazlyoun
#
#  ── ONE-TIME SETUP (run once on login node before first sbatch) ─────────────
#    module purge
#    module load cuda/12.3
#    module load python/3.11.3-gcc-12.2.0-hn7p65z
#    python -m venv --upgrade-deps $HOME/cfl_venv
#    source $HOME/cfl_venv/bin/activate
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#    pip install -r requirements.txt
#    deactivate
#
#  Submit:   sbatch scripts/runBS.sh
#  Dry-run:  DRY_RUN=true bash scripts/runBS.sh
#
#  To change dataset: set DATASET below to one of:
#    ucihar | femnist | cifar100 | shakespeare | speechcommands
# ═══════════════════════════════════════════════════════════════════════════════

# ── SLURM directives ───────────────────────────────────────────────────────────
#
#  zen3_0512_a100x2 partition rules (enforced by VSC-5):
#    Half-node (1 GPU): NO --nodes, NO --mem  → 64 cores, 256 GB RAM auto-assigned
#    Full-node (2 GPU): -N <n> + --gres=gpu:2, NO --mem
#
#SBATCH --job-name=CFL_baselines_ucihar
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --mem=128
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.younesi@uibk.ac.at
#SBATCH --output=slurmlogs/logs/slurm.%x.%j.out
#SBATCH --error=slurmlogs/logs/slurm.%x.%j.err
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
BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL IFCA APCfl GeoFL SnapCFL)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ FAULT SCENARIOS  "DEVICE_FAIL:ZONE_FAIL:LABEL"
# └─────────────────────────────────────────────────────────────────────────────
FAULT_CONFIGS=(
    "0.00:0.00:fault_free"
    "0.05:0.00:dev_low"
    "0.10:0.00:dev_moderate"
    "0.20:0.00:dev_high"
    "0.05:0.02:dev_low__zone_low"
    "0.10:0.05:dev_moderate__zone_moderate"
    "0.20:0.10:dev_high__zone_high"
    "0.30:0.15:severe"
)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PATHS
# └─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_NAME="baselines_${DATASET}_$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="$PROJECT_ROOT/results/$RUN_NAME"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_NAME"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ VSC-5 ENVIRONMENT
# │ Load modules then activate the pre-built venv (see ONE-TIME SETUP above).
# │ Do NOT pip install inside this script — that wastes compute time.
# └─────────────────────────────────────────────────────────────────────────────
module purge
module load cuda/12.3
module load python/3.11.3-gcc-12.2.0-hn7p65z

VENV_DIR="$HOME/cfl_venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "   Run the ONE-TIME SETUP steps in this script's header, then resubmit."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ── Info dump for the log ─────────────────────────────────────────────────────
echo "====== Node     : $(hostname) ======"
echo "====== Python   : $(which python) — $(python --version) ======"
nvidia-smi
echo "============================================"

# ── Sanity: confirm PyTorch sees the GPU ─────────────────────────────────────
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
echo "  ContinuumFL — Baselines only (all fault scenarios)"
echo "  Dataset  : $DATASET"
echo "  Methods  : ${BASELINE_METHODS[*]}"
echo "  Run name : $RUN_NAME"
echo "  Results  : $RESULTS_ROOT"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : $(hostname)"
echo "  Scenarios: ${#FAULT_CONFIGS[@]}"
echo "════════════════════════════════════════════════════════"

TOTAL=${#FAULT_CONFIGS[@]}
IDX=0

for cfg in "${FAULT_CONFIGS[@]}"; do
    IFS=':' read -r DEV_F ZONE_F LABEL <<< "$cfg"
    IDX=$(( IDX + 1 ))

    RESULTS_DIR="$RESULTS_ROOT/fault_${LABEL}"
    LOG_DIR="$LOG_ROOT/fault_${LABEL}"
    CHECKPOINT_DIR="$CHECKPOINT_ROOT/fault_${LABEL}"
    mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CHECKPOINT_DIR"

    echo ""
    echo "  [$IDX/$TOTAL] fault_${LABEL}  (dev_fail=${DEV_F}, zone_fail=${ZONE_F})"
    echo "  → $RESULTS_DIR"

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
        --ifca_k           "$NUM_ZONES"
    )

    if [[ "$DEV_F" != "0.00" || "$ZONE_F" != "0.00" ]]; then
        CMD+=(
            --enable_failure
            --device_failure_probability "$DEV_F"
            --zone_failure_probability   "$ZONE_F"
        )
    fi

    echo "    CMD: ${CMD[*]}"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "    [DRY_RUN] skipping"
        continue
    fi

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun "${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"
    else
        "${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/run.log"
    fi
    exit_code=${PIPESTATUS[0]}
    [[ $exit_code -ne 0 ]] && echo "  ⚠️  Exit code $exit_code — check $RESULTS_DIR/run.log"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ Done. Results → $RESULTS_ROOT"
echo ""
echo "  Folder structure:"
echo "    $RESULTS_ROOT/"
echo "    ├── fault_fault_free/baselines/<METHOD>_*/metrics.csv"
echo "    ├── fault_dev_low/..."
echo "    ├── fault_dev_moderate/..."
echo "    ├── fault_dev_high/..."
echo "    ├── fault_dev_low__zone_low/..."
echo "    ├── fault_dev_moderate__zone_moderate/..."
echo "    ├── fault_dev_high__zone_high/..."
echo "    └── fault_severe/..."
echo "════════════════════════════════════════════════════════"