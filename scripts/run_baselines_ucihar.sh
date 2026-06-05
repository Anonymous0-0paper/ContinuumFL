#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Baselines-only on UCI-HAR
#  Adapted for: VSC-5 (vsc5.vsc.ac.at) | Project p73209 | User abolfazlyoun
#
#  Submit:   sbatch scripts/run_baselines_ucihar_vsc5.sh
#  Dry-run:  DRY_RUN=true bash scripts/run_baselines_ucihar_vsc5.sh
#
#  To change dataset: set DATASET below to one of:
#    ucihar | femnist | cifar100 | shakespeare | speechcommands
# ═══════════════════════════════════════════════════════════════════════════════

# ── SLURM directives ───────────────────────────────────────────────────────────
#SBATCH --job-name=CFL_baselines_ucihar
#SBATCH --account=p73209                      # VSC-5 project account
#SBATCH --partition=zen3_0512_a100x2          # NVIDIA A100 partition on VSC-5
#SBATCH --qos=zen3_0512_a100x2               # Must match partition on VSC-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                          # Single A100 (40 GB VRAM)
#SBATCH --time=23:00:00                       # Max 3 days on VSC-5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Abolfazl.Younesi@uibk.ac.at
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
BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL IFCA APCfl GeoFL SnapCFL)

# ┌─────────────────────────────────────────────────────────────────────────────
# │ FAULT SCENARIOS
# │   Format: "DEVICE_FAIL:ZONE_FAIL:LABEL"
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
# │ PATHS  (VSC-5: run from $DATA, not $HOME)
# └─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_NAME="baselines_${DATASET}_$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="$PROJECT_ROOT/results/$RUN_NAME"
LOG_ROOT="$PROJECT_ROOT/logs/$RUN_NAME"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/$RUN_NAME"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ VSC-5 ENVIRONMENT SETUP
# │ VSC-5 uses AlmaLinux + module system — load CUDA before anything else
# └─────────────────────────────────────────────────────────────────────────────

# Clean the module environment
module purge

# Load CUDA 12.3 (available on VSC-5 A100 nodes)
module load cuda/12.3

# Print GPU info for the log
echo "====== Node: $(hostname) ======"
echo "====== CUDA module loaded ======"
nvidia-smi
echo "==============================="

# ── Python / venv setup ───────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    echo "❌ python3 not found — load a python module or set PYTHON_BIN."
    exit 1
fi

VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" --upgrade-deps
fi
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | tail -3

# ── Install PyTorch for CUDA 12.3 (A100 on VSC-5) ────────────────────────────
# VSC-5 A100 nodes have CUDA 12.3 — use cu121 wheel (closest stable match)
"$PYTHON_BIN" -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

# ── Install remaining requirements (skip torch lines to avoid conflicts) ───────
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    grep -v "^torch\|^#.*torch\|cuda\|rocm" "$PROJECT_ROOT/requirements.txt" \
        | "$PYTHON_BIN" -m pip install -r /dev/stdin 2>&1 || true
fi

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SANITY CHECK — confirm GPU is visible to PyTorch
# └─────────────────────────────────────────────────────────────────────────────
"$PYTHON_BIN" - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("❌ CUDA not available to PyTorch — check module and driver.")
    sys.exit(1)
n = torch.cuda.device_count()
for i in range(n):
    print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)} | "
          f"VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
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

    # Use srun inside SLURM, direct call otherwise
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