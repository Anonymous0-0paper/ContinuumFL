#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Single Slurm launcher
#  Combines: run_continuumfl_slurm.sh, run_continuumfl_slurm_quick.sh,
#            run_continuumfl.sh, continuumfl_common.sh
#
#  Submit:    sbatch scripts/run_continuumfl_slurm.sh [PRESET]
#  Dry-run:   DRY_RUN=true bash scripts/run_continuumfl_slurm.sh [PRESET]
#  Local run: bash scripts/run_continuumfl_slurm.sh [PRESET]
#
#  Available presets (see Section 2 below):
#    quick | femnist | cifar100 | shakespeare | zone_sweep | fault_sweep |
#    noniid_sweep | large | comm | baseline
# ═══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 1 — SLURM DIRECTIVES
# │ Change these to match your cluster partition and resource needs.
# └─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=CFL
#SBATCH --partition=IFItitan          # IFIall | IFIgpu2070 | IFIgpu2070S | IFItitan | IFIgpuL40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             # reduce to 4 for quick/small runs
#SBATCH --mem=24G                     # 16G for quick, 48G for standard, 96G for large
#SBATCH --gres=gpu:1                  # increase to gpu:2 or gpu:8 on IFIgpuL40S
#SBATCH --time=02-00:00:00            # quick=00:30:00 | standard=02:00:00 | large=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abolfazl.Younesi@uibk.ac.at
#SBATCH --account=DPS
#SBATCH --output=logs/slurm.%x.%j.out
#SBATCH --error=logs/slurm.%x.%j.err

set -euo pipefail

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 2 — PRESET SELECTION
# │ Pass the preset as the first argument, or change DEFAULT_PRESET here.
# │
# │   quick        → fast smoke-test (10 dev, 2 zones, 10 rounds, femnist)
# │   femnist       → best FEMNIST settings (50 dev, 5 zones, 200 rounds)
# │   cifar100      → best CIFAR-100 settings (50 dev, 5 zones, 200 rounds)
# │   shakespeare   → best Shakespeare settings (50 dev, 5 zones, 200 rounds)
# │   zone_sweep    → runs all (devices × zones) configs sequentially
# │   noniid_sweep  → sweeps inter_zone_alpha across 7 levels
# │   fault_sweep   → sweeps device & zone failure rates across 8 combos
# │   large         → scaled-up FEMNIST (500 dev, 50 zones, 150 rounds)
# │   comm          → communication-efficiency focus (compression ON)
# │   baseline      → standard + baselines comparison enabled
# └─────────────────────────────────────────────────────────────────────────────
DEFAULT_PRESET="femnist"
PRESET="${1:-$DEFAULT_PRESET}"
shift || true   # remaining args are forwarded to main.py

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 3 — PARAMETERS YOU CAN OVERRIDE
# │ These are the defaults BEFORE a preset is applied.
# │ After the preset block below, any variable can be overridden again.
# └─────────────────────────────────────────────────────────────────────────────
# ── Core ──────────────────────────────────────────────────────────────────────
DATASET="femnist"          # femnist | cifar100 | shakespeare
MAX_SAMPLES=-1             # -1 = use all; set e.g. 5000 for faster dev runs
NUM_DEVICES=50
NUM_ZONES=5
MIN_ZONE_SIZE=4
MAX_ZONE_SIZE=15
NUM_ROUNDS=200
LOCAL_EPOCHS=5
LEARNING_RATE=0.001        # femnist:0.001 | cifar100:0.001 | shakespeare:0.001
BATCH_SIZE=64              # shakespeare: use 16

# ── Non-IID heterogeneity ──────────────────────────────────────────────────────
INTRA_ZONE_ALPHA=100       # within-zone IID-ness: higher = more IID
INTER_ZONE_ALPHA=5.0       # across-zone heterogeneity: best from sweep = 5.0

# ── Compression ────────────────────────────────────────────────────────────────
COMPRESSION_RATE=0.10      # fraction of top-k gradients kept (0.10 = 10%)
ENABLE_COMPRESSION=false   # set true to activate top-k compression

# ── Spatial clustering weights ─────────────────────────────────────────────────
SPATIAL_WEIGHT=0.4
DATA_WEIGHT=0.4
NETWORK_WEIGHT=0.2
SPATIAL_REGULARIZATION=0.05
CORRELATION_THRESHOLD=0.05

# ── Fault tolerance ────────────────────────────────────────────────────────────
ENABLE_FAILURE=false
DEVICE_FAILURE_PROBABILITY=0.05   # per-round probability a device drops
ZONE_FAILURE_PROBABILITY=0.02     # per-round probability a zone fails

# ── Early stopping ─────────────────────────────────────────────────────────────
ENABLE_EARLY_STOPPING=true
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_MIN_DELTA=0.0001

# ── Aggregation ────────────────────────────────────────────────────────────────
ASYNC_AGGREGATION=false

# ── Baselines ──────────────────────────────────────────────────────────────────
RUN_BASELINES=false
BASELINES_ONLY=false
BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)

# ── Shakespeare-specific ───────────────────────────────────────────────────────
SHAKESPEARE_NUM_SPEAKERS=35

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_RESULTS=true
CREATE_VISUALIZATIONS=false    # set true to auto-generate plots (requires matplotlib)
DEVICE="cuda"
RANDOM_SEED=42

# Paths (resolved relative to the project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_DIR="$PROJECT_ROOT/logs"
RESULTS_ROOT="$PROJECT_ROOT/results"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 4 — PRESET DEFINITIONS
# │ Each preset overrides only the values relevant to that experiment.
# │ Values NOT listed in a preset keep whatever you set in Section 3.
# └─────────────────────────────────────────────────────────────────────────────
apply_preset() {
    case "$PRESET" in

        # ── Smoke-test: runs in ~15 min ──────────────────────────────────────
        quick)
            DATASET="femnist"; MAX_SAMPLES=5000
            NUM_DEVICES=10;    NUM_ZONES=2; MIN_ZONE_SIZE=2; MAX_ZONE_SIZE=8
            NUM_ROUNDS=10;     LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=5
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            ;;

        # ── Best FEMNIST settings (from noniid + zone + compression sweeps) ──
        femnist)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            ;;

        # ── Best CIFAR-100 settings (ResNet18, 100 classes) ──────────────────
        cifar100)
            DATASET="cifar100"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            ;;

        # ── Best Shakespeare settings (LSTM, 35 speakers) ────────────────────
        shakespeare)
            DATASET="shakespeare"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=2; MAX_ZONE_SIZE=8
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=16
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false; SHAKESPEARE_NUM_SPEAKERS=35
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            ;;

        # ── Zone-granularity sweep (7 configs × 1 dataset) ───────────────────
        zone_sweep)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            # NUM_DEVICES / NUM_ZONES handled inside the sweep loop below
            ;;

        # ── Non-IID heterogeneity sweep (7 alpha levels) ─────────────────────
        noniid_sweep)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            # INTER_ZONE_ALPHA swept inside the loop below
            ;;

        # ── Fault-rate sweep (8 dev×zone failure combos) ─────────────────────
        fault_sweep)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            RUN_BASELINES=false; CREATE_VISUALIZATIONS=false
            # ENABLE_FAILURE / DEV/ZONE prob swept inside the loop below
            ;;

        # ── Large-scale run ───────────────────────────────────────────────────
        large)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_DEVICES=500;   NUM_ZONES=50; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=150;    LOCAL_EPOCHS=3; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=true
            ;;

        # ── Communication-efficiency focus ────────────────────────────────────
        comm)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=true                     # ← compression ON
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false; RUN_BASELINES=false; CREATE_VISUALIZATIONS=true
            ;;

        # ── Baselines comparison ──────────────────────────────────────────────
        baseline)
            DATASET="femnist"; MAX_SAMPLES=-1
            NUM_DEVICES=50;    NUM_ZONES=5; MIN_ZONE_SIZE=4; MAX_ZONE_SIZE=15
            NUM_ROUNDS=200;    LOCAL_EPOCHS=5; LEARNING_RATE=0.001; BATCH_SIZE=64
            INTRA_ZONE_ALPHA=100; INTER_ZONE_ALPHA=5.0; COMPRESSION_RATE=0.10
            ENABLE_COMPRESSION=false
            ENABLE_EARLY_STOPPING=true; EARLY_STOPPING_PATIENCE=10
            ENABLE_FAILURE=false
            RUN_BASELINES=true                          # ← baselines ON
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=true
            ;;

        *)
            echo "❌ Unknown preset: '$PRESET'" >&2
            echo "   Available: quick | femnist | cifar100 | shakespeare | zone_sweep | noniid_sweep | fault_sweep | large | comm | baseline" >&2
            exit 1
            ;;
    esac
}

apply_preset

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 5 — ENVIRONMENT SETUP (Python + PyTorch)
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
# │ SECTION 6 — HELPERS
# └─────────────────────────────────────────────────────────────────────────────
bool_flag() {
    # Emit the flag string if value is true, nothing otherwise
    [[ "${1:-false}" == "true" ]] && echo "$2" || true
}

run_single() {
    # run_single <results_dir> <extra main.py args...>
    local run_results_dir="$1"; shift
    local run_log="$run_results_dir/run.log"
    mkdir -p "$run_results_dir" "$LOG_DIR" "$CHECKPOINT_DIR"

    local cmd=("$PYTHON_BIN" "$PROJECT_ROOT/main.py"
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
        --results_dir      "$run_results_dir"
        --checkpoint_dir   "$CHECKPOINT_DIR"
    )

    # Boolean flags
    [[ "$ENABLE_COMPRESSION"      == "true" ]] && cmd+=(--enable_compression)
    [[ "$ASYNC_AGGREGATION"       == "true" ]] && cmd+=(--async_aggregation)
    [[ "$ENABLE_FAILURE"          == "true" ]] && cmd+=(--enable_failure
        --device_failure_probability "$DEVICE_FAILURE_PROBABILITY"
        --zone_failure_probability   "$ZONE_FAILURE_PROBABILITY")
    [[ "$ENABLE_EARLY_STOPPING"   == "true" ]] && cmd+=(--enable_early_stopping
        --early_stopping_patience  "$EARLY_STOPPING_PATIENCE"
        --early_stopping_min_delta "$EARLY_STOPPING_MIN_DELTA")
    [[ "$RUN_BASELINES"           == "true" ]] && cmd+=(--run_baselines)
    [[ "$BASELINES_ONLY"          == "true" ]] && cmd+=(--baselines_only)
    [[ "$SAVE_RESULTS"            == "true" ]] && cmd+=(--save_results)
    [[ "$CREATE_VISUALIZATIONS"   == "true" ]] && cmd+=(--create_visualizations)
    [[ "$DATASET"                 == "shakespeare" ]] && \
        cmd+=(--shakespeare_num_speakers "$SHAKESPEARE_NUM_SPEAKERS")
    [[ ${#BASELINE_METHODS[@]}    -gt 0 ]] && \
        cmd+=(--baseline_methods "${BASELINE_METHODS[@]}")

    # Forward any extra args passed to this function
    cmd+=("$@")

    echo "   CMD: ${cmd[*]}"
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "   [DRY_RUN] skipping execution"
        return 0
    fi

    # On Slurm use srun; otherwise run directly
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun "${cmd[@]}" 2>&1 | tee "$run_log"
    else
        "${cmd[@]}" 2>&1 | tee "$run_log"
    fi

    local exit_code=${PIPESTATUS[0]}
    [[ $exit_code -ne 0 ]] && echo "⚠️  Exit code $exit_code — check $run_log" || true
    return $exit_code
}

# ┌─────────────────────────────────────────────────────────────────────────────
# │ SECTION 7 — RUN
# └─────────────────────────────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-${PRESET}_slurm_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_DIR/$RUN_NAME"
RESULTS_DIR="$RESULTS_ROOT/$RUN_NAME"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  ContinuumFL Slurm Launcher"
echo "  Preset   : $PRESET"
echo "  Run name : $RUN_NAME"
echo "  Slurm ID : ${SLURM_JOB_ID:-local}"
echo "  Results  : $RESULTS_DIR"
echo "  Log      : $LOG_DIR"
echo "════════════════════════════════════════════════════════════════"

# ── Single-run presets ───────────────────────────────────────────────────────
if [[ "$PRESET" != "zone_sweep" && "$PRESET" != "noniid_sweep" && "$PRESET" != "fault_sweep" ]]; then
    run_single "$RESULTS_DIR" "$@"
    echo "✅ Done. Results → $RESULTS_DIR"
    exit 0
fi

# ── zone_sweep ───────────────────────────────────────────────────────────────
if [[ "$PRESET" == "zone_sweep" ]]; then
    # Format: "NUM_DEVICES:NUM_ZONES:MIN_ZONE_SIZE:MAX_ZONE_SIZE"
    ZONE_CONFIGS=(
        "10:2:3:6"      #  10 clients, 2 zones  — coarse
        "10:5:1:4"      #  10 clients, 5 zones  — fine
        "20:4:3:8"      #  20 clients, 4 zones  — coarse
        "20:10:1:4"     #  20 clients, 10 zones — fine
        "50:5:4:15"     #  50 clients, 5 zones  — BEST BASELINE
        "50:10:3:8"     #  50 clients, 10 zones — finer
        "50:25:1:4"     #  50 clients, 25 zones — finest
    )
    TOTAL=${#ZONE_CONFIGS[@]}; IDX=0
    echo "  Zone-sweep: $TOTAL configs"; echo ""
    for CFG in "${ZONE_CONFIGS[@]}"; do
        IFS=':' read -r NUM_DEVICES NUM_ZONES MIN_ZONE_SIZE MAX_ZONE_SIZE <<< "$CFG"
        IDX=$((IDX+1))
        EXP="${DATASET}__intra${INTRA_ZONE_ALPHA}__inter${INTER_ZONE_ALPHA}__comp$(python3 -c "print(int($COMPRESSION_RATE*100))")pct__dev${NUM_DEVICES}__zones${NUM_ZONES}"
        RUN_DIR="$RESULTS_DIR/$EXP"
        echo "  [$IDX/$TOTAL] dev=$NUM_DEVICES zones=$NUM_ZONES → $RUN_DIR"
        run_single "$RUN_DIR" || true
    done
    echo "✅ Zone-sweep complete. Results → $RESULTS_DIR"
    exit 0
fi

# ── noniid_sweep ─────────────────────────────────────────────────────────────
if [[ "$PRESET" == "noniid_sweep" ]]; then
    INTER_ALPHAS=(0.1 0.5 1.0 5.0 10.0 50.0 100.0)
    TOTAL=${#INTER_ALPHAS[@]}; IDX=0
    echo "  Non-IID sweep: $TOTAL alpha levels"; echo ""
    for ALPHA in "${INTER_ALPHAS[@]}"; do
        IDX=$((IDX+1))
        INTER_ZONE_ALPHA="$ALPHA"
        EXP="${DATASET}__intra${INTRA_ZONE_ALPHA}__inter${ALPHA}__comp$(python3 -c "print(int($COMPRESSION_RATE*100))")pct"
        RUN_DIR="$RESULTS_DIR/$EXP"
        echo "  [$IDX/$TOTAL] inter_alpha=$ALPHA → $RUN_DIR"
        run_single "$RUN_DIR" || true
    done
    echo "✅ Non-IID sweep complete. Results → $RESULTS_DIR"
    exit 0
fi

# ── fault_sweep ───────────────────────────────────────────────────────────────
if [[ "$PRESET" == "fault_sweep" ]]; then
    # Format: "DEVICE_FAIL_PROB:ZONE_FAIL_PROB"
    FAULT_CONFIGS=(
        "0.00:0.00"   # fault-free baseline
        "0.05:0.00"   # low device fault only
        "0.10:0.00"   # moderate device fault only
        "0.20:0.00"   # high device fault only
        "0.05:0.02"   # low device + low zone  (defaults)
        "0.10:0.05"   # moderate device + zone
        "0.20:0.10"   # high device + zone
        "0.30:0.15"   # severe scenario
    )
    TOTAL=${#FAULT_CONFIGS[@]}; IDX=0
    echo "  Fault-sweep: $TOTAL configs"; echo ""
    for CFG in "${FAULT_CONFIGS[@]}"; do
        IFS=':' read -r DEV_F ZONE_F <<< "$CFG"
        IDX=$((IDX+1))
        EXP="${DATASET}_dev${DEV_F}_zone${ZONE_F}"
        RUN_DIR="$RESULTS_DIR/$EXP"
        echo "  [$IDX/$TOTAL] dev_fail=$DEV_F zone_fail=$ZONE_F → $RUN_DIR"

        if [[ "$DEV_F" == "0.00" && "$ZONE_F" == "0.00" ]]; then
            ENABLE_FAILURE=false
        else
            ENABLE_FAILURE=true
            DEVICE_FAILURE_PROBABILITY="$DEV_F"
            ZONE_FAILURE_PROBABILITY="$ZONE_F"
        fi
        run_single "$RUN_DIR" || true
    done
    echo "✅ Fault-sweep complete. Results → $RESULTS_DIR"
    exit 0
fi
