#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/continuumfl_common.sh"

DEFAULT_PRESET="standard"
PRESET="${1:-$DEFAULT_PRESET}"
shift || true

# User-editable defaults. The preset fills these in, and extra CLI args
# passed after the preset still override them when main.py parses argv.
DATASET="cifar100"
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
INTRA_ZONE_ALPHA=10
INTER_ZONE_ALPHA=0.3
ASYNC_AGGREGATION=false
ENABLE_FAILURE=false
DEVICE_FAILURE_PROBABILITY=0.05
ZONE_FAILURE_PROBABILITY=0.02
SHAKESPEARE_NUM_SPEAKERS=35
RUN_BASELINES=false
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

RUN_NAME="${RUN_NAME:-${PRESET}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$LOG_DIR/$RUN_NAME"
RESULTS_DIR="$RESULTS_ROOT/$RUN_NAME"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHECKPOINT_DIR"

build_main_args

CMD=("$PYTHON_BIN" "$PROJECT_ROOT/main.py" "${MAIN_ARGS[@]}" "$@")

echo "Preset: $PRESET"
echo "Run name: $RUN_NAME"
echo "Results: $RESULTS_DIR"
echo "Log: $LOG_DIR/run.log"
echo "Command: ${CMD[*]}"

if bool_true "${DRY_RUN:-false}"; then
    exit 0
fi

"${CMD[@]}" 2>&1 | tee "$LOG_DIR/run.log"
