#!/bin/bash
# =============================================================================
# Full UCI HAR benchmark: ContinuumFL + all baselines
# Algorithms: ContinuumFL, FedAvg, FedProx, HierFL, ClusterFL, AP-CFL, GeoFL
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET="ucihar"
NUM_ROUNDS=200
NUM_DEVICES=50
NUM_ZONES=10
LOCAL_EPOCHS=5
LR=0.001
BATCH_SIZE=32
INTRA_ALPHA=100
INTER_ALPHA=5
COMPRESSION_RATE=0.6
SEED=42
DEVICE="cuda"        # change to "cpu" if no GPU available

# AP-CFL paper hyperparameters for UCI HAR
APCFL_LR=0.0005
APCFL_LAMBDA=0.01
APCFL_LOCAL_EPOCHS=1
APCFL_SAMPLING_RATE=0.8

# GeoFL paper hyperparameters
GEOFL_LR=0.001
GEOFL_LOCAL_STEPS=5
GEOFL_R0=5
GEOFL_S0=0.01
GEOFL_S_MIN=0.001
GEOFL_ALPHA=0.95
GEOFL_BETA=0.2
GEOFL_SAMPLING=0.7

RESULTS_DIR="./results/ucihar_full_benchmark"
LOG_DIR="./logs/ucihar_full_benchmark"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$RESULTS_DIR/summary_${TIMESTAMP}.txt"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY_FILE"; }

run_experiment() {
    local name="$1"
    local extra_args="${2:-}"
    local logfile="$LOG_DIR/${name}_${TIMESTAMP}.log"

    log "▶  Starting: $name"
    local t_start=$SECONDS

    # shellcheck disable=SC2086
    python main.py \
        --dataset "$DATASET" \
        --num_rounds "$NUM_ROUNDS" \
        --num_devices "$NUM_DEVICES" \
        --num_zones "$NUM_ZONES" \
        --local_epochs "$LOCAL_EPOCHS" \
        --learning_rate "$LR" \
        --batch_size "$BATCH_SIZE" \
        --intra_zone_alpha "$INTRA_ALPHA" \
        --inter_zone_alpha "$INTER_ALPHA" \
        --compression_rate "$COMPRESSION_RATE" \
        --random_seed "$SEED" \
        --device "$DEVICE" \
        --results_dir "$RESULTS_DIR" \
        --log_dir "$LOG_DIR" \
        --save_results \
        $extra_args \
        2>&1 | tee "$logfile"

    local elapsed=$(( SECONDS - t_start ))
    log "✔  Done: $name  (${elapsed}s)  →  log: $logfile"
    echo ""
}

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------
log "=================================================================="
log "UCI HAR Full Benchmark  —  $(date)"
log "Rounds=$NUM_ROUNDS  Devices=$NUM_DEVICES  Zones=$NUM_ZONES"
log "LR=$LR  Batch=$BATCH_SIZE  Seed=$SEED  Device=$DEVICE"
log "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# 1. ContinuumFL (our method)
# ---------------------------------------------------------------------------
run_experiment "ContinuumFL" \
    "--run_baselines --baseline_methods FedAvg FedProx HierFL ClusterFL APCFL GeoFL \
     --apcfl_num_rounds $NUM_ROUNDS \
     --apcfl_local_epochs $APCFL_LOCAL_EPOCHS \
     --apcfl_lr $APCFL_LR \
     --apcfl_lambda $APCFL_LAMBDA \
     --apcfl_sampling_rate $APCFL_SAMPLING_RATE \
     --geofl_num_rounds $NUM_ROUNDS \
     --geofl_local_steps $GEOFL_LOCAL_STEPS \
     --geofl_lr $GEOFL_LR \
     --geofl_S0 $GEOFL_S0 \
     --geofl_S_min $GEOFL_S_MIN \
     --geofl_alpha $GEOFL_ALPHA \
     --geofl_R0 $GEOFL_R0 \
     --geofl_beta $GEOFL_BETA \
     --geofl_client_sampling_rate $GEOFL_SAMPLING"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "=================================================================="
log "All experiments finished."
log "Results directory: $RESULTS_DIR"
log "Logs directory:    $LOG_DIR"
log "Summary file:      $SUMMARY_FILE"
log "=================================================================="
