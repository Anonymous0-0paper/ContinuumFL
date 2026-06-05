#!/bin/bash
# Run ContinuumFL across different compression rates with compression ENABLED.
# Hyperparameters are fixed to the BEST values identified by compare_noniid_results.py:
#   inter_zone_alpha = 5.0  → composite-score winner (accuracy + F1 + conv. speed + comm. cost)
#   intra_zone_alpha = 100  → near-IID within zones (best generalisation)
#   early_stopping   = ON   → patience 20 rounds, min_delta 0.0001
# Only --compression_rate is swept across runs.

DATASET=${1:-femnist}
NUM_ROUNDS=${2:-200}
NUM_DEVICES=50
NUM_ZONES=5
INTRA_ALPHA=100   # best value: intra-zone near-IID
INTER_ALPHA=5.0   # best value: moderate non-IID (composite-score #1 from compare_noniid_results.py)
RESULTS_DIR="./results/compression_sweep"

mkdir -p "$RESULTS_DIR"

# Early stopping — always enabled with best-found values (patience=20, min_delta=0.0001).
# Override via env vars:  ES_PATIENCE=30 ES_DELTA=0.001 ./run_compression_experiments.sh
ES_PATIENCE=${ES_PATIENCE:-10}
ES_DELTA=${ES_DELTA:-0.0001}
ES_FLAGS="--enable_early_stopping --early_stopping_patience $ES_PATIENCE --early_stopping_min_delta $ES_DELTA"
echo "⏹  Early stopping: ON  (patience=$ES_PATIENCE, min_delta=$ES_DELTA)"

# Compression rates to sweep:
#   0.01 → 1 % of gradients transmitted  (extreme compression)
#   0.05 → 5 %
#   0.10 → 10 %  (framework default)
#   0.20 → 20 %
#   0.30 → 30 %
#   0.50 → 50 %
#   1.00 → no compression (full gradient, enable_compression has no effect)
COMP_RATES=(0.01 0.05 0.10 0.20 0.30 0.50 0.60 0.70 0.80 0.90 1.00)

echo "========================================================"
echo "ContinuumFL Compression Rate Sweep (compression=ENABLED)"
echo "[Hyperparams fixed to best non-IID setup from compare_noniid_results.py]"
echo "Dataset        : $DATASET"
echo "Rounds         : $NUM_ROUNDS"
echo "Devices        : $NUM_DEVICES  |  Zones: $NUM_ZONES"
echo "intra_alpha    : $INTRA_ALPHA  (best: near-IID within zones)"
echo "inter_alpha    : $INTER_ALPHA  (best: moderate non-IID, composite-score #1)"
echo "Comp rates     : ${COMP_RATES[*]}"
echo "Early stop     : patience=$ES_PATIENCE  min_delta=$ES_DELTA"
echo "========================================================"

for RATE in "${COMP_RATES[@]}"; do
    COMP_PCT=$(python3 -c "print(int(round(${RATE}*100)))")
    EXP_NAME="compression_rate${RATE}"
    LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- Running: compression_rate=$RATE  (${COMP_PCT}% of gradients kept) ---"
    echo "Log: $LOG_FILE"

    python main.py \
        --dataset "$DATASET" \
        --num_devices $NUM_DEVICES \
        --num_zones $NUM_ZONES \
        --num_rounds $NUM_ROUNDS \
        --intra_zone_alpha $INTRA_ALPHA \
        --inter_zone_alpha $INTER_ALPHA \
        --enable_compression \
        --compression_rate "$RATE" \
        $ES_FLAGS \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: experiment compression_rate=$RATE failed (exit $EXIT_CODE). Continuing..."
    else
        echo "Done: compression_rate=$RATE"
    fi
done

echo ""
echo "========================================================"
echo "All compression experiments finished."
echo "Results in: $RESULTS_DIR"
echo "========================================================"
