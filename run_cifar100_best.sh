#!/bin/bash
# run_cifar100_noniid.sh
# ─────────────────────────────────────────────────────────────────────────────
# Non-IID sweep on the CIFAR-100 dataset (ResNet18, 100 classes).
# Mirrors run_noniid_experiments.sh / run_shakespeare_noniid.sh in structure,
# sweeping inter_zone_alpha across the same 7 levels for a fair comparison.
#
# Fixed hyperparameters (best values from compare_noniid_results.py on FEMNIST):
#   • intra_zone_alpha = 100    → near-IID within zones
#   • compression_rate = 0.10   → 10% top-k
#   • batch_size       = 64     → ResNet18 + RTX 3060 optimum
#   • learning_rate    = 0.001
#   • early stopping   = ON     → patience=10, min_delta=0.0001
#
# Usage:
#   ./run_cifar100_noniid.sh [NUM_ROUNDS]
#   ./run_cifar100_noniid.sh 200
# ─────────────────────────────────────────────────────────────────────────────

NUM_ROUNDS=${1:-200}
NUM_DEVICES=50
NUM_ZONES=5
INTRA_ALPHA=100      # best: near-IID within zones
COMP_RATE=0.10       # best: 10% top-k compression
BATCH_SIZE=64        # ResNet18 handles 64 comfortably on RTX 3060
LEARNING_RATE=0.001
RESULTS_DIR="./results/cifar100_noniid_sweep"

mkdir -p "$RESULTS_DIR"

# ── Early stopping ─────────────────────────────────────────────────────────
ES_PATIENCE=${ES_PATIENCE:-10}
ES_DELTA=${ES_DELTA:-0.0001}
ES_FLAGS="--enable_early_stopping \
          --early_stopping_patience  $ES_PATIENCE \
          --early_stopping_min_delta $ES_DELTA"

# ── Inter-zone alpha sweep ─────────────────────────────────────────────────
# Same range used for FEMNIST and Shakespeare to allow direct comparison.
INTER_ALPHAS=(0.1 0.5 1.0 5.0 10.0 50.0 100.0)

echo "=================================================================="
echo "  ContinuumFL — CIFAR-100 Non-IID Sweep"
echo "=================================================================="
echo "  Dataset       : cifar100  (ResNet18, 100 classes)"
echo "  Rounds        : $NUM_ROUNDS"
echo "  Devices       : $NUM_DEVICES  |  Zones: $NUM_ZONES"
echo "  Batch size    : $BATCH_SIZE"
echo "  Learning rate : $LEARNING_RATE"
echo "  Compression   : $COMP_RATE  (10% top-k)"
echo "  intra_alpha   : $INTRA_ALPHA  (near-IID within zones)"
echo "  inter_alphas  : ${INTER_ALPHAS[*]}"
echo "  Early stop    : patience=$ES_PATIENCE  min_delta=$ES_DELTA"
echo "=================================================================="

for ALPHA in "${INTER_ALPHAS[@]}"; do
    EXP_NAME="cifar100_noniid_alpha${ALPHA}"
    LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- Running: inter_zone_alpha=$ALPHA ---"
    echo "    Log: $LOG_FILE"

    python main.py \
        --dataset          cifar100 \
        --num_devices      $NUM_DEVICES \
        --num_zones        $NUM_ZONES \
        --num_rounds       $NUM_ROUNDS \
        --intra_zone_alpha $INTRA_ALPHA \
        --inter_zone_alpha "$ALPHA" \
        --learning_rate    $LEARNING_RATE \
        --batch_size       $BATCH_SIZE \
        --compression_rate $COMP_RATE \
        $ES_FLAGS \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: alpha=$ALPHA failed (exit $EXIT_CODE). Continuing..."
    else
        echo "Done: inter_zone_alpha=$ALPHA"
    fi
done

echo ""
echo "=================================================================="
echo "  All CIFAR-100 non-IID experiments finished."
echo "  Results in: $RESULTS_DIR"
echo "=================================================================="
