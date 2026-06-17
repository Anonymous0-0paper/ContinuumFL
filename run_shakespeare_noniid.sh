#!/bin/bash
# run_shakespeare_noniid.sh
# ─────────────────────────────────────────────────────────────────────────────
# Non-IID inter-zone alpha sweep, dataset-selectable via DATASET env var.
# Mirrors run_noniid_experiments.sh but with production-ready defaults:
#   • batch_size=16  (shared optimum across datasets)
#   • learning_rate=0.001
#   • intra_zone_alpha=100  (near-IID within zones — best from FEMNIST analysis)
#   • inter_zone_alpha swept: 0.1 → 100
#
# Usage:
#   ./run_shakespeare_noniid.sh [NUM_ROUNDS]
#   DATASET=shakespeare ./run_shakespeare_noniid.sh 200
#   DATASET=femnist     ./run_shakespeare_noniid.sh
#   DATASET=cifar100    ./run_shakespeare_noniid.sh
#   DATASET=ucihar      ./run_shakespeare_noniid.sh
#   DATASET=speechcommands ./run_shakespeare_noniid.sh
# ─────────────────────────────────────────────────────────────────────────────

DATASET="${DATASET:-shakespeare}"
NUM_ROUNDS=${1:-200}
NUM_DEVICES=50
NUM_ZONES=5
INTRA_ALPHA=100             # near-IID within zones (best from FEMNIST comparison)
RESULTS_DIR="./results/${DATASET}_noniid_sweep"

mkdir -p "$RESULTS_DIR"

# ── Dataset-specific extra flags ──────────────────────────────────────────────
DATASET_FLAGS=""
case "$DATASET" in
    shakespeare)
        NUM_SPEAKERS="${NUM_SPEAKERS:-35}"
        DATASET_FLAGS="--shakespeare_num_speakers $NUM_SPEAKERS"
        ;;
    *)
        ;;
esac

# ── Early stopping (always ON — same best values as FEMNIST analysis) ─────────
ES_PATIENCE=${ES_PATIENCE:-10}
ES_DELTA=${ES_DELTA:-0.0001}
ES_FLAGS="--enable_early_stopping \
          --early_stopping_patience  $ES_PATIENCE \
          --early_stopping_min_delta $ES_DELTA"

# ── Inter-zone alpha sweep ────────────────────────────────────────────────────
INTER_ALPHAS=(0.1 0.5 1.0 5.0 10.0 50.0 100.0)

echo "=================================================================="
echo "  ContinuumFL — Non-IID Sweep"
echo "=================================================================="
echo "  Dataset       : $DATASET"
echo "  Rounds        : $NUM_ROUNDS"
echo "  Devices       : $NUM_DEVICES  |  Zones    : $NUM_ZONES"
echo "  Batch size    : 16"
echo "  Learning rate : 0.001"
echo "  intra_alpha   : $INTRA_ALPHA  (near-IID within zones)"
echo "  inter_alphas  : ${INTER_ALPHAS[*]}"
echo "  Early stop    : patience=$ES_PATIENCE  min_delta=$ES_DELTA"
[ -n "$DATASET_FLAGS" ] && echo "  Extra flags   : $DATASET_FLAGS"
echo "=================================================================="

for ALPHA in "${INTER_ALPHAS[@]}"; do
    EXP_NAME="${DATASET}_noniid_alpha${ALPHA}"
    LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- Running: inter_zone_alpha=$ALPHA ---"
    echo "    Log: $LOG_FILE"

    python main.py \
        --dataset          "$DATASET" \
        --num_devices      $NUM_DEVICES \
        --num_zones        $NUM_ZONES \
        --num_rounds       $NUM_ROUNDS \
        --intra_zone_alpha $INTRA_ALPHA \
        --inter_zone_alpha "$ALPHA" \
        --learning_rate    0.001 \
        --batch_size       16 \
        $DATASET_FLAGS \
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
echo "  All $DATASET non-IID experiments finished."
echo "  Results in: $RESULTS_DIR"
echo "=================================================================="
