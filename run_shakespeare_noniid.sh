#!/bin/bash
# run_shakespeare_noniid.sh
# ─────────────────────────────────────────────────────────────────────────────
# Non-IID sweep on the Shakespeare dataset.
# Mirrors run_noniid_experiments.sh but with Shakespeare-specific settings:
#   • batch_size=16  (framework auto-tunes this; kept explicit for clarity)
#   • learning_rate=0.001  (LSTM is unstable at higher LRs)
#   • shakespeare_num_speakers=35  (number of distinct "clients" / text sources)
#   • intra_zone_alpha=100  (near-IID within zones — best value from FEMNIST analysis)
#   • inter_zone_alpha swept: 0.1 → 100 (same range as run_noniid_experiments.sh)
#
# Usage:
#   ./run_shakespeare_noniid.sh [NUM_ROUNDS] [NUM_SPEAKERS]
#   ./run_shakespeare_noniid.sh 200 35
# ─────────────────────────────────────────────────────────────────────────────

NUM_ROUNDS=${1:-200}
NUM_SPEAKERS=${2:-35}       # --shakespeare_num_speakers
NUM_DEVICES=50
NUM_ZONES=5
INTRA_ALPHA=100             # near-IID within zones (best from FEMNIST comparison)
RESULTS_DIR="./results/shakespeare_noniid_sweep"

mkdir -p "$RESULTS_DIR"

# ── Early stopping (always ON — same best values as FEMNIST analysis) ─────────
ES_PATIENCE=${ES_PATIENCE:-10}
ES_DELTA=${ES_DELTA:-0.0001}
ES_FLAGS="--enable_early_stopping \
          --early_stopping_patience  $ES_PATIENCE \
          --early_stopping_min_delta $ES_DELTA"

# ── Inter-zone alpha sweep ────────────────────────────────────────────────────
# Same range as run_noniid_experiments.sh for apples-to-apples comparison.
INTER_ALPHAS=(0.1 0.5 1.0 5.0 10.0 50.0 100.0)

echo "=================================================================="
echo "  ContinuumFL — Shakespeare Non-IID Sweep"
echo "=================================================================="
echo "  Dataset       : shakespeare"
echo "  Rounds        : $NUM_ROUNDS"
echo "  Devices       : $NUM_DEVICES  |  Zones    : $NUM_ZONES"
echo "  Speakers      : $NUM_SPEAKERS"
echo "  Batch size    : 16  (Shakespeare optimum)"
echo "  Learning rate : 0.001  (LSTM stability)"
echo "  intra_alpha   : $INTRA_ALPHA  (near-IID within zones)"
echo "  inter_alphas  : ${INTER_ALPHAS[*]}"
echo "  Early stop    : patience=$ES_PATIENCE  min_delta=$ES_DELTA"
echo "=================================================================="

for ALPHA in "${INTER_ALPHAS[@]}"; do
    EXP_NAME="shakespeare_noniid_alpha${ALPHA}"
    LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- Running: inter_zone_alpha=$ALPHA ---"
    echo "    Log: $LOG_FILE"

    python main.py \
        --dataset          shakespeare \
        --num_devices      $NUM_DEVICES \
        --num_zones        $NUM_ZONES \
        --num_rounds       $NUM_ROUNDS \
        --intra_zone_alpha $INTRA_ALPHA \
        --inter_zone_alpha "$ALPHA" \
        --learning_rate    0.001 \
        --batch_size       16 \
        --shakespeare_num_speakers $NUM_SPEAKERS \
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
echo "  All Shakespeare non-IID experiments finished."
echo "  Results in: $RESULTS_DIR"
echo "=================================================================="
