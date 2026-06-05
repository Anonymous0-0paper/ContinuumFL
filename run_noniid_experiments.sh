#!/bin/bash
# Run ContinuumFL across different non-IID levels (inter_zone_alpha).
# Lower alpha = more heterogeneous data across zones.
# intra_zone_alpha is kept high (100) so devices within a zone stay similar.

DATASET=${1:-femnist}
NUM_ROUNDS=${2:-200}
NUM_DEVICES=50
NUM_ZONES=5
RESULTS_DIR="./results/noniid_sweep"

mkdir -p "$RESULTS_DIR"

# Early stopping
ES_FLAGS=""
if [ -t 0 ]; then
    echo "⏹  Enable early stopping? [y/N]"
    read -r use_es
    if [[ "$use_es" =~ ^[Yy] ]]; then
        echo "   Patience (rounds without improvement, default 20):"
        read -r es_patience
        if [ -z "$es_patience" ]; then es_patience=20; fi
        echo "   Min delta (minimum accuracy gain, default 0.0001):"
        read -r es_delta
        if [ -z "$es_delta" ]; then es_delta=0.0001; fi
        ES_FLAGS="--enable_early_stopping --early_stopping_patience $es_patience --early_stopping_min_delta $es_delta"
        echo "   Early stopping enabled: patience=$es_patience, min_delta=$es_delta"
    fi
fi

# Alpha values: 0.1 (extreme non-IID) → 100 (near-IID)
INTER_ALPHAS=(0.1 0.5 1.0 5.0 10.0 50.0 100.0)

echo "========================================"
echo "ContinuumFL Non-IID Sweep"
echo "Dataset     : $DATASET"
echo "Rounds      : $NUM_ROUNDS"
echo "Inter alphas: ${INTER_ALPHAS[*]}"
echo "Early stop  : ${ES_FLAGS:-disabled}"
echo "========================================"

for ALPHA in "${INTER_ALPHAS[@]}"; do
    EXP_NAME="noniid_alpha${ALPHA}"
    LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- Running: inter_zone_alpha=$ALPHA ---"
    echo "Log: $LOG_FILE"

    python main.py \
        --dataset "$DATASET" \
        --num_devices $NUM_DEVICES \
        --num_zones $NUM_ZONES \
        --num_rounds $NUM_ROUNDS \
        --inter_zone_alpha "$ALPHA" \
        --intra_zone_alpha 100 \
        $ES_FLAGS \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: experiment alpha=$ALPHA failed (exit $EXIT_CODE). Continuing..."
    else
        echo "Done: alpha=$ALPHA"
    fi
done

echo ""
echo "========================================"
echo "All experiments finished."
echo "Results in: $RESULTS_DIR"
echo "========================================"
