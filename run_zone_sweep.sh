#!/bin/bash
# run_zone_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
# Studies the impact of zone granularity on ContinuumFL performance.
# Sweeps over (num_devices, num_zones) configurations:
#
#   Clients │ Zones │ Avg devices/zone │ Notes
#   ────────┼───────┼──────────────────┼──────────────────────────────────────
#      10   │   2   │       5          │ small scale, coarse zoning
#      10   │   5   │       2          │ small scale, fine zoning
#      20   │   4   │       5          │ medium scale, coarse zoning
#      20   │   10  │       2          │ medium scale, fine zoning
#      50   │   5   │      10          │ BEST from non-IID comparison (baseline)
#      50   │   10  │       5          │ larger-scale fine zoning
#      50   │   25  │       2          │ very fine zoning, 1 device/zone edge case
#
# All other hyperparameters are fixed to the best values from
# compare_noniid_results.py:
#   • inter_zone_alpha = 5.0   (moderate non-IID)
#   • intra_zone_alpha = 100   (near-IID within zones)
#   • compression_rate = 0.10  (10% top-k)
#   • early_stopping   = ON    (patience=10, min_delta=0.0001)
#
# Usage:
#   ./run_zone_sweep.sh [DATASET] [NUM_ROUNDS]
#   ./run_zone_sweep.sh femnist 200
# ─────────────────────────────────────────────────────────────────────────────

DATASET=${1:-femnist}
NUM_ROUNDS=${2:-200}
INTRA_ALPHA=100
INTER_ALPHA=5.0
COMP_RATE=0.60
LEARNING_RATE=0.001
RESULTS_DIR="./results/zone_sweep"

mkdir -p "$RESULTS_DIR"

# ── Early stopping ─────────────────────────────────────────────────────────
ES_PATIENCE=${ES_PATIENCE:-10}
ES_DELTA=${ES_DELTA:-0.0001}
ES_FLAGS="--enable_early_stopping \
          --early_stopping_patience  $ES_PATIENCE \
          --early_stopping_min_delta $ES_DELTA"

# ── Zone / client configurations ────────────────────────────────────────────
# Format: "NUM_DEVICES:NUM_ZONES:MIN_ZONE_SIZE:MAX_ZONE_SIZE"
# min/max zone sizes are tuned so the clustering algorithm can always
# form the requested number of zones from the available devices.
#
#   Rule of thumb used here:
#     min_zone_size = max(2, floor(num_devices / num_zones) - 1)
#     max_zone_size = max(min_zone_size+1, floor(num_devices / num_zones) + 3)
CONFIGS=(
    "10:2:3:6"      #  10 clients, 2 zones  — coarse (5 dev/zone avg)
    "10:5:1:4"      #  10 clients, 5 zones  — fine   (2 dev/zone avg)
    "20:4:3:8"      #  20 clients, 4 zones  — coarse (5 dev/zone avg)
    "20:10:1:4"     #  20 clients, 10 zones — fine   (2 dev/zone avg)
    "50:5:4:15"     #  50 clients, 5 zones  — BEST BASELINE (10 dev/zone avg)
    "50:10:3:8"     #  50 clients, 10 zones — finer  (5 dev/zone avg)
    "50:25:1:4"     #  50 clients, 25 zones — finest (2 dev/zone avg)
)

echo "=================================================================="
echo "  ContinuumFL — Zone & Client Scale Sweep"
echo "=================================================================="
echo "  Dataset       : $DATASET"
echo "  Rounds        : $NUM_ROUNDS"
echo "  intra_alpha   : $INTRA_ALPHA  (near-IID within zones)"
echo "  inter_alpha   : $INTER_ALPHA  (best: moderate non-IID)"
echo "  Compression   : $COMP_RATE"
echo "  Early stop    : patience=$ES_PATIENCE  min_delta=$ES_DELTA"
echo ""
echo "  Configurations to run:"
echo "  Clients │ Zones │ min_zone │ max_zone"
echo "  ────────┼───────┼──────────┼─────────"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r DEVS ZONES MIN_Z MAX_Z <<< "$CFG"
    printf "     %3s  │  %3s  │    %3s   │   %3s\n" "$DEVS" "$ZONES" "$MIN_Z" "$MAX_Z"
done
echo "=================================================================="

TOTAL=${#CONFIGS[@]}
IDX=0

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r NUM_DEVICES NUM_ZONES MIN_ZONE MAX_ZONE <<< "$CFG"
    IDX=$((IDX + 1))

    EXP_NAME="${DATASET}_dev${NUM_DEVICES}_zones${NUM_ZONES}"
    LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"

    echo ""
    echo "--- [$IDX/$TOTAL] Running: $NUM_DEVICES devices / $NUM_ZONES zones ---"
    echo "    Avg devices/zone: $(python3 -c "print(round($NUM_DEVICES/$NUM_ZONES,1))")"
    echo "    Log: $LOG_FILE"

    python main.py \
        --dataset          "$DATASET" \
        --num_devices      $NUM_DEVICES \
        --num_zones        $NUM_ZONES \
        --min_zone_size    $MIN_ZONE \
        --max_zone_size    $MAX_ZONE \
        --num_rounds       $NUM_ROUNDS \
        --intra_zone_alpha $INTRA_ALPHA \
        --inter_zone_alpha $INTER_ALPHA \
        --learning_rate    $LEARNING_RATE \
        --compression_rate $COMP_RATE \
        $ES_FLAGS \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: config dev=${NUM_DEVICES} zones=${NUM_ZONES} failed (exit $EXIT_CODE). Continuing..."
    else
        echo "Done: $NUM_DEVICES devices / $NUM_ZONES zones"
    fi
done

echo ""
echo "=================================================================="
echo "  All zone-sweep experiments finished."
echo "  Results in: $RESULTS_DIR"
echo ""
echo "  Summary of structured results:"
COMP_PCT=$(python3 -c "print(int($COMP_RATE*100))")
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r DEVS ZONES _ _ <<< "$CFG"
    SUMMARY="./results/${DATASET}__intra${INTRA_ALPHA}.0__inter${INTER_ALPHA}__comp${COMP_PCT}pct__dev${DEVS}__zones${ZONES}/summary.csv"
    if [ -f "$SUMMARY" ]; then
        ACC=$(python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$SUMMARY')) if r.get('best_accuracy')]
if rows: print(f\"{float(rows[-1]['best_accuracy'])*100:.2f}%\")
else: print('N/A')
" 2>/dev/null)
        echo "    dev=$DEVS  zones=$ZONES  → BestAcc: $ACC"
    fi
done
echo "=================================================================="

