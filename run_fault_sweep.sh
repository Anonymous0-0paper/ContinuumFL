#!/bin/bash
# run_fault_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
# Studies the impact of device and zone fault rates on ContinuumFL performance.
#
# Sweeps over (device_failure_probability, zone_failure_probability) pairs:
#
#   dev_fail │ zone_fail │ Notes
#   ─────────┼───────────┼─────────────────────────────────────────────────
#     0.00   │  0.00     │ fault-free baseline
#     0.05   │  0.00     │ low device fault, no zone fault
#     0.10   │  0.00     │ moderate device fault, no zone fault
#     0.20   │  0.00     │ high device fault, no zone fault
#     0.05   │  0.02     │ low device + low zone fault  (default values)
#     0.10   │  0.05     │ moderate device + moderate zone fault
#     0.20   │  0.10     │ high device + high zone fault
#     0.30   │  0.15     │ severe fault scenario
#
# All other hyperparameters are fixed to the best values discovered so far:
#   • dataset          = femnist  (or $1)
#   • num_devices      = 50
#   • num_zones        = 5
#   • intra_zone_alpha = 100  (near-IID within zones)
#   • inter_zone_alpha = 5.0  (best from non-IID sweep)
#   • compression_rate = 0.10
#   • early_stopping   = ON   (patience=10, min_delta=0.0001)
#
# Usage:
#   ./run_fault_sweep.sh [DATASET] [NUM_ROUNDS]
#   ./run_fault_sweep.sh femnist 200
# ─────────────────────────────────────────────────────────────────────────────

DATASET=${1:-femnist}
NUM_ROUNDS=${2:-200}
NUM_DEVICES=50
NUM_ZONES=5
MIN_ZONE=4
MAX_ZONE=15
INTRA_ALPHA=100
INTER_ALPHA=5.0
COMP_RATE=0.60
LEARNING_RATE=0.001
BASE_RESULTS_DIR="./results/fault_sweep"

mkdir -p "$BASE_RESULTS_DIR"

# ── Early stopping ─────────────────────────────────────────────────────────
ES_PATIENCE=${ES_PATIENCE:-10}
ES_DELTA=${ES_DELTA:-0.0001}
ES_FLAGS="--enable_early_stopping \
          --early_stopping_patience  $ES_PATIENCE \
          --early_stopping_min_delta $ES_DELTA"

# ── Fault-rate configurations ───────────────────────────────────────────────
# Format: "DEVICE_FAIL_PROB:ZONE_FAIL_PROB"
CONFIGS=(
    "0.00:0.00"   # fault-free baseline
    "0.05:0.00"   # low device fault only
    "0.10:0.00"   # moderate device fault only
    "0.20:0.00"   # high device fault only
    "0.05:0.02"   # low device + low zone (default values)
    "0.10:0.05"   # moderate device + moderate zone
    "0.20:0.10"   # high device + high zone
    "0.30:0.15"   # severe fault scenario
)

echo "=================================================================="
echo "  ContinuumFL — Device & Zone Fault Rate Sweep"
echo "=================================================================="
echo "  Dataset       : $DATASET"
echo "  Rounds        : $NUM_ROUNDS"
echo "  Devices       : $NUM_DEVICES  |  Zones : $NUM_ZONES"
echo "  intra_alpha   : $INTRA_ALPHA  (near-IID within zones)"
echo "  inter_alpha   : $INTER_ALPHA  (best from non-IID sweep)"
echo "  Compression   : $COMP_RATE"
echo "  Early stop    : patience=$ES_PATIENCE  min_delta=$ES_DELTA"
echo ""
echo "  Configurations to run:"
echo "  dev_fail │ zone_fail │ Label"
echo "  ─────────┼───────────┼──────────────────────────────────"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r DEV_FAIL ZONE_FAIL <<< "$CFG"
    if   [[ "$DEV_FAIL" == "0.00" && "$ZONE_FAIL" == "0.00" ]]; then LABEL="fault-free baseline"
    elif [[ "$ZONE_FAIL" == "0.00" ]];                               then LABEL="device fault only"
    elif [[ "$DEV_FAIL" == "0.30" ]];                               then LABEL="severe fault scenario"
    else                                                                   LABEL="device + zone fault"
    fi
    printf "   %5s   │   %5s   │ %s\n" "$DEV_FAIL" "$ZONE_FAIL" "$LABEL"
done
echo "=================================================================="

TOTAL=${#CONFIGS[@]}
IDX=0

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r DEV_FAIL ZONE_FAIL <<< "$CFG"
    IDX=$((IDX + 1))

    # Sanitise floats for use in directory/file names (0.05 → dev0.05_zone0.00)
    EXP_NAME="${DATASET}_dev${DEV_FAIL}_zone${ZONE_FAIL}"
    LOG_FILE="$BASE_RESULTS_DIR/${EXP_NAME}.log"

    # Each run gets its own results sub-directory so CSVs never collide
    RUN_RESULTS_DIR="$BASE_RESULTS_DIR/$EXP_NAME"
    mkdir -p "$RUN_RESULTS_DIR"

    echo ""
    echo "--- [$IDX/$TOTAL] dev_fail=$DEV_FAIL  zone_fail=$ZONE_FAIL ---"
    echo "    Log    : $LOG_FILE"
    echo "    CSVs   : $RUN_RESULTS_DIR/"

    # Build failure flags — omit --enable_failure for the fault-free baseline
    FAIL_FLAGS=""
    if [[ "$DEV_FAIL" != "0.00" || "$ZONE_FAIL" != "0.00" ]]; then
        FAIL_FLAGS="--enable_failure \
                    --device_failure_probability $DEV_FAIL \
                    --zone_failure_probability   $ZONE_FAIL"
    fi

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
        --enable_compression \
        --results_dir      "$RUN_RESULTS_DIR" \
        $FAIL_FLAGS \
        $ES_FLAGS \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: dev_fail=$DEV_FAIL zone_fail=$ZONE_FAIL failed (exit $EXIT_CODE). Continuing..."
    else
        echo "Done: dev_fail=$DEV_FAIL  zone_fail=$ZONE_FAIL"
    fi
done

# ── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  All fault-sweep experiments finished."
echo "  Results in: $BASE_RESULTS_DIR"
echo ""
echo "  Summary:"
echo "  dev_fail │ zone_fail │ BestAcc"
echo "  ─────────┼───────────┼────────"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r DEV_FAIL ZONE_FAIL <<< "$CFG"
    EXP_NAME="${DATASET}_dev${DEV_FAIL}_zone${ZONE_FAIL}"
    RUN_RESULTS_DIR="$BASE_RESULTS_DIR/$EXP_NAME"

    # _results_dir() inside the coordinator appends its own sub-path;
    # find the deepest summary.csv under each run's results dir.
    SUMMARY=$(find "$RUN_RESULTS_DIR" -name "summary.csv" 2>/dev/null | head -1)
    if [ -f "$SUMMARY" ]; then
        ACC=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$SUMMARY')) if r.get('best_accuracy')]
print(f\"{float(rows[-1]['best_accuracy'])*100:.2f}%\" if rows else 'N/A')
" 2>/dev/null)
    else
        ACC="(no summary yet)"
    fi
    printf "   %5s   │   %5s   │ %s\n" "$DEV_FAIL" "$ZONE_FAIL" "$ACC"
done
echo "=================================================================="
