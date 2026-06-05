#!/bin/bash
# Training script for UCI HAR and Google Speech Commands datasets in ContinuumFL.
# Usage:
#   bash run_new_datasets.sh              # interactive menu
#   bash run_new_datasets.sh ucihar       # run UCI HAR directly
#   bash run_new_datasets.sh speech       # run Speech Commands directly
#   bash run_new_datasets.sh both         # run both sequentially
#   bash run_new_datasets.sh quick        # small-scale smoke-test for both

set -euo pipefail

# ── GPU detection ─────────────────────────────────────────────────────────────
USE_DEVICE="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    USE_DEVICE="cuda"
elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi -i 2>/dev/null | grep -q GPU; then
    USE_DEVICE="cuda"
fi

if [ "$USE_DEVICE" = "cuda" ]; then
    echo "GPU detected — using CUDA."
    DEVICE_ARG="--device cuda"
else
    echo "No GPU detected — using CPU."
    DEVICE_ARG="--device cpu"
fi

# ── Helper ────────────────────────────────────────────────────────────────────
run_experiment() {
    local name=$1
    local cmd=$2
    local log_dir="./results/${name}"
    mkdir -p "$log_dir"
    echo ""
    echo "============================================================"
    echo "  Experiment : $name"
    echo "  Log        : $log_dir/train.log"
    echo "============================================================"
    # shellcheck disable=SC2086
    eval "$cmd $DEVICE_ARG --results_dir \"$log_dir\"" 2>&1 | tee "$log_dir/train.log"
    echo "  Done. Results in $log_dir"
}

# ── Experiment definitions ────────────────────────────────────────────────────

run_ucihar() {
    # UCI HAR: 30 subjects → 30 natural FL clients; 6 activity classes.
    # 50 devices across 10 zones with mild non-IID (intra=50, inter=5).
    run_experiment "ucihar_standard" "python main.py \
        --dataset ucihar \
        --num_devices 30 \
        --num_zones 6 \
        --min_zone_size 3 \
        --max_zone_size 8 \
        --num_rounds 100 \
        --local_epochs 5 \
        --learning_rate 0.001 \
        --batch_size 64 \
        --intra_zone_alpha 50 \
        --inter_zone_alpha 5 \
        --compression_rate 0.5 \
        --enable_early_stopping \
        --early_stopping_patience 15 \
        --save_results"
}

run_speechcommands() {
    # Speech Commands v2: 35-keyword classification from log-mel spectrograms.
    # Partitioned by speaker so each FL device represents one speaker shard.
    run_experiment "speechcommands_standard" "python main.py \
        --dataset speechcommands \
        --num_devices 40 \
        --num_zones 8 \
        --min_zone_size 3 \
        --max_zone_size 8 \
        --num_rounds 100 \
        --local_epochs 3 \
        --learning_rate 0.001 \
        --batch_size 32 \
        --intra_zone_alpha 100 \
        --inter_zone_alpha 10 \
        --compression_rate 0.5 \
        --enable_early_stopping \
        --early_stopping_patience 15 \
        --save_results"
}

run_quick() {
    # Tiny smoke-test — downloads data (if needed) and runs a few rounds.
    run_experiment "ucihar_quick" "python main.py \
        --dataset ucihar \
        --max_samples 1000 \
        --num_devices 10 \
        --num_zones 3 \
        --min_zone_size 2 \
        --max_zone_size 5 \
        --num_rounds 5 \
        --local_epochs 2 \
        --learning_rate 0.001 \
        --batch_size 32 \
        --save_results"

    run_experiment "speechcommands_quick" "python main.py \
        --dataset speechcommands \
        --max_samples 500 \
        --num_devices 10 \
        --num_zones 3 \
        --min_zone_size 2 \
        --max_zone_size 5 \
        --num_rounds 5 \
        --local_epochs 2 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --save_results"
}

# ── Menu / CLI dispatch ───────────────────────────────────────────────────────
MODE="${1:-menu}"

case "$MODE" in
    ucihar)  run_ucihar ;;
    speech)  run_speechcommands ;;
    both)    run_ucihar; run_speechcommands ;;
    quick)   run_quick ;;
    menu)
        echo ""
        echo "Available training runs:"
        echo "  1) UCI HAR          — time-series activity recognition (1D-CNN + LSTM)"
        echo "  2) Speech Commands  — keyword spotting on mel-spectrograms (2D-CNN)"
        echo "  3) Both             — UCI HAR then Speech Commands"
        echo "  4) Quick smoke-test — tiny subsets of both (fast, no GPU needed)"
        echo ""
        read -rp "Select [1-4]: " choice
        case "$choice" in
            1) run_ucihar ;;
            2) run_speechcommands ;;
            3) run_ucihar; run_speechcommands ;;
            4) run_quick ;;
            *) echo "Invalid choice."; exit 1 ;;
        esac
        ;;
    *)
        echo "Usage: $0 [ucihar|speech|both|quick]"
        exit 1
        ;;
esac

echo ""
echo "All done. Results are in ./results/"
