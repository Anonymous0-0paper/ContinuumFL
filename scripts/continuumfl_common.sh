#!/usr/bin/env bash

# Shared helpers for ContinuumFL launch scripts.
# Source this file from local or Slurm wrappers.

set -euo pipefail

bool_true() {
    case "${1:-false}" in
        true|TRUE|1|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

append_bool_flag() {
    local flag_name=$1
    local flag_value=$2
    if bool_true "$flag_value"; then
        MAIN_ARGS+=("$flag_name")
    fi
}

apply_preset_defaults() {
    local preset=${1:-standard}

    case "$preset" in
        quick)
            DATASET="femnist"
            MAX_SAMPLES=5000
            NUM_DEVICES=10
            NUM_ZONES=2
            MIN_ZONE_SIZE=2
            MAX_ZONE_SIZE=8
            NUM_ROUNDS=10
            LOCAL_EPOCHS=5
            BATCH_SIZE=64
            LEARNING_RATE=0.01
            SPATIAL_WEIGHT=0.4
            DATA_WEIGHT=0.4
            NETWORK_WEIGHT=0.2
            SPATIAL_REGULARIZATION=0.1
            CORRELATION_THRESHOLD=0.05
            COMPRESSION_RATE=0.1
            ENABLE_COMPRESSION=false
            INTRA_ZONE_ALPHA=100
            INTER_ZONE_ALPHA=10
            ASYNC_AGGREGATION=false
            ENABLE_FAILURE=false
            DEVICE_FAILURE_PROBABILITY=0.05
            ZONE_FAILURE_PROBABILITY=0.02
            SHAKESPEARE_NUM_SPEAKERS=35
            RUN_BASELINES=false
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=false
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        standard)
            DATASET="femnist"
            MAX_SAMPLES=-1
            NUM_DEVICES=100
            NUM_ZONES=20
            MIN_ZONE_SIZE=4
            MAX_ZONE_SIZE=15
            NUM_ROUNDS=100
            LOCAL_EPOCHS=5
            BATCH_SIZE=64
            LEARNING_RATE=0.01
            SPATIAL_WEIGHT=0.4
            DATA_WEIGHT=0.4
            NETWORK_WEIGHT=0.2
            SPATIAL_REGULARIZATION=0.1
            CORRELATION_THRESHOLD=0.05
            COMPRESSION_RATE=0.1
            ENABLE_COMPRESSION=false
            INTRA_ZONE_ALPHA=100
            INTER_ZONE_ALPHA=10
            ASYNC_AGGREGATION=false
            ENABLE_FAILURE=false
            DEVICE_FAILURE_PROBABILITY=0.05
            ZONE_FAILURE_PROBABILITY=0.02
            SHAKESPEARE_NUM_SPEAKERS=35
            RUN_BASELINES=false
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=true
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        large)
            DATASET="femnist"
            MAX_SAMPLES=-1
            NUM_DEVICES=500
            NUM_ZONES=50
            MIN_ZONE_SIZE=4
            MAX_ZONE_SIZE=15
            NUM_ROUNDS=150
            LOCAL_EPOCHS=3
            BATCH_SIZE=64
            LEARNING_RATE=0.01
            SPATIAL_WEIGHT=0.4
            DATA_WEIGHT=0.4
            NETWORK_WEIGHT=0.2
            SPATIAL_REGULARIZATION=0.15
            CORRELATION_THRESHOLD=0.05
            COMPRESSION_RATE=0.1
            ENABLE_COMPRESSION=false
            INTRA_ZONE_ALPHA=100
            INTER_ZONE_ALPHA=10
            ASYNC_AGGREGATION=false
            ENABLE_FAILURE=false
            DEVICE_FAILURE_PROBABILITY=0.05
            ZONE_FAILURE_PROBABILITY=0.02
            SHAKESPEARE_NUM_SPEAKERS=35
            RUN_BASELINES=false
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=true
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        baseline)
            DATASET="femnist"
            MAX_SAMPLES=-1
            NUM_DEVICES=100
            NUM_ZONES=20
            MIN_ZONE_SIZE=4
            MAX_ZONE_SIZE=15
            NUM_ROUNDS=100
            LOCAL_EPOCHS=5
            BATCH_SIZE=64
            LEARNING_RATE=0.01
            SPATIAL_WEIGHT=0.4
            DATA_WEIGHT=0.4
            NETWORK_WEIGHT=0.2
            SPATIAL_REGULARIZATION=0.1
            CORRELATION_THRESHOLD=0.05
            COMPRESSION_RATE=0.1
            ENABLE_COMPRESSION=false
            INTRA_ZONE_ALPHA=100
            INTER_ZONE_ALPHA=10
            ASYNC_AGGREGATION=false
            ENABLE_FAILURE=false
            DEVICE_FAILURE_PROBABILITY=0.05
            ZONE_FAILURE_PROBABILITY=0.02
            SHAKESPEARE_NUM_SPEAKERS=35
            RUN_BASELINES=true
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=true
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        comm)
            DATASET="femnist"
            MAX_SAMPLES=-1
            NUM_DEVICES=100
            NUM_ZONES=20
            MIN_ZONE_SIZE=4
            MAX_ZONE_SIZE=15
            NUM_ROUNDS=100
            LOCAL_EPOCHS=5
            BATCH_SIZE=64
            LEARNING_RATE=0.01
            SPATIAL_WEIGHT=0.4
            DATA_WEIGHT=0.4
            NETWORK_WEIGHT=0.2
            SPATIAL_REGULARIZATION=0.1
            CORRELATION_THRESHOLD=0.05
            COMPRESSION_RATE=0.05
            ENABLE_COMPRESSION=true
            INTRA_ZONE_ALPHA=100
            INTER_ZONE_ALPHA=10
            ASYNC_AGGREGATION=false
            ENABLE_FAILURE=false
            DEVICE_FAILURE_PROBABILITY=0.05
            ZONE_FAILURE_PROBABILITY=0.02
            SHAKESPEARE_NUM_SPEAKERS=35
            RUN_BASELINES=false
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=true
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        femnist)
            DATASET="femnist"
            MAX_SAMPLES=-1
            NUM_DEVICES=100
            NUM_ZONES=20
            MIN_ZONE_SIZE=4
            MAX_ZONE_SIZE=15
            NUM_ROUNDS=100
            LOCAL_EPOCHS=5
            BATCH_SIZE=64
            LEARNING_RATE=0.01
            SPATIAL_WEIGHT=0.4
            DATA_WEIGHT=0.4
            NETWORK_WEIGHT=0.2
            SPATIAL_REGULARIZATION=0.1
            CORRELATION_THRESHOLD=0.05
            COMPRESSION_RATE=0.1
            ENABLE_COMPRESSION=false
            INTRA_ZONE_ALPHA=100
            INTER_ZONE_ALPHA=10
            ASYNC_AGGREGATION=false
            ENABLE_FAILURE=false
            DEVICE_FAILURE_PROBABILITY=0.05
            ZONE_FAILURE_PROBABILITY=0.02
            SHAKESPEARE_NUM_SPEAKERS=35
            RUN_BASELINES=true
            BASELINES_ONLY=false
            BASELINE_METHODS=(FedAvg FedProx HierFL ClusterFL)
            CREATE_VISUALIZATIONS=true
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        shakespeare)
            DATASET="shakespeare"
            MAX_SAMPLES=-1
            NUM_DEVICES=35
            NUM_ZONES=7
            MIN_ZONE_SIZE=2
            MAX_ZONE_SIZE=8
            NUM_ROUNDS=100
            LOCAL_EPOCHS=5
            BATCH_SIZE=32
            LEARNING_RATE=0.01
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
            CREATE_VISUALIZATIONS=true
            SAVE_RESULTS=true
            DEVICE="cuda"
            ;;
        *)
            echo "Unknown preset: $preset" >&2
            echo "Available presets: quick, standard, large, baseline, comm, femnist, shakespeare" >&2
            return 1
            ;;
    esac
}

build_main_args() {
    MAIN_ARGS=(
        --dataset "$DATASET"
        --max_samples "$MAX_SAMPLES"
        --num_devices "$NUM_DEVICES"
        --num_zones "$NUM_ZONES"
        --min_zone_size "$MIN_ZONE_SIZE"
        --max_zone_size "$MAX_ZONE_SIZE"
        --num_rounds "$NUM_ROUNDS"
        --local_epochs "$LOCAL_EPOCHS"
        --learning_rate "$LEARNING_RATE"
        --batch_size "$BATCH_SIZE"
        --spatial_weight "$SPATIAL_WEIGHT"
        --data_weight "$DATA_WEIGHT"
        --network_weight "$NETWORK_WEIGHT"
        --spatial_regularization "$SPATIAL_REGULARIZATION"
        --correlation_threshold "$CORRELATION_THRESHOLD"
        --compression_rate "$COMPRESSION_RATE"
        --intra_zone_alpha "$INTRA_ZONE_ALPHA"
        --inter_zone_alpha "$INTER_ZONE_ALPHA"
        --device "$DEVICE"
        --random_seed "$RANDOM_SEED"
        --log_dir "$LOG_DIR"
        --results_dir "$RESULTS_DIR"
        --checkpoint_dir "$CHECKPOINT_DIR"
    )

    if [[ "$ENABLE_COMPRESSION" == true ]]; then
        MAIN_ARGS+=(--enable_compression)
    fi

    if [[ "$ASYNC_AGGREGATION" == true ]]; then
        MAIN_ARGS+=(--async_aggregation)
    fi

    if [[ "$ENABLE_FAILURE" == true ]]; then
        MAIN_ARGS+=(--enable_failure)
    fi

    if [[ "$RUN_BASELINES" == true ]]; then
        MAIN_ARGS+=(--run_baselines)
    fi

    if [[ "$BASELINES_ONLY" == true ]]; then
        MAIN_ARGS+=(--baselines_only)
    fi

    if [[ "$SAVE_RESULTS" == true ]]; then
        MAIN_ARGS+=(--save_results)
    fi

    if [[ "$CREATE_VISUALIZATIONS" == true ]]; then
        MAIN_ARGS+=(--create_visualizations)
    fi

    if [[ "$DATASET" == "shakespeare" ]]; then
        MAIN_ARGS+=(--shakespeare_num_speakers "$SHAKESPEARE_NUM_SPEAKERS")
    fi

    if [[ ${#BASELINE_METHODS[@]} -gt 0 ]]; then
        MAIN_ARGS+=(--baseline_methods "${BASELINE_METHODS[@]}")
    fi
}
