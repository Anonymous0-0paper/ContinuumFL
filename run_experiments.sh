#!/bin/bash
# ContinuumFL Experimental Run Scripts
# This file contains different experimental scenarios for ContinuumFL

echo "🚀 ContinuumFL Experimental Run Scripts"
echo "========================================"

# GPU detection and selection
GPU_AVAILABLE=0
GPU_CMD=""
GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        GPU_AVAILABLE=1
        GPU_CMD="nvidia"
    fi
elif command -v rocm-smi >/dev/null 2>&1; then
    GPU_COUNT=$(rocm-smi -i 2>/dev/null | grep -c '^GPU')
    if [ "$GPU_COUNT" -gt 0 ]; then
        GPU_AVAILABLE=1
        GPU_CMD="rocm"
    fi
fi

if [ "$GPU_AVAILABLE" -eq 1 ]; then
    if [ -t 0 ]; then
        echo "Detected $GPU_COUNT GPU(s) ($GPU_CMD). Use GPUs? [Y/n]"
        read -r use_gpu
        if [ -z "$use_gpu" ] || [[ "$use_gpu" =~ ^[Yy] ]]; then
            echo "Enter GPU ids to use (comma-separated) or press Enter for '0':"
            read -r gpu_ids
            if [ -z "$gpu_ids" ]; then gpu_ids="0"; fi
            export CUDA_VISIBLE_DEVICES="$gpu_ids"
            echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
            USE_DEVICE_ARG="--device cuda"
        else
            USE_DEVICE_ARG="--device cpu"
        fi
    else
        # Non-interactive / piped runs default to GPU 0 if available
        export CUDA_VISIBLE_DEVICES="0"
        USE_DEVICE_ARG="--device cuda"
    fi
else
    echo "No GPUs detected. Using CPU."
    USE_DEVICE_ARG="--device cpu"
fi

# Function to run experiment with logging
run_experiment() {
    local exp_name=$1
    local cmd=$2
    
    echo "📊 Running experiment: $exp_name"
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    # Create experiment-specific results directory
    mkdir -p "./results/$exp_name"
    
    # Run experiment and log output (append device arg determined above)
    eval "$cmd $USE_DEVICE_ARG --results_dir \"./results/$exp_name\"" 2>&1 | tee "./results/$exp_name/experiment.log"
    
    echo "✅ Experiment $exp_name completed"
    echo "Results saved to: ./results/$exp_name"
    echo ""
}

# Early stopping defaults (can be overridden by the user below)
ES_FLAGS=""
if [ -t 0 ]; then
    echo ""
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

# Experiment 1: Quick Test (Small scale for verification)
echo ""
echo "🧪 Available Experiments:"
echo "1. Quick Test"
echo "2. Standard CIFAR-100 Experiment"
echo "3. Large Scale Experiment"
echo "4. Communication Efficiency Study"
echo "5. Baseline Comparison"
echo "6. All Experiments"
echo ""

read -p "Select experiment (1-6): " choice

case $choice in
    1)
        echo "Running Quick Test..."
        run_experiment "quick_test" "python main.py \
            --dataset shakespeare \
            --num_devices 20 \
            --num_zones 5 \
            --num_rounds 20 \
            --local_epochs 5 \
            --batch_size 16 \
            --create_visualizations $ES_FLAGS"
        ;;

    2)
        echo "Running Standard CIFAR-100 Experiment..."
        run_experiment "standard_cifar100" "python main.py \
            --dataset cifar100 \
            --num_devices 100 \
            --num_zones 20 \
            --num_rounds 200 \
            --local_epochs 5 \
            --learning_rate 0.01 \
            --spatial_regularization 0.1 \
            --compression_rate 0.1 \
            --create_visualizations $ES_FLAGS"
        ;;

    3)
        echo "Running Large Scale Experiment..."
        run_experiment "large_scale" "python main.py \
            --dataset cifar100 \
            --num_devices 500 \
            --num_zones 50 \
            --num_rounds 300 \
            --local_epochs 3 \
            --batch_size 64 \
            --spatial_regularization 0.15 \
            --create_visualizations $ES_FLAGS"
        ;;

    4)
        echo "Running Communication Efficiency Study..."
        for comp_rate in 0.05 0.1 0.2 0.3; do
            run_experiment "comm_study_${comp_rate}" "python main.py \
                --dataset cifar100 \
                --num_devices 100 \
                --num_zones 20 \
                --num_rounds 150 \
                --compression_rate $comp_rate \
                --create_visualizations $ES_FLAGS"
        done
        ;;

    5)
        echo "Running Baseline Comparison..."
        run_experiment "baseline_comparison" "python main.py \
            --dataset cifar100 \
            --num_devices 100 \
            --num_zones 20 \
            --num_rounds 200 \
            --run_baselines \
            --create_visualizations $ES_FLAGS"
        ;;

    6)
        echo "Running All Experiments..."

        # Quick test first
        run_experiment "quick_test" "python main.py --dataset shakespeare --num_devices 20 --num_zones 5 --num_rounds 10 $ES_FLAGS"

        # Standard experiment
        run_experiment "standard_cifar100" "python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 200 --create_visualizations $ES_FLAGS"

        # With baselines
        run_experiment "with_baselines" "python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 150 --run_baselines --create_visualizations $ES_FLAGS"

        # Different datasets
        run_experiment "femnist_experiment" "python main.py --dataset femnist --num_devices 80 --num_zones 16 --num_rounds 100 --create_visualizations $ES_FLAGS"
        
        echo "📊 Creating comparison report..."
        python -c "
import os
import json
import matplotlib.pyplot as plt

# Collect results from all experiments
experiments = ['quick_test', 'standard_cifar100', 'with_baselines', 'femnist_experiment']
results = {}

for exp in experiments:
    result_file = f'./results/{exp}/experiment_results.json'
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results[exp] = json.load(f)

# Create comparison plot
if results:
    plt.figure(figsize=(12, 8))
    for exp, data in results.items():
        if 'continuum_fl_results' in data:
            final_acc = data['continuum_fl_results'].get('final_accuracy', 0)
            plt.bar(exp, final_acc, alpha=0.7)
    
    plt.ylabel('Final Accuracy')
    plt.title('ContinuumFL Experiment Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./results/experiment_comparison.png', dpi=300, bbox_inches='tight')
    print('📈 Comparison plot saved to ./results/experiment_comparison.png')
"
        ;;
    
    *)
        echo "❌ Invalid choice. Please select 1-6."
        exit 1
        ;;
esac

echo "🎉 Selected experiments completed!"
echo "📁 Results available in ./results/"
echo ""
echo "📊 To view results:"
echo "   - Check ./results/[experiment_name]/ for detailed results"
echo "   - View .png files for visualizations" 
echo "   - Check experiment.log for detailed logs"
echo ""
echo "🔧 To run custom experiments:"
echo "   python main.py --help"