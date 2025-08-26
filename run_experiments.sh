#!/bin/bash
# ContinuumFL Experimental Run Scripts
# This file contains different experimental scenarios for ContinuumFL

echo "🚀 ContinuumFL Experimental Run Scripts"
echo "========================================"

# Function to run experiment with logging
run_experiment() {
    local exp_name=$1
    local cmd=$2
    
    echo "📊 Running experiment: $exp_name"
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    # Create experiment-specific results directory
    mkdir -p "./results/$exp_name"
    
    # Run experiment and log output
    eval $cmd --results_dir "./results/$exp_name" 2>&1 | tee "./results/$exp_name/experiment.log"
    
    echo "✅ Experiment $exp_name completed"
    echo "Results saved to: ./results/$exp_name"
    echo ""
}

# Experiment 1: Quick Test (Small scale for verification)
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
            --dataset cifar100 \
            --num_devices 20 \
            --num_zones 5 \
            --num_rounds 10 \
            --local_epochs 2 \
            --batch_size 16 \
            --create_visualizations"
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
            --create_visualizations"
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
            --create_visualizations"
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
                --create_visualizations"
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
            --create_visualizations"
        ;;
    
    6)
        echo "Running All Experiments..."
        
        # Quick test first
        run_experiment "quick_test" "python main.py --dataset cifar100 --num_devices 20 --num_zones 5 --num_rounds 10"
        
        # Standard experiment
        run_experiment "standard_cifar100" "python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 200 --create_visualizations"
        
        # With baselines
        run_experiment "with_baselines" "python main.py --dataset cifar100 --num_devices 100 --num_zones 20 --num_rounds 150 --run_baselines --create_visualizations"
        
        # Different datasets
        run_experiment "femnist_experiment" "python main.py --dataset femnist --num_devices 80 --num_zones 16 --num_rounds 100 --create_visualizations"
        
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