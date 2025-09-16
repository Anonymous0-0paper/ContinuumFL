#!/usr/bin/env python3
"""
Main execution script for ContinuumFL framework.
Runs the complete spatial-aware federated learning experiment.
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import json
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ContinuumFLConfig
from src.continuum_fl_coordinator import ContinuumFLCoordinator
from src.visualization.visualizer import ContinuumFLVisualizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ContinuumFL: Spatial-Aware Federated Learning')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar100', 
                       choices=['cifar100', 'femnist', 'shakespeare'],
                       help='Dataset to use for training')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Limit Dataset-Size')
    
    # System configuration
    parser.add_argument('--num_devices', type=int, default=100,
                       help='Number of edge devices')
    parser.add_argument('--num_zones', type=int, default=20,
                       help='Number of spatial zones')
    parser.add_argument('--min_zone_size', type=int, default=4,
                        help='Minimum number of devices within a zone')
    parser.add_argument('--max_zone_size', type=int, default=15,
                        help='Maximum number of devices within a zone')
    parser.add_argument('--num_rounds', type=int, default=200,
                       help='Number of training rounds')
    
    # Training parameters
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # ContinuumFL specific parameters
    parser.add_argument('--spatial_weight', type=float, default=0.4,
                       help='Weight for spatial similarity')
    parser.add_argument('--data_weight', type=float, default=0.4,
                       help='Weight for data similarity')
    parser.add_argument('--network_weight', type=float, default=0.2,
                       help='Weight for network similarity')
    parser.add_argument('--spatial_regularization', type=float, default=0.1,
                       help='Spatial regularization parameter')
    parser.add_argument('--correlation_threshold', type=float, default=0.05,
                        help='Correlation threshold for neighborship')
    parser.add_argument('--compression_rate', type=float, default=0.1,
                        help='Gradient compression rate')
    parser.add_argument('--enable_compression', action='store_true', default=False,
                        help='Enable compression during parameter transmission')
    parser.add_argument('--intra_zone_alpha', type=float, default=10,
                        help='Dirichlet alpha value within zones')
    parser.add_argument('--inter_zone_alpha', type=float, default=0.3,
                        help='Dirichlet alpha value across zones')

    # Aggregation Settings
    parser.add_argument('--async_aggregation', action='store_true', default=False,
                        help='Enable asynchronous inter zone aggregation')

    # Device Settings
    parser.add_argument('--enable_failure', action='store_true', default=False,
                        help='Enable simulation of failure for devices')
    parser.add_argument('--shakespeare_num_speakers', type=int, default=35,
                        help='Set number of heterogeneous text sources (only available for shakespear dataset)')
    # Experiment options
    parser.add_argument('--run_baselines', action='store_true',
                       help='Run baseline comparison')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save experiment results')
    parser.add_argument('--create_visualizations', action='store_true', default=True,
                       help='Create visualization plots')
    
    # System options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    # Output options
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for log files')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    
    return parser.parse_args()

def check_gpu_availability():
    """Check GPU availability and provide recommendations"""
    print("🔍 Checking GPU availability...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
        memory_free = (torch.cuda.get_device_properties(current_device).total_memory - torch.cuda.memory_allocated(current_device)) / (1024**3)
        
        print(f"✅ CUDA Available: Yes")
        print(f"📟 Device Count: {device_count}")
        print(f"🎯 Current Device: {current_device} ({device_name})")
        print(f"💾 GPU Memory: {memory_free:.2f}GB free / {memory_total:.2f}GB total")
        
        # Memory recommendations
        if memory_free < 2.0:
            print(f"⚠️  Warning: Low GPU memory ({memory_free:.2f}GB). Consider:")
            print(f"   - Using smaller batch sizes (--batch_size 16 or 8)")
            print(f"   - Reducing number of devices (--num_devices 50)")
            print(f"   - Using CPU instead (--device cpu)")
        elif memory_free < 4.0:
            print(f"💡 Moderate GPU memory. Recommended settings:")
            print(f"   - Batch size: 32 or lower")
            print(f"   - Max devices: 100-200")
        else:
            print(f"🚀 Excellent GPU memory! You can use larger experiments.")
        
        return 'cuda', True
    else:
        print(f"❌ CUDA Available: No")
        print(f"💻 Will use CPU for training")
        print(f"💡 For faster training, consider:")
        print(f"   - Installing CUDA-enabled PyTorch")
        print(f"   - Using Google Colab or cloud GPU instances")
        print(f"   - Running smaller experiments on CPU")
        
        return 'cpu', False

def setup_configuration(args) -> ContinuumFLConfig:
    """Setup configuration from arguments"""
    
    if args.config_file and os.path.exists(args.config_file):
        # Load from config file
        config = ContinuumFLConfig.load_config(args.config_file)
        print(f"Configuration loaded from {args.config_file}")
    else:
        # Create from command line arguments
        config = ContinuumFLConfig()
    
    # Override with command line arguments
    config.dataset_name = args.dataset
    config.num_devices = args.num_devices
    config.num_zones = args.num_zones
    config.min_zone_size = args.min_zone_size
    config.max_zone_size = args.max_zone_size
    config.num_rounds = args.num_rounds
    config.local_epochs = args.local_epochs
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples

    # ContinuumFL parameters
    config.similarity_weights['spatial'] = args.spatial_weight
    config.similarity_weights['data'] = args.data_weight
    config.similarity_weights['network'] = args.network_weight
    config.spatial_regularization = args.spatial_regularization
    config.correlation_threshold = args.correlation_threshold
    config.compression_rate = args.compression_rate
    config.enable_compression = args.enable_compression
    config.intra_zone_alpha = args.intra_zone_alpha
    config.inter_zone_alpha = args.inter_zone_alpha

    # Aggregation Settings
    config.async_aggregation = args.async_aggregation

    # Device Settings
    config.enable_failure = args.enable_failure
    config.shakespeare_num_speakers = args.shakespeare_num_speakers
    # System options - GPU checking will be done separately
    config.device = args.device
    config.random_seed = args.random_seed
    
    # Output directories
    config.log_dir = args.log_dir
    config.results_dir = args.results_dir
    config.checkpoint_dir = args.checkpoint_dir
    
    # Validate configuration
    config.validate_config()
    
    return config

def print_experiment_info(config: ContinuumFLConfig, args):
    """Print experiment information"""
    print("="*80)
    print("ContinuumFL: Spatial-Aware Federated Learning Framework")
    print("="*80)
    print(f"Dataset: {config.dataset_name}")
    print(f"Devices: {config.num_devices}")
    print(f"Zones: {config.num_zones}")
    print(f"Rounds: {config.num_rounds}")
    print(f"Local Epochs: {config.local_epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Spatial Regularization: {config.spatial_regularization}")
    print(f"Compression Rate: {config.compression_rate}")
    print(f"Compute Device: {config.device.upper()}")
    
    # Show GPU details if using CUDA
    if config.device == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Details: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"Random Seed: {config.random_seed}")
    print(f"Run Baselines: {args.run_baselines}")
    print("="*80)

def run_continuum_fl_experiment(config: ContinuumFLConfig) -> Tuple[Dict[str, Any], Any]:
    """Run the main ContinuumFL experiment"""
    
    print("\n🚀 Starting ContinuumFL Experiment...")
    
    # Initialize coordinator
    print("📋 Initializing ContinuumFL Coordinator...")
    coordinator = ContinuumFLCoordinator(config)
    
    # Initialize system
    print("🔧 Initializing system components...")
    coordinator.initialize_system()
    
    # Run federated learning
    print("🎯 Starting federated learning...")
    training_results = coordinator.run_federated_learning()
    print("✅ ContinuumFL experiment completed!")
    return training_results, coordinator

def run_baseline_experiments(coordinator: ContinuumFLCoordinator) -> Dict[str, Any]:
    """Run baseline comparison experiments"""
    
    print("\n📊 Running baseline comparisons...")
    baseline_results = coordinator.run_baseline_comparison()
    print("✅ Baseline experiments completed!")
    return baseline_results

def create_visualizations(coordinator: ContinuumFLCoordinator, 
                        baseline_results: Dict[str, Any], config: ContinuumFLConfig):
    """Create comprehensive visualizations"""
    
    print("\n📈 Creating visualizations...")
    
    # Initialize visualizer
    visualizer = ContinuumFLVisualizer(config, save_dir=config.results_dir)
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(coordinator, baseline_results)
    
    print("✅ Visualizations created!")

def save_experiment_results(training_results: Dict[str, Any], 
                          baseline_results: Dict[str, Any],
                          config: ContinuumFLConfig):
    """Save experiment results"""
    
    print("\n💾 Saving experiment results...")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Prepare comprehensive results
    experiment_results = {
        "experiment_config": config.to_dict(),
        "continuum_fl_results": training_results,
        "baseline_results": baseline_results,
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework_version": "1.0.0"
    }
    
    # Save results to JSON
    results_file = os.path.join(config.results_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    # Save configuration
    config_file = os.path.join(config.results_dir, "experiment_config.json")
    config.save_config(config_file)
    
    print(f"📁 Results saved to: {config.results_dir}")
    print(f"📄 Configuration saved to: {config_file}")

def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Check GPU availability first
    recommended_device, gpu_available = check_gpu_availability()
    
    # Auto-adjust device setting based on availability and user preference
    if args.device == 'cuda':
        if gpu_available:
            print(f"🎯 Using GPU as requested")
            config_device = 'cuda'
        else:
            print(f"⚠️  CUDA requested but not available, switching to CPU")
            config_device = 'cpu'
    else:
        print(f"💻 Using CPU as requested")
        config_device = 'cpu'
    
    # Setup configuration
    config = setup_configuration(args)
    config.device = config_device  # Override with checked device

    # Adjust batch size and other parameters based on device
    if config.device == 'cpu':
        print("🔧 Optimizing settings for CPU:")
        if config.batch_size > 32:
            config.batch_size = 32
            print(f"   - Reduced batch size to {config.batch_size}")
        if config.num_devices > 100:
            print(f"   - Large device count ({config.num_devices}) may be slow on CPU")
            print(f"   - Consider reducing to 50-100 devices for faster training")
    else:
        print("🚀 Using GPU optimized settings")
    
    # Print experiment information
    print_experiment_info(config, args)
    
    try:
        # Run main ContinuumFL experiment
        training_results, coordinator = run_continuum_fl_experiment(config)
        print(f"Training Results: {training_results}")
        # Initialize baseline results
        baseline_results = {}
        
        # Run baseline comparisons if requested
        if args.run_baselines:
            baseline_results = run_baseline_experiments(coordinator)
        # Create visualizations if requested
        if args.create_visualizations:
            create_visualizations(coordinator, baseline_results, config)
        # Save results if requested
        if args.save_results:
            save_experiment_results(training_results, baseline_results, config)
        
        # Print final summary
        print("\n" + "="*80)
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Final ContinuumFL Accuracy: {training_results.get('final_accuracy', 0.0):.4f}")
        print(f"⏱️  Total Training Time: {training_results.get('total_training_time', 0.0):.2f}s")
        print(f"📡 Total Communication Cost: {training_results.get('total_communication_cost', 0.0):.2f}MB")
        print(f"💻 Compute Device Used: {config.device.upper()}")
        
        if baseline_results:
            print("\n📈 Baseline Comparison:")
            for method, results in baseline_results.items():
                acc = results.get('final_accuracy', 0.0)
                time_taken = results.get('total_time', 0.0)
                print(f"   {method}: Accuracy={acc:.4f}, Time={time_taken:.2f}s")
        
        print(f"\n📁 Results available in: {config.results_dir}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()