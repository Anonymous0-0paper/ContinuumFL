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
    parser.add_argument('--dataset', type=str, default='shakespeare',
                       choices=['cifar100', 'femnist', 'shakespeare', 'ucihar', 'speechcommands'],
                       help='Dataset to use for training')
    parser.add_argument('--max_samples', type=int, default=50000,
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
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate model every N rounds (default: 1, set higher to speed up baselines)')

    # Training parameters
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    
    # ContinuumFL specific parameters
    parser.add_argument('--spatial_weight', type=float, default=0.4,
                       help='Weight for spatial similarity')
    parser.add_argument('--data_weight', type=float, default=0.4,
                       help='Weight for data similarity')
    parser.add_argument('--network_weight', type=float, default=0.2,
                       help='Weight for network similarity')
    parser.add_argument('--spatial_regularization', type=float, default=0.05,
                       help='Spatial regularization parameter')
    parser.add_argument('--correlation_threshold', type=float, default=0.05,
                        help='Correlation threshold for neighborship')
    parser.add_argument('--compression_rate', type=float, default=0.6,
                        help='Gradient compression rate')
    parser.add_argument('--enable_compression', action='store_true', default=True,
                        help='Enable compression during parameter transmission')
    parser.add_argument('--enable_lr_scheduler', action='store_true', default=False,
                        help='Enable per-dataset LR scheduler (StepLR/Cosine/ReduceLROnPlateau)')
    parser.add_argument('--enable_early_stopping', action='store_true', default=True,
                        help='Stop training when accuracy stops improving')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Rounds without improvement before early stop (default: 10)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                        help='Minimum accuracy gain to count as improvement (default: 1e-4)')
    parser.add_argument('--intra_zone_alpha', type=float, default=100,
                        help='Dirichlet alpha value within zones (1000=IID, 0.1=highly non-IID)')
    parser.add_argument('--inter_zone_alpha', type=float, default=5,
                        help='Dirichlet alpha value across zones (1000=IID, 0.1=highly non-IID)')

    # Aggregation Settings
    parser.add_argument('--async_aggregation', action='store_true', default=False,
                        help='Enable asynchronous inter zone aggregation')

    # Device Settings
    parser.add_argument('--enable_failure', action='store_true', default=False,
                        help='Enable simulation of failure for devices')
    parser.add_argument('--device_failure_probability', type=float, default=0.05,
                        help='Probability of device failure')
    parser.add_argument('--zone_failure_probability', type=float, default=0.02,
                        help='Probability of zone failure')

    parser.add_argument('--shakespeare_num_speakers', type=int, default=35,
                        help='Set number of heterogeneous text sources (only available for shakespear dataset)')
    # Experiment options
    parser.add_argument('--run_baselines', action='store_true',
                       help='Run baseline comparison')
    parser.add_argument('--baselines_only', action='store_true',
                        help='Run baselines only')
    parser.add_argument('--baseline_methods', nargs='+',
                        default=["FedAvg", "FedProx", "HierFL", "ClusterFL", "APCFL", "IFCA", "SnapCFL", "GeoFL"],
                        help='Baseline Methods to compare against')

    # GeoFL hyperparameters (namespaced as geofl.*)
    parser.add_argument('--geofl_num_rounds', type=int, default=None,
                        help='GeoFL: number of rounds (default: same as --num_rounds)')
    parser.add_argument('--geofl_local_steps', type=int, default=5,
                        help='GeoFL: LocalSGD steps per client (paper default: 5)')
    parser.add_argument('--geofl_batch_size', type=int, default=16,
                        help='GeoFL: batch size (paper default: 16)')
    parser.add_argument('--geofl_lr', type=float, default=0.001,
                        help='GeoFL: learning rate')
    parser.add_argument('--geofl_S0', type=float, default=0.01,
                        help='GeoFL: initial importance threshold S0')
    parser.add_argument('--geofl_S_min', type=float, default=0.001,
                        help='GeoFL: minimum importance threshold S_min')
    parser.add_argument('--geofl_alpha', type=float, default=0.95,
                        help='GeoFL: decay factor alpha')
    parser.add_argument('--geofl_R0', type=int, default=5,
                        help='GeoFL: upload up-bound R0')
    parser.add_argument('--geofl_beta', type=float, default=0.2,
                        help='GeoFL: staleness exponent beta')
    parser.add_argument('--geofl_client_sampling_rate', type=float, default=0.7,
                        help='GeoFL: client participation rate')

    # IFCA hyperparameters (namespaced as ifca.*)
    parser.add_argument('--ifca_num_rounds', type=int, default=None,
                        help='IFCA: number of rounds (default: same as --num_rounds)')
    parser.add_argument('--ifca_k', type=int, default=4,
                        help='IFCA: number of clusters k')
    parser.add_argument('--ifca_local_steps', type=int, default=5,
                        help='IFCA: local SGD steps τ')
    parser.add_argument('--ifca_lr', type=float, default=0.01,
                        help='IFCA: step size γ')
    parser.add_argument('--ifca_lr_decay', type=float, default=1.0,
                        help='IFCA: per-round LR multiplier (0.99 for CIFAR)')
    parser.add_argument('--ifca_batch_size', type=int, default=32,
                        help='IFCA: batch size')
    parser.add_argument('--ifca_sampling_rate', type=float, default=0.7,
                        help='IFCA: client participation rate')
    parser.add_argument('--ifca_variant', type=str, default='model',
                        choices=['model', 'gradient'],
                        help='IFCA variant: model (Option II) or gradient (Option I)')
    parser.add_argument('--ifca_weight_sharing', action='store_true', default=False,
                        help='IFCA: enable weight-sharing extension')

    # SnapCFL hyperparameters (namespaced as snapcfl.*)
    parser.add_argument('--snapcfl_num_rounds', type=int, default=None,
                        help='SnapCFL: number of rounds (default: same as --num_rounds)')
    parser.add_argument('--snapcfl_lr', type=float, default=0.01,
                        help='SnapCFL: learning rate')
    parser.add_argument('--snapcfl_batch_size', type=int, default=32,
                        help='SnapCFL: batch size')
    parser.add_argument('--snapcfl_local_epochs', type=int, default=5,
                        help='SnapCFL: local epochs per round')
    parser.add_argument('--snapcfl_sampling_rate', type=float, default=0.7,
                        help='SnapCFL: client sampling rate')
    parser.add_argument('--snapcfl_pre_cluster_rounds', type=int, default=10,
                        help='SnapCFL: FedAvg rounds per pairwise binary classifier')
    parser.add_argument('--snapcfl_eps', type=float, default=0.25,
                        help='SnapCFL: DBSCAN epsilon')
    parser.add_argument('--snapcfl_min_samples', type=int, default=2,
                        help='SnapCFL: DBSCAN min_samples')
    parser.add_argument('--snapcfl_intra_algo', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox'],
                        help='SnapCFL: intra-cluster algorithm')
    parser.add_argument('--snapcfl_global_averaging', action='store_true', default=False,
                        help='SnapCFL: average across clusters each round')
    parser.add_argument('--snapcfl_mu_fedprox', type=float, default=0.01,
                        help='SnapCFL: FedProx proximal term μ')
    parser.add_argument('--snapcfl_H', type=int, default=10,
                        help='SnapCFL: frequency constraint window H')
    parser.add_argument('--snapcfl_c_thre', type=int, default=3,
                        help='SnapCFL: max selections per client per window H')

    # AP-CFL hyperparameters (namespaced as apcfl.*)
    parser.add_argument('--apcfl_num_rounds', type=int, default=None,
                        help='AP-CFL: number of rounds (default: same as --num_rounds)')
    parser.add_argument('--apcfl_local_epochs', type=int, default=1,
                        help='AP-CFL: local epochs E (paper default: 1)')
    parser.add_argument('--apcfl_batch_size', type=int, default=32,
                        help='AP-CFL: batch size B (paper default: 32)')
    parser.add_argument('--apcfl_lr', type=float, default=0.0001,
                        help='AP-CFL: learning rate η (paper default: 0.0001)')
    parser.add_argument('--apcfl_lambda', type=float, default=0.05,
                        help='AP-CFL: encoder regularisation λ (paper default: 0.05)')
    parser.add_argument('--apcfl_sampling_rate', type=float, default=0.7,
                        help='AP-CFL: client sampling rate α')

    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save experiment results')
    parser.add_argument('--create_visualizations', action='store_true', default=False,
                       help='Create visualization plots')
    
    # System options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training (e.g. cuda, cuda:0, cpu)')
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

def check_amd_gpu_availability():
    """Check if AMD GPU (ROCm) is available"""
    try:
        # Check if HIP is available (AMD GPU support)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"✅ AMD GPU (ROCm) Available: Yes")
            print(f"   HIP version: {torch.version.hip}")
            
            # Try to get AMD device info
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
                memory_free = (torch.cuda.get_device_properties(current_device).total_memory - torch.cuda.memory_allocated(current_device)) / (1024**3)
                
                print(f"📟 AMD Device Count: {device_count}")
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
                
                return True
    except Exception as e:
        pass
    
    return False

def check_cuda_availability():
    """Check if NVIDIA CUDA GPU is available"""
    try:
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            # Make sure this is CUDA, not ROCm
            if not (hasattr(torch.version, 'hip') and torch.version.hip is not None):
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)  # GB
                memory_free = (torch.cuda.get_device_properties(current_device).total_memory - torch.cuda.memory_allocated(current_device)) / (1024**3)
                
                print(f"✅ NVIDIA CUDA Available: Yes")
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
                
                return True
    except Exception as e:
        pass
    
    return False

def check_gpu_availability():
    """Check GPU availability (AMD first, then CUDA, then CPU)
    
    Returns:
        tuple: (device_str, is_gpu_available, gpu_type)
               device_str: 'cuda' or 'cpu'
               is_gpu_available: True if GPU found
               gpu_type: 'amd', 'cuda', or 'cpu'
    """
    print("🔍 Checking GPU availability...")
    
    # Check AMD GPU (ROCm) first
    if check_amd_gpu_availability():
        return 'cuda', True, 'amd'  # AMD ROCm uses 'cuda' backend in PyTorch
    
    # Check NVIDIA CUDA
    if check_cuda_availability():
        return 'cuda', True, 'cuda'
    
    # Fall back to CPU
    print(f"❌ No GPU Available (AMD or CUDA)")
    print(f"💻 Will use CPU for training")
    print(f"💡 For faster training, consider:")
    print(f"   - Installing ROCm-enabled PyTorch (for AMD GPUs)")
    print(f"   - Installing CUDA-enabled PyTorch (for NVIDIA GPUs)")
    print(f"   - Using Google Colab or cloud GPU instances")
    print(f"   - Running smaller experiments on CPU")
    
    return 'cpu', False, 'cpu'

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
    config.enable_lr_scheduler = args.enable_lr_scheduler
    config.enable_early_stopping = args.enable_early_stopping
    config.early_stopping_patience = args.early_stopping_patience
    config.early_stopping_min_delta = args.early_stopping_min_delta
    config.intra_zone_alpha = args.intra_zone_alpha
    config.inter_zone_alpha = args.inter_zone_alpha

    # Aggregation Settings
    config.async_aggregation = args.async_aggregation

    # Device Settings
    config.enable_failure = args.enable_failure
    config.zone_failure_probability = args.zone_failure_probability
    config.device_failure_probability = args.device_failure_probability

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

    config.baselines_only = args.baselines_only
    config.baselines = args.baseline_methods

    # GeoFL hyperparameters
    config.geofl_num_rounds = args.geofl_num_rounds if args.geofl_num_rounds is not None else config.num_rounds
    config.geofl_local_steps = args.geofl_local_steps
    config.geofl_batch_size = args.geofl_batch_size
    config.geofl_lr = args.geofl_lr
    config.geofl_S0 = args.geofl_S0
    config.geofl_S_min = args.geofl_S_min
    config.geofl_alpha = args.geofl_alpha
    config.geofl_R0 = args.geofl_R0
    config.geofl_beta = args.geofl_beta
    config.geofl_client_sampling_rate = args.geofl_client_sampling_rate

    # AP-CFL hyperparameters
    config.apcfl_num_rounds = args.apcfl_num_rounds if args.apcfl_num_rounds is not None else config.num_rounds
    config.apcfl_local_epochs = args.apcfl_local_epochs
    config.apcfl_batch_size = args.apcfl_batch_size
    config.apcfl_lr = args.apcfl_lr
    config.apcfl_lambda = args.apcfl_lambda
    config.apcfl_sampling_rate = args.apcfl_sampling_rate

    # IFCA hyperparameters
    config.ifca_num_rounds     = args.ifca_num_rounds if args.ifca_num_rounds is not None else config.num_rounds
    config.ifca_k              = args.ifca_k
    config.ifca_local_steps    = args.ifca_local_steps
    config.ifca_lr             = args.ifca_lr
    config.ifca_lr_decay       = args.ifca_lr_decay
    config.ifca_batch_size     = args.ifca_batch_size
    config.ifca_sampling_rate  = args.ifca_sampling_rate
    config.ifca_variant        = args.ifca_variant
    config.ifca_weight_sharing = args.ifca_weight_sharing

    # SnapCFL hyperparameters
    config.snapcfl_num_rounds         = args.snapcfl_num_rounds if args.snapcfl_num_rounds is not None else config.num_rounds
    config.snapcfl_lr                 = args.snapcfl_lr
    config.snapcfl_batch_size         = args.snapcfl_batch_size
    config.snapcfl_local_epochs       = args.snapcfl_local_epochs
    config.snapcfl_sampling_rate      = args.snapcfl_sampling_rate
    config.snapcfl_pre_cluster_rounds = args.snapcfl_pre_cluster_rounds
    config.snapcfl_eps                = args.snapcfl_eps
    config.snapcfl_min_samples        = args.snapcfl_min_samples
    config.snapcfl_intra_algo         = args.snapcfl_intra_algo
    config.snapcfl_global_averaging   = args.snapcfl_global_averaging
    config.snapcfl_mu_fedprox         = args.snapcfl_mu_fedprox
    config.snapcfl_H                  = args.snapcfl_H
    config.snapcfl_c_thre             = args.snapcfl_c_thre

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
    training_results = {}
    if not config.baselines_only:
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

    # Build descriptive run name: dataset__intra{a}__inter{b}__comp{c}
    compression_pct = int(round(config.compression_rate * 100))
    run_name = (
        f"{config.dataset_name}"
        f"__intra{config.intra_zone_alpha}"
        f"__inter{config.inter_zone_alpha}"
        f"__comp{compression_pct}pct"
    )
    run_dir = os.path.join(config.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Prepare comprehensive results
    experiment_results = {
        "experiment_config": config.to_dict(),
        "continuum_fl_results": training_results,
        "baseline_results": baseline_results,
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework_version": "1.0.0"
    }

    # Save results to JSON
    results_file = os.path.join(run_dir, "experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)

    # Save configuration
    config_file = os.path.join(run_dir, "experiment_config.json")
    config.save_config(config_file)

    print(f"📁 Results saved to: {run_dir}")
    print(f"📄 Configuration saved to: {config_file}")

def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Check GPU availability first
    recommended_device, gpu_available, gpu_type = check_gpu_availability()
    
    # Auto-adjust device setting based on availability and user preference
    if args.device == 'cuda':
        if gpu_available:
            if gpu_type == 'amd':
                print(f"🎯 Using AMD GPU (ROCm) as requested")
            elif gpu_type == 'cuda':
                print(f"🎯 Using NVIDIA CUDA GPU as requested")
            config_device = 'cuda'
        else:
            print(f"⚠️  GPU requested but not available, switching to CPU")
            config_device = 'cpu'
    else:
        print(f"💻 Using CPU as requested")
        config_device = 'cpu'
    
    # Setup configuration
    config = setup_configuration(args)
    config.device = config_device  # Override with checked device

    # Per-dataset auto-tuning of defaults that are known to be poor fits
    if config.dataset_name.lower() == 'shakespeare':
        if args.learning_rate == 0.01:
            config.learning_rate = 0.01
            print("🔧 Shakespeare auto-tuning: reduced learning rate to 0.001")
        if args.batch_size > 16:
            config.batch_size = 16
            print(f"🔧 Shakespeare auto-tuning: reduced batch size to {config.batch_size}")

    if config.dataset_name.lower() == 'ucihar':
        if args.learning_rate == 0.001:
            config.learning_rate = 0.001
            print("🔧 UCI HAR auto-tuning: learning rate 0.001")
        if args.batch_size > 64:
            config.batch_size = 64
            print(f"🔧 UCI HAR auto-tuning: batch size capped at {config.batch_size}")

    if config.dataset_name.lower() == 'speechcommands':
        if args.learning_rate == 0.001:
            config.learning_rate = 0.001
            print("🔧 Speech Commands auto-tuning: learning rate 0.001")
        if args.batch_size > 32:
            config.batch_size = 32
            print(f"🔧 Speech Commands auto-tuning: batch size capped at {config.batch_size}")

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
        if args.run_baselines or args.baselines_only:
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
        
        compression_pct = int(round(config.compression_rate * 100))
        run_name = (
            f"{config.dataset_name}"
            f"__intra{config.intra_zone_alpha}"
            f"__inter{config.inter_zone_alpha}"
            f"__comp{compression_pct}pct"
        )
        print(f"\n📁 Results available in: {os.path.join(config.results_dir, run_name)}")
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