#!/usr/bin/env python3
"""
Minimal test for ContinuumFL - CPU only, very small scale
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_minimal_test():
    """Run minimal ContinuumFL test"""
    print("🧪 Running Minimal ContinuumFL Test (CPU-only)")
    print("=" * 50)
    
    try:
        # Import necessary modules
        from config import ContinuumFLConfig
        from src.continuum_fl_coordinator import ContinuumFLCoordinator
        
        # Create minimal config
        config = ContinuumFLConfig()
        config.num_devices = 6          # Very small
        config.num_zones = 2            # Just 2 zones
        config.num_rounds = 2           # Very few rounds
        config.local_epochs = 1         # Single epoch
        config.batch_size = 8           # Small batch
        config.dataset_name = 'cifar100'
        config.device = 'cpu'           # Force CPU
        config.log_level = 'WARNING'    # Reduce log verbosity
        
        print(f"✅ Configuration created: {config.num_devices} devices, {config.num_zones} zones")
        
        # Initialize coordinator
        coordinator = ContinuumFLCoordinator(config)
        print("✅ Coordinator initialized")
        
        # Test system initialization (this will download CIFAR-100)
        print("📥 Initializing system (may download dataset)...")
        coordinator.initialize_system()
        print("✅ System initialization completed")
        
        print(f"📊 System status:")
        print(f"   - Devices: {len(coordinator.devices)}")
        print(f"   - Zones: {len(coordinator.zones)}")
        print(f"   - Device: {config.device}")
        
        # Run just 1 round of training
        print("🎯 Running 1 round of federated learning...")
        training_results = coordinator.run_federated_learning()
        
        print("🎉 Minimal test completed successfully!")
        print(f"📈 Final accuracy: {training_results.get('final_accuracy', 0.0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"📋 Full error:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = run_minimal_test()
    if success:
        print("\n🎉 Ready to run larger experiments!")
        print("💡 Try: python main.py --device cpu --num_devices 20 --num_rounds 10")
    else:
        print("\n❌ Please check the errors above and try again.")