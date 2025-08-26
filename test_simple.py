#!/usr/bin/env python3
"""
Simple ContinuumFL test without Unicode characters.
"""

import os
import sys
import torch
import numpy as np
import traceback
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main test function"""
    
    print("ContinuumFL Framework Test")
    print("=" * 50)
    
    try:
        # Test imports
        print("\nTesting Imports...")
        from config import ContinuumFLConfig
        print("SUCCESS: Config imported")
        
        from src.core.device import EdgeDevice, DeviceResources  
        print("SUCCESS: Device imported")
        
        from src.core.zone import Zone
        print("SUCCESS: Zone imported")
        
        from src.models.model_factory import ModelFactory
        print("SUCCESS: Model factory imported")
        
        from src.data.federated_dataset import FederatedDataset
        print("SUCCESS: Dataset imported")
        
        from src.continuum_fl_coordinator import ContinuumFLCoordinator
        print("SUCCESS: Coordinator imported")
        
        # Test configuration
        print("\nTesting Configuration...")
        config = ContinuumFLConfig()
        config.num_devices = 10
        config.num_zones = 3
        config.num_rounds = 5
        config.local_epochs = 1
        config.batch_size = 8
        config.dataset_name = 'cifar100'
        print("SUCCESS: Configuration created")
        
        # Test device creation
        print("\nTesting Device Creation...")
        resources = DeviceResources(5.0, 4.0, 100.0)
        device = EdgeDevice("test_device", (10.0, 20.0), resources)
        print(f"SUCCESS: Device created: {device.device_id}")
        
        # Test zone creation
        print("\nTesting Zone Creation...")
        zone = Zone("test_zone", "test_server")
        zone.add_device(device)
        print(f"SUCCESS: Zone created with {len(zone.devices)} device(s)")
        
        # Test model creation
        print("\nTesting Model Creation...")
        model = ModelFactory.create_model(config)
        model_params = sum(p.numel() for p in model.parameters())
        print(f"SUCCESS: Model created: {type(model).__name__} with {model_params} parameters")
        
        # Test dataset (without downloading)
        print("\nTesting Dataset Creation...")
        dataset = FederatedDataset(config)
        print("SUCCESS: Dataset object created")
        
        # Test coordinator initialization  
        print("\nTesting Coordinator Initialization...")
        coordinator = ContinuumFLCoordinator(config)
        print("SUCCESS: Coordinator initialized")
        
        # GPU/Device check
        print("\nDevice Information...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"SUCCESS: GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
            config.device = 'cuda'
        else:
            print("INFO: Using CPU (no CUDA available)")
            config.device = 'cpu'
        
        print("\n" + "=" * 50)
        print("ALL BASIC TESTS PASSED!")
        print("ContinuumFL framework is working correctly")
        print("You can now run full tests with:")
        print("  python test_continuumfl.py")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        print(f"Error details:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed - see error details above")