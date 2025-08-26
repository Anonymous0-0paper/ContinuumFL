#!/usr/bin/env python3
"""
Step-by-step ContinuumFL test to identify issues.
"""

import os
import sys
import torch
import numpy as np
import traceback
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def step_test(step_name, test_function):
    """Run a test step and report results"""
    print(f"\n[STEP] {step_name}")
    print("-" * 40)
    
    start_time = time.time()
    try:
        result = test_function()
        elapsed = time.time() - start_time
        print(f"SUCCESS: {step_name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"FAILED: {step_name} failed after {elapsed:.2f}s")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None

def test_imports():
    """Test all necessary imports"""
    from config import ContinuumFLConfig
    from src.core.device import EdgeDevice, DeviceResources  
    from src.core.zone import Zone
    from src.models.model_factory import ModelFactory
    from src.data.federated_dataset import FederatedDataset
    from src.continuum_fl_coordinator import ContinuumFLCoordinator
    return True

def test_configuration():
    """Test configuration creation"""
    from config import ContinuumFLConfig
    config = ContinuumFLConfig()
    
    # Use minimal settings for testing
    config.num_devices = 5
    config.num_zones = 2
    config.num_rounds = 3
    config.local_epochs = 1
    config.batch_size = 4
    config.dataset_name = 'cifar100'
    
    # Set device
    if torch.cuda.is_available():
        config.device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        config.device = 'cpu'
        print("Using CPU")
    
    return config

def test_device_creation():
    """Test edge device creation"""
    from src.core.device import EdgeDevice, DeviceResources
    resources = DeviceResources(5.0, 4.0, 100.0)
    device = EdgeDevice("test_device", (10.0, 20.0), resources)
    print(f"Device created: {device.device_id} at {device.location}")
    return device

def test_zone_creation(device):
    """Test zone creation"""
    from src.core.zone import Zone
    zone = Zone("test_zone", "test_server")
    zone.add_device(device)
    print(f"Zone created with {len(zone.devices)} device(s)")
    return zone

def test_model_creation(config):
    """Test model creation"""
    from src.models.model_factory import ModelFactory
    model = ModelFactory.create_model(config)
    
    # Move to appropriate device
    if config.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__} with {model_params:,} parameters")
    return model

def test_dataset_creation(config):
    """Test dataset creation"""
    from src.data.federated_dataset import FederatedDataset
    dataset = FederatedDataset(config)
    print("Dataset object created successfully")
    return dataset

def test_coordinator_initialization(config):
    """Test coordinator initialization"""
    from src.continuum_fl_coordinator import ContinuumFLCoordinator
    coordinator = ContinuumFLCoordinator(config)
    print("Coordinator initialized successfully")
    return coordinator

def test_system_initialization(coordinator):
    """Test system initialization (the critical step)"""
    print("Starting system initialization...")
    print("WARNING: This will download CIFAR-100 dataset if not present")
    
    # This is where most issues occur
    coordinator.initialize_system()
    
    print(f"System initialized:")
    print(f"  Devices: {len(coordinator.devices)}")
    print(f"  Zones: {len(coordinator.zones)}")
    print(f"  Global model: {coordinator.global_model is not None}")
    
    return True

def test_single_round(coordinator):
    """Test a single round of federated learning"""
    print("Testing single round of federated learning...")
    
    # Run just one round
    original_rounds = coordinator.config.num_rounds
    coordinator.config.num_rounds = 1
    
    results = coordinator.run_federated_learning()
    
    # Restore original setting
    coordinator.config.num_rounds = original_rounds
    
    print(f"Single round completed:")
    print(f"  Accuracy: {results.get('final_accuracy', 0.0):.4f}")
    print(f"  Time: {results.get('total_training_time', 0.0):.2f}s")
    
    return results

def main():
    """Main test function"""
    print("ContinuumFL Step-by-Step Test")
    print("=" * 50)
    
    # Step 1: Test imports
    if not step_test("Import Testing", test_imports):
        return False
    
    # Step 2: Test configuration
    config = step_test("Configuration Creation", test_configuration)
    if config is None:
        return False
    
    # Step 3: Test device creation
    device = step_test("Device Creation", test_device_creation)
    if device is None:
        return False
    
    # Step 4: Test zone creation
    zone = step_test("Zone Creation", lambda: test_zone_creation(device))
    if zone is None:
        return False
    
    # Step 5: Test model creation
    model = step_test("Model Creation", lambda: test_model_creation(config))
    if model is None:
        return False
    
    # Step 6: Test dataset creation
    dataset = step_test("Dataset Creation", lambda: test_dataset_creation(config))
    if dataset is None:
        return False
    
    # Step 7: Test coordinator initialization
    coordinator = step_test("Coordinator Initialization", lambda: test_coordinator_initialization(config))
    if coordinator is None:
        return False
    
    print("\n" + "=" * 50)
    print("BASIC TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    # Ask user if they want to continue with system initialization
    print("\nNext steps will:")
    print("1. Download CIFAR-100 dataset (if not present)")
    print("2. Initialize the complete system")
    print("3. Run a test round of federated learning")
    
    print("\nContinuing with system initialization...")
    
    # Step 8: Test system initialization (this is the big one)
    if not step_test("System Initialization", lambda: test_system_initialization(coordinator)):
        print("\nSystem initialization failed. This is usually due to:")
        print("- Dataset download issues")
        print("- Memory issues")
        print("- Model-device compatibility issues")
        return False
    
    # Step 9: Test single round
    results = step_test("Single Round Test", lambda: test_single_round(coordinator))
    if results is None:
        return False
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("ContinuumFL framework is fully functional!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nTest completed successfully!")
        print("You can now run full experiments with:")
        print("  python main.py --dataset cifar100 --num_devices 20 --num_rounds 50")
    else:
        print("\nTest failed - see error details above")