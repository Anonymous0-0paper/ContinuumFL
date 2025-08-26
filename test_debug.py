#!/usr/bin/env python3
"""
Debug test script for ContinuumFL framework.
This script tests components step by step and reports errors clearly.
"""

import os
import sys
import traceback
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all imports"""
    print("🔍 Testing Imports...")
    
    try:
        from config import ContinuumFLConfig
        print("✅ Config import OK")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.core.device import EdgeDevice, DeviceResources
        print("✅ Device import OK")
    except Exception as e:
        print(f"❌ Device import failed: {e}")
        return False
    
    try:
        from src.core.zone import Zone
        print("✅ Zone import OK")  
    except Exception as e:
        print(f"❌ Zone import failed: {e}")
        return False
    
    try:
        from src.models.model_factory import ModelFactory
        print("✅ Model factory import OK")
    except Exception as e:
        print(f"❌ Model factory import failed: {e}")
        return False
    
    try:
        from src.data.federated_dataset import FederatedDataset
        print("✅ Federated dataset import OK")
    except Exception as e:
        print(f"❌ Federated dataset import failed: {e}")
        return False
    
    try:
        # Try importing coordinator
        from src.continuum_fl_coordinator import ContinuumFLCoordinator
        print("✅ Coordinator import OK")
    except Exception as e:
        print(f"❌ Coordinator import failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality step by step"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        # Test configuration
        print("1️⃣ Testing Configuration...")
        from config import ContinuumFLConfig
        config = ContinuumFLConfig()
        config.num_devices = 5
        config.num_zones = 2
        config.num_rounds = 2
        config.local_epochs = 1
        config.batch_size = 8
        config.dataset_name = 'cifar100'
        print("✅ Configuration OK")
        
        # Test device creation
        print("2️⃣ Testing Device Creation...")
        from src.core.device import EdgeDevice, DeviceResources
        resources = DeviceResources(5.0, 4.0, 100.0)
        device = EdgeDevice("test_device", (10.0, 20.0), resources)
        print("✅ Device creation OK")
        
        # Test zone creation
        print("3️⃣ Testing Zone Creation...")
        from src.core.zone import Zone
        zone = Zone("test_zone", "test_server")
        zone.add_device(device)
        print("✅ Zone creation OK")
        
        # Test model creation
        print("4️⃣ Testing Model Creation...")
        from src.models.model_factory import ModelFactory
        model = ModelFactory.create_model(config)
        print("✅ Model creation OK")
        
        # Test dataset creation (without download)
        print("5️⃣ Testing Dataset Creation...")
        from src.data.federated_dataset import FederatedDataset
        dataset = FederatedDataset(config)
        print("✅ Dataset creation OK")
        
        print("🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_coordinator_initialization():
    """Test coordinator initialization separately"""
    print("\n🚀 Testing Coordinator Initialization...")
    
    try:
        from config import ContinuumFLConfig
        from src.continuum_fl_coordinator import ContinuumFLCoordinator
        
        config = ContinuumFLConfig()
        config.num_devices = 5
        config.num_zones = 2
        config.num_rounds = 2
        config.local_epochs = 1
        config.batch_size = 8
        config.dataset_name = 'cifar100'
        
        coordinator = ContinuumFLCoordinator(config)
        print("✅ Coordinator initialization OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinator initialization failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def main():
    """Main debug function"""
    print("="*60)
    print("ContinuumFL Debug Test")
    print("="*60)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed - stopping here")
        return
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed - stopping here")
        return
    
    # Test coordinator
    if not test_coordinator_initialization():
        print("\n❌ Coordinator tests failed")
        return
    
    print("\n" + "="*60)
    print("🎉 ALL DEBUG TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    main()