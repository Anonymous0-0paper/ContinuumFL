#!/usr/bin/env python3
"""
Debug test script for ContinuumFL.
Tests basic functionality step by step to identify issues.
"""

import os
import sys
import traceback

def test_step(step_name, test_func):
    """Test a step and report results"""
    print(f"🧪 Testing: {step_name}")
    try:
        result = test_func()
        if result:
            print(f"✅ {step_name} - PASSED")
        else:
            print(f"❌ {step_name} - FAILED")
        return result
    except Exception as e:
        print(f"❌ {step_name} - ERROR: {str(e)}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

def test_imports():
    """Test basic imports"""
    try:
        import torch
        import numpy as np
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from config import ContinuumFLConfig
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   NumPy version: {np.__version__}")
        return True
    except Exception as e:
        print(f"   Import error: {e}")
        return False

def test_config():
    """Test configuration creation"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from config import ContinuumFLConfig
        config = ContinuumFLConfig()
        config.num_devices = 5
        config.num_zones = 2
        config.num_rounds = 2
        config.device = 'cpu'
        config.validate_config()
        print(f"   Config created: {config.num_devices} devices, {config.num_zones} zones")
        return True
    except Exception as e:
        print(f"   Config error: {e}")
        return False

def test_device_creation():
    """Test device creation"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from src.core.device import EdgeDevice, DeviceResources
        resources = DeviceResources(5.0, 4.0, 100.0)
        device = EdgeDevice("test_device", (10.0, 20.0), resources)
        print(f"   Device created: {device.device_id} at {device.location}")
        return True
    except Exception as e:
        print(f"   Device creation error: {e}")
        return False

def test_model_factory():
    """Test model creation"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from config import ContinuumFLConfig
        from src.models.model_factory import ModelFactory
        config = ContinuumFLConfig()
        config.dataset_name = 'cifar100'
        model = ModelFactory.create_model(config)
        print(f"   Model created: {model.__class__.__name__}")
        return True
    except Exception as e:
        print(f"   Model creation error: {e}")
        return False

def test_dataset_init():
    """Test dataset initialization"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from config import ContinuumFLConfig
        from src.data.federated_dataset import FederatedDataset
        config = ContinuumFLConfig()
        config.dataset_name = 'cifar100'
        dataset = FederatedDataset(config)
        print(f"   Dataset initialized: {dataset.dataset_name}")
        return True
    except Exception as e:
        print(f"   Dataset init error: {e}")
        return False

def main():
    """Main debug function"""
    print("🔍 ContinuumFL Debug Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Configuration", test_config),
        ("Device Creation", test_device_creation),
        ("Model Factory", test_model_factory),
        ("Dataset Initialization", test_dataset_init),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_step(test_name, test_func):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Framework is working.")
        print("💡 Next step: Run full test with 'python test_continuumfl.py'")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()