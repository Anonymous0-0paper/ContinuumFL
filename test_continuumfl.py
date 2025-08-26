#!/usr/bin/env python3
"""
Simple test script for ContinuumFL framework.
Tests basic functionality with minimal resources.
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import ContinuumFLConfig
from src.continuum_fl_coordinator import ContinuumFLCoordinator

def check_system_requirements():
    """Check system requirements and GPU availability"""
    print("🔍 Checking System Requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version OK")
    
    # Check PyTorch
    try:
        print(f"PyTorch Version: {torch.__version__}")
    except:
        print("❌ PyTorch not found")
        return False
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA Available: {device_count} device(s)")
        print(f"🎯 Primary GPU: {device_name}")
        
        # Test GPU memory
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"💾 GPU Memory: {memory_gb:.1f}GB")
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU test successful")
        except Exception as e:
            print(f"⚠️  GPU test failed: {e}")
    else:
        print("❌ CUDA not available - will use CPU")
    
    # Check NumPy
    try:
        print(f"NumPy Version: {np.__version__}")
    except:
        print("❌ NumPy not found")
        return False
    
    return True

def test_basic_functionality():
    """Test basic ContinuumFL functionality"""
    print("🧪 Testing ContinuumFL Basic Functionality...")
    
    # Determine device to use
    if torch.cuda.is_available():
        test_device = 'cuda'
        batch_size = 16  # Smaller batch for safety
        print(f"🚀 Using GPU for testing")
    else:
        test_device = 'cpu'
        batch_size = 8   # Even smaller for CPU
        print(f"💻 Using CPU for testing")
    
    # Create minimal configuration for testing
    config = ContinuumFLConfig()
    config.num_devices = 10          # Small number for testing
    config.num_zones = 3             # Few zones
    config.num_rounds = 5            # Few rounds
    config.local_epochs = 1          # Single epoch
    config.batch_size = batch_size   # Appropriate batch size
    config.dataset_name = 'cifar100'
    config.device = test_device      # Use detected device
    config.log_level = 'INFO'
    
    try:
        # Test configuration validation
        print("✅ Testing configuration validation...")
        config.validate_config()
        print("✅ Configuration validation passed")
        
        # Test coordinator initialization
        print("✅ Testing coordinator initialization...")
        coordinator = ContinuumFLCoordinator(config)
        print("✅ Coordinator initialization passed")
        
        # Test system initialization (this will download data)
        print("✅ Testing system initialization...")
        coordinator.initialize_system()
        print("✅ System initialization passed")
        
        # Test single round of federated learning
        print("✅ Testing federated learning round...")
        training_results = coordinator.run_federated_learning()
        print("✅ Federated learning test passed")
        
        # Print results
        final_accuracy = training_results.get('final_accuracy', 0.0)
        total_time = training_results.get('total_training_time', 0.0)
        
        print(f"📊 Test Results:")
        print(f"   Final Accuracy: {final_accuracy:.4f}")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Devices: {len(coordinator.devices)}")
        print(f"   Zones: {len(coordinator.zones)}")
        
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components separately"""
    print("\n🔧 Testing Individual Components...")
    
    try:
        # Test configuration
        print("Testing configuration...")
        config = ContinuumFLConfig()
        assert config.num_devices > 0
        assert config.num_zones > 0
        print("✅ Configuration test passed")
        
        # Test device creation
        print("Testing device creation...")
        from src.core.device import EdgeDevice, DeviceResources
        resources = DeviceResources(5.0, 4.0, 100.0)
        device = EdgeDevice("test_device", (10.0, 20.0), resources)
        assert device.device_id == "test_device"
        assert device.location == (10.0, 20.0)
        print("✅ Device creation test passed")
        
        # Test zone creation  
        print("Testing zone creation...")
        from src.core.zone import Zone
        zone = Zone("test_zone", "test_server")
        zone.add_device(device)
        assert len(zone.devices) == 1
        print("✅ Zone creation test passed")
        
        # Test model factory
        print("Testing model factory...")
        from src.models.model_factory import ModelFactory
        model = ModelFactory.create_model(config)
        assert model is not None
        print("✅ Model factory test passed")
        
        print("🎉 All component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("="*60)
    print("ContinuumFL Framework Test Suite")
    print("="*60)
    
    # Check system requirements first
    if not check_system_requirements():
        print("\n❌ System requirements check failed")
        sys.exit(1)
    
    print("\n✅ System requirements OK")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test individual components first
    component_test_passed = test_individual_components()
    
    if component_test_passed:
        # Test full functionality
        full_test_passed = test_basic_functionality()
        
        if full_test_passed:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED! ContinuumFL is working correctly.")
            print("="*60)
            print("You can now run the full experiment with:")
            print("python main.py --dataset cifar100 --num_devices 50 --num_rounds 100")
            print("="*60)
        else:
            print("\n❌ Full functionality test failed")
            sys.exit(1)
    else:
        print("\n❌ Component tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()