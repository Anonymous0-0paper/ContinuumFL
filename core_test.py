#!/usr/bin/env python3
"""
Super minimal test for ContinuumFL - tests core components without dataset download
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_core_components():
    """Test core components without dataset operations"""
    print("🔧 Testing Core Components")
    print("=" * 40)
    
    try:
        # 1. Test Configuration
        print("1️⃣ Testing Configuration...")
        from config import ContinuumFLConfig
        config = ContinuumFLConfig()
        config.num_devices = 4
        config.num_zones = 2
        config.device = 'cpu'
        print("   ✅ Configuration OK")
        
        # 2. Test Device Creation
        print("2️⃣ Testing Device Creation...")
        from src.core.device import EdgeDevice, DeviceResources
        resources = DeviceResources(5.0, 4.0, 100.0)
        device = EdgeDevice("test_device", (10.0, 20.0), resources)
        print(f"   ✅ Device created: {device.device_id}")
        
        # 3. Test Zone Creation
        print("3️⃣ Testing Zone Creation...")
        from src.core.zone import Zone
        zone = Zone("test_zone", "test_server")
        zone.add_device(device)
        print(f"   ✅ Zone created with {len(zone.devices)} device(s)")
        
        # 4. Test Model Factory
        print("4️⃣ Testing Model Creation...")
        from src.models.model_factory import ModelFactory
        model = ModelFactory.create_model(config)
        print(f"   ✅ Model created: {model.__class__.__name__}")
        
        # 5. Test Zone Discovery
        print("5️⃣ Testing Zone Discovery...")
        from src.core.zone_discovery import ZoneDiscovery
        zone_discovery = ZoneDiscovery(config)
        print("   ✅ Zone Discovery initialized")
        
        # 6. Test Aggregator
        print("6️⃣ Testing Hierarchical Aggregator...")
        from src.aggregation.hierarchical_aggregator import HierarchicalAggregator
        aggregator = HierarchicalAggregator(config)
        aggregator.set_global_model(model)
        print("   ✅ Aggregator initialized")
        
        # 7. Test Communication
        print("7️⃣ Testing Communication Module...")
        from src.communication.compression import GradientCompressor
        compressor = GradientCompressor(config)
        print("   ✅ Communication module OK")
        
        # 8. Test Baseline Methods
        print("8️⃣ Testing Baseline Methods...")
        from src.baselines.baseline_fl import BaselineFLMethods
        baselines = BaselineFLMethods(config)
        print("   ✅ Baseline methods OK")
        
        print("\n🎉 All core components working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {str(e)}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return False

def test_simple_training():
    """Test very basic training without real dataset"""
    print("\n🏃 Testing Simple Training Logic")
    print("=" * 40)
    
    try:
        from config import ContinuumFLConfig
        from src.models.model_factory import ModelFactory
        
        # Create simple config
        config = ContinuumFLConfig()
        config.device = 'cpu'
        
        # Create model
        model = ModelFactory.create_model(config)
        
        # Create fake data
        fake_data = torch.randn(4, 3, 32, 32)  # 4 samples of CIFAR-like data
        fake_labels = torch.randint(0, 100, (4,))  # Random labels
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(fake_data)
            print(f"   ✅ Forward pass: {output.shape}")
        
        # Test simple training step
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = model(fake_data)
        loss = criterion(output, fake_labels)
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ Training step: loss = {loss.item():.4f}")
        print("\n🎉 Basic training logic working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training test failed: {str(e)}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("🧪 ContinuumFL Core Components Test")
    print("=" * 50)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test core components
    components_ok = test_core_components()
    
    if components_ok:
        # Test simple training
        training_ok = test_simple_training()
        
        if training_ok:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED!")
            print("✅ ContinuumFL core functionality is working")
            print("💡 Next steps:")
            print("   1. Run with dataset: python minimal_test.py")
            print("   2. Run full test: python test_continuumfl.py")
            print("   3. Run experiment: python main.py --device cpu --num_devices 10")
            return True
    
    print("\n❌ Some tests failed. Please check errors above.")
    return False

if __name__ == "__main__":
    main()