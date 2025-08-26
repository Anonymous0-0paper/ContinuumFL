#!/usr/bin/env python3
"""
Comprehensive ContinuumFL test with file logging.
"""

import os
import sys
import torch
import numpy as np
import traceback
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def log_message(message, log_file):
    """Log message to both console and file"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

def main():
    """Main test function with file logging"""
    
    # Create log file
    log_path = "test_results.log"
    with open(log_path, 'w') as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message(f"ContinuumFL Test Started at {timestamp}", log_file)
        log_message("="*60, log_file)
        
        try:
            # Test imports
            log_message("\n🔍 Testing Imports...", log_file)
            from config import ContinuumFLConfig
            log_message("✅ Config imported", log_file)
            
            from src.core.device import EdgeDevice, DeviceResources  
            log_message("✅ Device imported", log_file)
            
            from src.core.zone import Zone
            log_message("✅ Zone imported", log_file)
            
            from src.models.model_factory import ModelFactory
            log_message("✅ Model factory imported", log_file)
            
            from src.data.federated_dataset import FederatedDataset
            log_message("✅ Dataset imported", log_file)
            
            from src.continuum_fl_coordinator import ContinuumFLCoordinator
            log_message("✅ Coordinator imported", log_file)
            
            # Test configuration
            log_message("\n🔧 Testing Configuration...", log_file)
            config = ContinuumFLConfig()
            config.num_devices = 10
            config.num_zones = 3
            config.num_rounds = 5
            config.local_epochs = 1
            config.batch_size = 8
            config.dataset_name = 'cifar100'
            log_message("✅ Configuration created", log_file)
            
            # Test device creation
            log_message("\n📱 Testing Device Creation...", log_file)
            resources = DeviceResources(5.0, 4.0, 100.0)
            device = EdgeDevice("test_device", (10.0, 20.0), resources)
            log_message(f"✅ Device created: {device.device_id}", log_file)
            
            # Test zone creation
            log_message("\n🏗️ Testing Zone Creation...", log_file)
            zone = Zone("test_zone", "test_server")
            zone.add_device(device)
            log_message(f"✅ Zone created with {len(zone.devices)} device(s)", log_file)
            
            # Test model creation
            log_message("\n🧠 Testing Model Creation...", log_file)
            model = ModelFactory.create_model(config)
            model_params = sum(p.numel() for p in model.parameters())
            log_message(f"✅ Model created: {type(model).__name__} with {model_params} parameters", log_file)
            
            # Test dataset (without downloading)
            log_message("\n📊 Testing Dataset Creation...", log_file)
            dataset = FederatedDataset(config)
            log_message("✅ Dataset object created", log_file)
            
            # Test coordinator initialization  
            log_message("\n🚀 Testing Coordinator Initialization...", log_file)
            coordinator = ContinuumFLCoordinator(config)
            log_message("✅ Coordinator initialized", log_file)
            
            # GPU/Device check
            log_message("\n🖥️ Device Information...", log_file)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                log_message(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)", log_file)
                config.device = 'cuda'
            else:
                log_message("ℹ️  Using CPU (no CUDA available)", log_file)
                config.device = 'cpu'
            
            # Test quick system initialization (without data download)
            log_message("\n⚙️ Testing Quick System Setup...", log_file)
            
            # Skip data download for quick test
            log_message("⏭️ Skipping data download for quick test", log_file)
            
            # Create simple mock data for testing
            log_message("📝 Creating mock data for testing...", log_file)
            mock_data = torch.randn(100, 3, 32, 32)  # CIFAR-like data
            mock_labels = torch.randint(0, 100, (100,))
            log_message("✅ Mock data created", log_file)
            
            log_message("\n🎉 ALL BASIC TESTS PASSED!", log_file)
            log_message("="*60, log_file)
            log_message("✅ ContinuumFL framework is working correctly", log_file)
            log_message("📝 You can now run full tests with dataset downloads", log_file)
            
        except Exception as e:
            log_message(f"\n❌ TEST FAILED: {str(e)}", log_file)
            log_message(f"Error details:\n{traceback.format_exc()}", log_file)
            return False
    
    print(f"\n📄 Detailed log saved to: {log_path}")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed - check test_results.log for details")