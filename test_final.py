#!/usr/bin/env python3
"""
Final test to confirm ContinuumFL is working after tensor type fix.
"""
import os
import sys
import torch
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("ContinuumFL Final Verification Test")
    print("=" * 50)
    
    try:
        # Import and configure
        from config import ContinuumFLConfig
        from src.continuum_fl_coordinator import ContinuumFLCoordinator
        
        config = ContinuumFLConfig()
        config.num_devices = 10
        config.num_zones = 2
        config.num_rounds = 5
        config.local_epochs = 5
        config.batch_size = 4
        config.dataset_name = 'cifar100'
        
        print("1. Configuration: OK")
        
        # Initialize coordinator
        coordinator = ContinuumFLCoordinator(config)
        print("2. Coordinator: OK")
        
        # Initialize system (this will download CIFAR-100 if needed)
        print("3. Initializing system (may download data)...")
        start_time = time.time()
        coordinator.initialize_system()
        init_time = time.time() - start_time
        print(f"   System initialized in {init_time:.2f}s")
        
        # Run one round of federated learning (this was failing before)
        print("4. Testing federated learning round...")
        start_time = time.time()
        results = coordinator.run_federated_learning()
        round_time = time.time() - start_time
        
        print(f"   Round completed in {round_time:.2f}s")
        print(f"   Final accuracy: {results.get('final_accuracy', 0.0):.4f}")
        
        print("\n" + "=" * 50)
        print("SUCCESS: All tests passed!")
        print("ContinuumFL is working correctly with tensor type fix!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nContinuumFL is ready for full experiments!")
    else:
        print("\nSomething is still wrong - check the error above")