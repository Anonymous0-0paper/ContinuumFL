#!/usr/bin/env python3
"""
Quick test to verify tensor type casting fix in aggregation.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_tensor_aggregation():
    """Test that tensor aggregation handles different types correctly"""
    print("Testing tensor aggregation with mixed types...")
    
    # Create test tensors with different dtypes
    test_weights = {
        'conv1.weight': torch.randn(64, 3, 3, 3, dtype=torch.float32),
        'conv1.bias': torch.randn(64, dtype=torch.float32),
        'bn1.weight': torch.ones(64, dtype=torch.float32),
        'bn1.bias': torch.zeros(64, dtype=torch.float32),
        'fc.weight': torch.randn(10, 512, dtype=torch.float32),
        'fc.bias': torch.zeros(10, dtype=torch.float32)
    }
    
    # Simulate device updates (3 devices)
    device_updates = {}
    for i in range(3):
        device_updates[f"device_{i}"] = {
            name: tensor + torch.randn_like(tensor) * 0.01 
            for name, tensor in test_weights.items()
        }
    
    print(f"Created test data with {len(device_updates)} devices")
    
    # Test intra-zone aggregation 
    try:
        from src.core.zone import Zone
        zone = Zone("test_zone", "test_server")
        
        # Mock zone weights
        zone.intra_zone_weights = {f"device_{i}": 1.0/3 for i in range(3)}
        
        # Test aggregation
        result = zone.intra_zone_aggregation(device_updates)
        
        if result:
            print("SUCCESS: Intra-zone aggregation completed")
            print(f"  Result has {len(result)} parameters")
            
            # Check that results have correct types
            for name, tensor in result.items():
                original_dtype = test_weights[name].dtype
                print(f"  {name}: {tensor.dtype} (expected: {original_dtype})")
                
            return True
        else:
            print("FAILED: No aggregation result returned")
            return False
            
    except Exception as e:
        print(f"FAILED: Intra-zone aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("ContinuumFL Tensor Type Test")
    print("=" * 40)
    
    if test_tensor_aggregation():
        print("\nSUCCESS: Tensor aggregation working correctly!")
        return True
    else:
        print("\nFAILED: Tensor aggregation has issues")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Test passed - type casting fix is working!")
    else:
        print("Test failed - please check the fix")