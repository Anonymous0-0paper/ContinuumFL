#!/usr/bin/env python3
"""Simple aggregation test"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_aggregation():
    try:
        from src.core.zone import Zone
        from src.core.device import EdgeDevice, DeviceResources
        
        # Create zone with devices
        zone = Zone("test_zone", "test_server")
        
        # Add mock devices
        for i in range(2):
            device_id = f"device_{i}"
            resources = DeviceResources(5.0, 4.0, 100.0)
            device = EdgeDevice(device_id, (10.0*i, 20.0*i), resources)
            device.is_active = True
            device.dataset_size = 100
            device.data_quality_score = 0.9
            device.reliability_score = 0.8
            zone.add_device(device)
        
        print(f"Zone has {len(zone.devices)} devices")
        
        # Create test updates
        device_updates = {
            "device_0": {"param1": torch.tensor([1.0, 2.0])},
            "device_1": {"param1": torch.tensor([3.0, 4.0])}
        }
        
        print("Testing aggregation...")
        result = zone.intra_zone_aggregation(device_updates)
        
        if result:
            print("SUCCESS!")
            for name, tensor in result.items():
                print(f"  {name}: {tensor}")
            return True
        else:
            print("FAILED: Empty result")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_aggregation()