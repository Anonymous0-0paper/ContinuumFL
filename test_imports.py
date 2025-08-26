#!/usr/bin/env python3
"""
Simple import test to identify what's failing.
"""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_single_import(module_name, from_package=None):
    """Test importing a single module"""
    try:
        if from_package:
            print(f"Testing: from {from_package} import {module_name}")
            exec(f"from {from_package} import {module_name}")
        else:
            print(f"Testing: import {module_name}")
            exec(f"import {module_name}")
        print("✅ SUCCESS")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"Details: {traceback.format_exc()}")
        return False

def main():
    print("🔍 Testing Individual Imports...")
    print("="*50)
    
    # Test basic imports
    print("\n📦 Testing Basic Imports:")
    test_single_import("config")
    
    print("\n📦 Testing Core Imports:")
    test_single_import("device", "src.core")
    test_single_import("zone", "src.core")
    test_single_import("zone_discovery", "src.core")
    
    print("\n📦 Testing Model Factory:")
    test_single_import("model_factory", "src.models")
    
    print("\n📦 Testing Dataset:")
    test_single_import("federated_dataset", "src.data")
    
    print("\n📦 Testing Aggregation:")
    test_single_import("hierarchical_aggregator", "src.aggregation")
    
    print("\n📦 Testing Communication:")
    test_single_import("compression", "src.communication")
    
    print("\n📦 Testing Baselines:")
    test_single_import("baseline_fl", "src.baselines")
    
    print("\n🚀 Testing Main Coordinator:")
    test_single_import("continuum_fl_coordinator", "src")
    
    print("\n" + "="*50)
    print("Import testing completed!")

if __name__ == "__main__":
    main()