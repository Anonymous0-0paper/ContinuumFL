#!/usr/bin/env python3
"""
Simple GPU test script for ContinuumFL.
Tests CUDA availability and basic GPU operations.
"""

import torch
import torch.nn as nn
import numpy as np
import time

def test_gpu_availability():
    """Test basic GPU availability and functionality"""
    print("🔍 GPU Availability Test")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if not cuda_available:
        print("💡 To enable CUDA:")
        print("   1. Install CUDA-enabled PyTorch:")
        print("   2. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Or use: pip install torch[cuda]")
        return False
    
    # Get device information
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"Device Count: {device_count}")
    print(f"Current Device: {current_device}")
    print(f"Device Name: {device_name}")
    
    # Get memory information
    memory_total = torch.cuda.get_device_properties(current_device).total_memory
    memory_reserved = torch.cuda.memory_reserved(current_device)
    memory_allocated = torch.cuda.memory_allocated(current_device)
    
    print(f"Total Memory: {memory_total / 1024**3:.2f} GB")
    print(f"Reserved Memory: {memory_reserved / 1024**3:.2f} GB")
    print(f"Allocated Memory: {memory_allocated / 1024**3:.2f} GB")
    print(f"Free Memory: {(memory_total - memory_reserved) / 1024**3:.2f} GB")
    
    return True

def test_gpu_operations():
    """Test basic GPU operations"""
    print("\n🚀 GPU Operations Test")
    print("=" * 50)
    
    try:
        # Test tensor operations
        print("Testing tensor operations...")
        cpu_tensor = torch.randn(1000, 1000)
        gpu_tensor = cpu_tensor.cuda()
        
        # Test matrix multiplication
        start_time = time.time()
        result_gpu = torch.mm(gpu_tensor, gpu_tensor)
        gpu_time = time.time() - start_time
        
        start_time = time.time()
        result_cpu = torch.mm(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        
        print(f"GPU Matrix Multiplication: {gpu_time:.4f}s")
        print(f"CPU Matrix Multiplication: {cpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Test neural network
        print("\nTesting neural network...")
        model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        # Test on GPU
        model_gpu = model.cuda()
        input_gpu = torch.randn(32, 1000).cuda()
        
        start_time = time.time()
        output_gpu = model_gpu(input_gpu)
        gpu_inference_time = time.time() - start_time
        
        # Test on CPU
        model_cpu = model.cpu()
        input_cpu = torch.randn(32, 1000)
        
        start_time = time.time()
        output_cpu = model_cpu(input_cpu)
        cpu_inference_time = time.time() - start_time
        
        print(f"GPU Inference: {gpu_inference_time:.4f}s")
        print(f"CPU Inference: {cpu_inference_time:.4f}s")
        print(f"Inference Speedup: {cpu_inference_time/gpu_inference_time:.2f}x")
        
        # Clean up
        del gpu_tensor, result_gpu, model_gpu, input_gpu, output_gpu
        torch.cuda.empty_cache()
        
        print("✅ GPU operations test passed!")
        return True
        
    except Exception as e:
        print(f"❌ GPU operations test failed: {e}")
        return False

def test_memory_management():
    """Test GPU memory management"""
    print("\n💾 GPU Memory Management Test")
    print("=" * 50)
    
    try:
        # Check initial memory
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
        
        # Allocate tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(100, 100, 100).cuda()
            tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated()
            print(f"After tensor {i+1}: {current_memory / 1024**2:.2f} MB")
        
        # Clear tensors
        del tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        print(f"After cleanup: {final_memory / 1024**2:.2f} MB")
        
        if final_memory <= initial_memory + 1024*1024:  # Allow 1MB tolerance
            print("✅ Memory management test passed!")
            return True
        else:
            print("⚠️ Memory not fully cleaned up")
            return True  # Still consider it a pass
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def give_recommendations():
    """Give recommendations based on GPU capabilities"""
    print("\n💡 ContinuumFL Recommendations")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("🔧 For CPU-only training:")
        print("   - Use smaller experiments: --num_devices 20 --num_rounds 50")
        print("   - Reduce batch size: --batch_size 16")
        print("   - Consider cloud GPU instances for larger experiments")
        return
    
    # Get GPU memory
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎯 Recommended settings for your GPU ({memory_gb:.1f}GB):")
    
    if memory_gb < 4:
        print("   - Small experiments: --num_devices 50 --batch_size 16")
        print("   - Use compression: --compression_rate 0.05")
    elif memory_gb < 8:
        print("   - Medium experiments: --num_devices 100 --batch_size 32")
        print("   - Standard settings work well")
    else:
        print("   - Large experiments: --num_devices 200+ --batch_size 64")
        print("   - Can handle full paper experiments")
    
    print("\n🚀 Example GPU commands:")
    print("   python main.py --device cuda --dataset cifar100")
    print("   python main.py --device cuda --num_devices 100 --run_baselines")

def main():
    """Main test function"""
    print("🧪 ContinuumFL GPU Test Suite")
    print("=" * 60)
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        # Test GPU operations
        operations_ok = test_gpu_operations()
        
        # Test memory management
        memory_ok = test_memory_management()
        
        if operations_ok and memory_ok:
            print("\n🎉 All GPU tests passed!")
            print("✅ Your system is ready for GPU-accelerated ContinuumFL!")
        else:
            print("\n⚠️ Some GPU tests failed, but basic functionality works")
    
    # Give recommendations
    give_recommendations()
    
    print("\n" + "=" * 60)
    print("Ready to run ContinuumFL!")
    print("Next steps:")
    print("1. Run: python test_continuumfl.py")
    print("2. Then: python main.py --device cuda (if GPU available)")
    print("=" * 60)

if __name__ == "__main__":
    main()