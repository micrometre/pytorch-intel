#!/usr/bin/env python3
"""
Intel Extension for PyTorch (IPEX) - Device Detection Example
This script checks available devices and Intel GPU support.
"""

import torch
import intel_extension_for_pytorch as ipex
import sys

def check_ipex_setup():
    """Check IPEX installation and available devices"""
    print("=" * 60)
    print("Intel Extension for PyTorch (IPEX) Setup Check")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"IPEX version: {ipex.__version__}")
    
    # Check CUDA availability (if any)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Check Intel GPU (XPU) availability
    try:
        xpu_available = ipex.xpu.is_available()
        print(f"Intel XPU available: {xpu_available}")
        
        if xpu_available:
            device_count = ipex.xpu.device_count()
            print(f"Intel XPU device count: {device_count}")
            
            for i in range(device_count):
                device_name = ipex.xpu.get_device_name(i)
                print(f"  Device {i}: {device_name}")
                
        return xpu_available
    except Exception as e:
        print(f"Intel XPU check failed: {e}")
        return False

def test_basic_operations(device):
    """Test basic tensor operations on specified device"""
    print(f"\nTesting basic operations on {device}...")
    
    try:
        # Create tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        start_time = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
        end_time = torch.cuda.Event(enable_timing=True) if device.startswith('cuda') else None
        
        import time
        start = time.time()
        
        if start_time:
            start_time.record()
        
        z = torch.matmul(x, y)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
        else:
            elapsed_time = (time.time() - start) * 1000  # Convert to ms
        
        print(f"  Matrix multiplication (1000x1000): {elapsed_time:.2f} ms")
        print(f"  Result shape: {z.shape}")
        print(f"  Result mean: {z.mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"  Error during testing: {e}")
        return False

def main():
    """Main function to run IPEX device checks and tests"""
    
    # Check setup
    xpu_available = check_ipex_setup()
    
    print("\n" + "=" * 60)
    print("Performance Testing")
    print("=" * 60)
    
    # Test CPU
    print("\n🖥️  Testing CPU performance...")
    cpu_success = test_basic_operations('cpu')
    
    # Test Intel GPU if available
    if xpu_available:
        print("\n🎮 Testing Intel XPU (GPU) performance...")
        xpu_success = test_basic_operations('xpu')
        
        if xpu_success:
            print("\n✅ Intel GPU acceleration is working!")
        else:
            print("\n❌ Intel GPU acceleration failed")
    else:
        print("\n⚠️  Intel XPU not available. Make sure:")
        print("   1. Intel GPU drivers are installed")
        print("   2. Intel Extension for PyTorch supports the GPU")
        print("   3. Level Zero runtime is available")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("\n🚀 Testing CUDA performance...")
        cuda_success = test_basic_operations('cuda:0')
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"CPU: {'✅ Working' if cpu_success else '❌ Failed'}")
    if xpu_available:
        print(f"Intel XPU: {'✅ Working' if 'xpu_success' in locals() and xpu_success else '❌ Failed'}")
    else:
        print("Intel XPU: ⚠️ Not available")
    
    if torch.cuda.is_available():
        print(f"CUDA: {'✅ Working' if 'cuda_success' in locals() and cuda_success else '❌ Failed'}")

if __name__ == "__main__":
    main()