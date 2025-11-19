#!/usr/bin/env python3
"""
Intel AI Acceleration Showcase
This example demonstrates:
1. Intel Extension for PyTorch (IPEX) CPU optimization
2. OpenVINO Intel GPU inference
3. Performance comparison
"""

import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
import openvino as ov
import numpy as np
import time
import tempfile
import os

class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def benchmark_pytorch_cpu(model, data, iterations=100):
    """Benchmark PyTorch on CPU"""
    print("🖥️  Benchmarking PyTorch CPU (Standard)...")
    
    model_cpu = model.cpu()
    model_cpu.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(data)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for i in range(iterations):
            output = model_cpu(data)
            if i % 20 == 0:
                print(f"  Completed {i}/{iterations} iterations")
    
    total_time = time.time() - start_time
    avg_time = (total_time / iterations) * 1000  # Convert to ms
    
    print(f"📊 Standard PyTorch CPU:")
    print(f"   Average inference time: {avg_time:.2f} ms")
    print(f"   Throughput: {iterations/total_time:.1f} inferences/second")
    
    return avg_time

def benchmark_ipex_cpu(model, data, iterations=100):
    """Benchmark PyTorch with IPEX CPU optimization"""
    print("\n🚀 Benchmarking Intel Extension for PyTorch (IPEX)...")
    
    model_cpu = model.cpu()
    model_cpu.eval()
    
    # Apply IPEX optimization
    model_ipex = ipex.optimize(model_cpu)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model_ipex(data)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for i in range(iterations):
            output = model_ipex(data)
            if i % 20 == 0:
                print(f"  Completed {i}/{iterations} iterations")
    
    total_time = time.time() - start_time
    avg_time = (total_time / iterations) * 1000  # Convert to ms
    
    print(f"📊 IPEX CPU Optimization:")
    print(f"   Average inference time: {avg_time:.2f} ms")
    print(f"   Throughput: {iterations/total_time:.1f} inferences/second")
    
    return avg_time

def convert_to_openvino(model, input_shape):
    """Convert PyTorch model to OpenVINO format"""
    print("\n🔄 Converting model to OpenVINO IR format...")
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX first
    temp_onnx = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            temp_onnx.name,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # Convert ONNX to OpenVINO IR
        ov_model = ov.convert_model(temp_onnx.name)
        print("✅ Successfully converted to OpenVINO")
        return ov_model
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_onnx.name):
            os.unlink(temp_onnx.name)

def benchmark_openvino_cpu(ov_model, data, iterations=100):
    """Benchmark OpenVINO on CPU"""
    print("\n🔧 Benchmarking OpenVINO CPU...")
    
    # Compile model for CPU
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, 'CPU')
    
    # Convert data to numpy
    input_data = data.numpy()
    
    # Warm-up
    for _ in range(10):
        result = compiled_model([input_data])
    
    # Benchmark
    start_time = time.time()
    for i in range(iterations):
        result = compiled_model([input_data])
        if i % 20 == 0:
            print(f"  Completed {i}/{iterations} iterations")
    
    total_time = time.time() - start_time
    avg_time = (total_time / iterations) * 1000  # Convert to ms
    
    print(f"📊 OpenVINO CPU:")
    print(f"   Average inference time: {avg_time:.2f} ms")
    print(f"   Throughput: {iterations/total_time:.1f} inferences/second")
    
    return avg_time

def benchmark_openvino_gpu(ov_model, data, iterations=100):
    """Benchmark OpenVINO on Intel GPU"""
    print("\n🎮 Benchmarking OpenVINO Intel GPU...")
    
    core = ov.Core()
    
    # Check if GPU device is available
    available_devices = core.available_devices
    gpu_devices = [d for d in available_devices if 'GPU' in d]
    
    if not gpu_devices:
        print("⚠️ No GPU device found in OpenVINO")
        return float('inf')
    
    print(f"🔍 Available devices: {available_devices}")
    gpu_device = gpu_devices[0]
    print(f"📱 Using GPU device: {gpu_device}")
    
    try:
        # Compile model for GPU
        compiled_model = core.compile_model(ov_model, gpu_device)
        
        # Convert data to numpy
        input_data = data.numpy()
        
        # Warm-up
        for _ in range(10):
            result = compiled_model([input_data])
        
        # Benchmark
        start_time = time.time()
        for i in range(iterations):
            result = compiled_model([input_data])
            if i % 20 == 0:
                print(f"  Completed {i}/{iterations} iterations")
        
        total_time = time.time() - start_time
        avg_time = (total_time / iterations) * 1000  # Convert to ms
        
        print(f"📊 OpenVINO Intel GPU ({gpu_device}):")
        print(f"   Average inference time: {avg_time:.2f} ms")
        print(f"   Throughput: {iterations/total_time:.1f} inferences/second")
        
        return avg_time
        
    except Exception as e:
        print(f"❌ GPU benchmark failed: {e}")
        return float('inf')

def main():
    """Main function to run all benchmarks"""
    print("🧠 Intel AI Acceleration Showcase")
    print("=" * 60)
    
    # Setup
    batch_size = 16
    input_size = 784  # 28x28 flattened
    input_shape = (batch_size, 1, 28, 28)
    
    print(f"📋 Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Input shape: {input_shape}")
    print(f"   Model: Simple 3-layer neural network")
    
    # Create model and data
    model = SimpleModel(input_size=input_size)
    data = torch.randn(input_shape)
    
    print(f"\n🔍 System Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   IPEX version: {ipex.__version__}")
    print(f"   OpenVINO version: {ov.__version__}")
    
    # Results storage
    results = {}
    
    # 1. Standard PyTorch CPU
    pytorch_cpu_time = benchmark_pytorch_cpu(model, data, iterations=50)
    results['PyTorch CPU'] = pytorch_cpu_time
    
    # 2. IPEX CPU optimization
    ipex_cpu_time = benchmark_ipex_cpu(model, data, iterations=50)
    results['IPEX CPU'] = ipex_cpu_time
    
    # 3. Convert to OpenVINO
    ov_model = convert_to_openvino(model, input_shape)
    
    if ov_model:
        # 4. OpenVINO CPU
        openvino_cpu_time = benchmark_openvino_cpu(ov_model, data, iterations=50)
        results['OpenVINO CPU'] = openvino_cpu_time
        
        # 5. OpenVINO Intel GPU
        openvino_gpu_time = benchmark_openvino_gpu(ov_model, data, iterations=50)
        if openvino_gpu_time != float('inf'):
            results['OpenVINO Intel GPU'] = openvino_gpu_time
    
    # Print final results
    print(f"\n{'='*60}")
    print("🏆 PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        print(f"{'Method':<25} {'Avg Time (ms)':<15} {'Speedup':<10}")
        print("-" * 50)
        
        baseline = sorted_results[0][1] if sorted_results else 1
        
        for method, avg_time in sorted_results:
            if avg_time != float('inf'):
                speedup = f"{baseline/avg_time:.1f}x" if avg_time != baseline else "baseline"
                print(f"{method:<25} {avg_time:<15.2f} {speedup:<10}")
        
        # Analysis
        if 'IPEX CPU' in results and 'PyTorch CPU' in results:
            ipex_speedup = pytorch_cpu_time / ipex_cpu_time
            print(f"\n📈 IPEX CPU provides {ipex_speedup:.1f}x speedup over standard PyTorch!")
        
        if 'OpenVINO Intel GPU' in results:
            gpu_time = results['OpenVINO Intel GPU']
            cpu_time = results.get('OpenVINO CPU', pytorch_cpu_time)
            gpu_speedup = cpu_time / gpu_time
            print(f"🚀 Your Intel Iris Xe Graphics provides {gpu_speedup:.1f}x speedup!")
            print(f"   GPU inference time: {gpu_time:.2f}ms per batch ({batch_size} samples)")
        
    else:
        print("❌ No successful benchmarks completed")
    
    print(f"\n💡 Your Intel Iris Xe Graphics (96 EUs) Performance Summary:")
    if 'OpenVINO Intel GPU' in results:
        gpu_time = results['OpenVINO Intel GPU']
        print(f"✅ GPU acceleration working: {gpu_time:.2f}ms per {batch_size}-sample batch")
        samples_per_second = (batch_size * 1000) / gpu_time
        print(f"✅ Throughput: {samples_per_second:.0f} samples/second")
    else:
        print("⚠️ GPU acceleration not available")
        print("   Try updating Intel GPU drivers and Level Zero runtime")
    
    print(f"\n🎯 Recommended use cases for your Intel Iris Xe:")
    print(f"   • Computer vision inference (image classification, object detection)")
    print(f"   • Small to medium neural network training")
    print(f"   • Edge AI applications")
    print(f"   • Prototype development and testing")

if __name__ == "__main__":
    main()