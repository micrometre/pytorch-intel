#!/usr/bin/env python3
"""
Computer Vision Example with Intel Extension for PyTorch
Real-world example using a pre-trained model for image classification.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import intel_extension_for_pytorch as ipex
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

def create_sample_image():
    """Create a sample RGB image for testing"""
    # Create a simple synthetic image with patterns
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Add some patterns
    for i in range(0, 224, 20):
        image[i:i+10, :, 0] = 255  # Red stripes
    for j in range(0, 224, 30):
        image[:, j:j+15, 1] = 255  # Green stripes
    
    # Add some noise
    noise = np.random.randint(0, 50, (224, 224, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image)

def load_model(device):
    """Load a pre-trained ResNet model"""
    print(f"🤖 Loading ResNet18 model on {device}...")
    
    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    
    return model

def preprocess_image(image):
    """Preprocess image for ResNet"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension
    return transform(image).unsqueeze(0)

def inference_benchmark(model, input_tensor, device, num_runs=100):
    """Benchmark inference performance"""
    print(f"⚡ Running {num_runs} inference iterations...")
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_runs):
            output = model(input_tensor)
            if i % 20 == 0:
                print(f"  Completed {i}/{num_runs} iterations")
    
    total_time = time.time() - start_time
    avg_time = (total_time / num_runs) * 1000  # Convert to ms
    
    print(f"📊 Average inference time: {avg_time:.2f} ms")
    print(f"📊 Total time for {num_runs} inferences: {total_time:.2f} s")
    print(f"📊 Throughput: {num_runs/total_time:.1f} inferences/second")
    
    return avg_time, output

def run_inference_test(device_name, enable_ipex=False):
    """Run inference test on specified device"""
    print(f"\n{'='*60}")
    print(f"🔬 INFERENCE TEST - {device_name.upper()}")
    if enable_ipex:
        print("🚀 IPEX Optimization: ENABLED")
    print(f"{'='*60}")
    
    try:
        device = torch.device(device_name)
        
        # Create sample image
        print("🖼️  Creating sample image...")
        image = create_sample_image()
        
        # Preprocess image
        input_tensor = preprocess_image(image).to(device)
        print(f"📥 Input tensor shape: {input_tensor.shape}")
        
        # Load model
        model = load_model(device)
        
        # Apply IPEX optimization
        if enable_ipex:
            if device_name == 'xpu':
                print("🔧 Applying IPEX XPU optimization...")
                model = ipex.optimize(model, device=device)
            elif device_name == 'cpu':
                print("🔧 Applying IPEX CPU optimization...")
                model = ipex.optimize(model, device=device)
        
        # Run benchmark
        avg_time, output = inference_benchmark(model, input_tensor, device, num_runs=50)
        
        # Get predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        print(f"\n🎯 Top 5 predictions (ImageNet classes):")
        for i in range(5):
            print(f"  Class {top5_idx[i].item()}: {top5_prob[i].item():.1%}")
        
        return avg_time, True
        
    except Exception as e:
        print(f"❌ Error during inference test: {e}")
        return float('inf'), False

def main():
    """Main function"""
    print("🔍 Intel Extension for PyTorch - Computer Vision Example")
    print("=" * 60)
    
    # Check device availability
    cpu_available = True
    xpu_available = ipex.xpu.is_available() if hasattr(ipex, 'xpu') else False
    cuda_available = torch.cuda.is_available()
    
    print(f"💻 CPU available: {cpu_available}")
    print(f"🎮 Intel XPU available: {xpu_available}")
    print(f"🚀 CUDA available: {cuda_available}")
    
    results = {}
    
    # Test CPU with IPEX
    cpu_time, cpu_success = run_inference_test('cpu', enable_ipex=True)
    if cpu_success:
        results['CPU (IPEX)'] = cpu_time
    
    # Test Intel XPU if available
    if xpu_available:
        xpu_time, xpu_success = run_inference_test('xpu', enable_ipex=True)
        if xpu_success:
            results['Intel XPU (IPEX)'] = xpu_time
    
    # Test CUDA if available
    if cuda_available:
        cuda_time, cuda_success = run_inference_test('cuda:0', enable_ipex=False)
        if cuda_success:
            results['CUDA'] = cuda_time
    
    # Print final results
    print(f"\n{'='*60}")
    print("🏆 INFERENCE BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        print(f"{'Device':<20} {'Avg Time (ms)':<15} {'Speedup':<10}")
        print("-" * 45)
        
        baseline = sorted_results[0][1]  # Fastest as baseline
        
        for device, avg_time in sorted_results:
            speedup = f"{baseline/avg_time:.1f}x" if avg_time != baseline else "baseline"
            print(f"{device:<20} {avg_time:<15.2f} {speedup:<10}")
        
        if xpu_available and 'Intel XPU (IPEX)' in results:
            xpu_time = results['Intel XPU (IPEX)']
            print(f"\n🎉 the Intel Iris Xe Graphics achieved {xpu_time:.2f}ms per inference!")
            
            if 'CPU (IPEX)' in results:
                cpu_time = results['CPU (IPEX)']
                speedup = cpu_time / xpu_time
                print(f"🚀 That's {speedup:.1f}x faster than CPU!")
        
    else:
        print("❌ No successful benchmarks completed")
    
    print(f"\n💡 Tips for better performance:")
    print("   • Use mixed precision (FP16) for inference")
    print("   • Batch multiple images together")
    print("   • Use Intel-optimized models from Model Zoo")
    print("   • Consider quantization for production deployment")

if __name__ == "__main__":
    main()