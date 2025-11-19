#!/usr/bin/env python3
"""
Simple Neural Network Example with Intel Extension for PyTorch
This example demonstrates training a simple CNN on synthetic data with IPEX optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch as ipex
import time
import numpy as np

class SimpleCNN(nn.Module):
    """Simple CNN for demonstration"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_synthetic_data(num_samples=1000, device='cpu'):
    """Create synthetic data for testing"""
    # Create random images (32x32 RGB)
    X = torch.randn(num_samples, 3, 32, 32, device=device)
    # Create random labels (10 classes)
    y = torch.randint(0, 10, (num_samples,), device=device)
    return X, y

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    """Train the model"""
    model.train()
    total_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        avg_loss = total_loss / num_batches
        
        print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s')
    
    return total_time

def benchmark_device(device_name, enable_ipex=False):
    """Benchmark training on specified device"""
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device_name.upper()}")
    if enable_ipex:
        print("With IPEX optimization enabled")
    print(f"{'='*60}")
    
    try:
        # Set device
        device = torch.device(device_name)
        
        # Create model
        model = SimpleCNN(num_classes=10)
        model = model.to(device)
        
        # Apply IPEX optimization if enabled
        if enable_ipex and device_name == 'xpu':
            print("🚀 Applying IPEX XPU optimization...")
            model = ipex.optimize(model, device=device)
        elif enable_ipex and device_name == 'cpu':
            print("🚀 Applying IPEX CPU optimization...")
            model = ipex.optimize(model, device=device)
        
        # Create synthetic dataset
        print("📊 Creating synthetic dataset...")
        X, y = create_synthetic_data(num_samples=640, device=device)  # Multiple of 32
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        print("🏋️ Starting training...")
        start_time = time.time()
        total_training_time = train_model(
            model, train_loader, criterion, optimizer, device, epochs=3
        )
        total_time = time.time() - start_time
        
        print(f"\n✅ Training completed successfully!")
        print(f"📈 Total time: {total_time:.2f} seconds")
        print(f"📈 Training time: {total_training_time:.2f} seconds")
        
        return total_time, True
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        return float('inf'), False

def main():
    """Main function"""
    print("🧠 Intel Extension for PyTorch - Neural Network Example")
    print("=" * 60)
    
    # Check device availability
    cpu_available = True
    xpu_available = ipex.xpu.is_available() if hasattr(ipex, 'xpu') else False
    cuda_available = torch.cuda.is_available()
    
    print(f"💻 CPU available: {cpu_available}")
    print(f"🎮 Intel XPU available: {xpu_available}")
    print(f"🚀 CUDA available: {cuda_available}")
    
    results = {}
    
    # Benchmark CPU
    print("\n" + "🖥️ " * 20)
    cpu_time, cpu_success = benchmark_device('cpu', enable_ipex=True)
    if cpu_success:
        results['CPU (IPEX)'] = cpu_time
    
    # Benchmark Intel XPU if available
    if xpu_available:
        print("\n" + "🎮 " * 20)
        xpu_time, xpu_success = benchmark_device('xpu', enable_ipex=True)
        if xpu_success:
            results['Intel XPU (IPEX)'] = xpu_time
    
    # Benchmark CUDA if available
    if cuda_available:
        print("\n" + "🚀 " * 20)
        cuda_time, cuda_success = benchmark_device('cuda:0', enable_ipex=False)
        if cuda_success:
            results['CUDA'] = cuda_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 BENCHMARK RESULTS")
    print("=" * 60)
    
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for i, (device, time_taken) in enumerate(sorted_results, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏃"
            print(f"{icon} {device}: {time_taken:.2f}s")
    else:
        print("❌ No successful benchmarks completed")
    
    if xpu_available and 'Intel XPU (IPEX)' in results:
        print(f"\n🎉 Intel GPU acceleration is working!")
        print(f"   the Iris Xe Graphics completed training in {results['Intel XPU (IPEX)']:.2f}s")
    elif xpu_available:
        print(f"\n⚠️ Intel GPU detected but benchmark failed")
        print("   Check drivers and IPEX installation")
    else:
        print(f"\n📝 Intel GPU not available for acceleration")
        print("   Install Intel GPU drivers and Level Zero runtime")

if __name__ == "__main__":
    main()