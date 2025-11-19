# Intel Extension for PyTorch (IPEX) Example

This directory contains minimal examples for getting started with Intel Extension for PyTorch (IPEX) on Intel Iris Xe Graphics.

## 🚀 Quick Start

1. **Setup and Installation:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Run Examples:**
   ```bash
   # Test device detection and basic operations
   python3 device_test.py
   
   # Train a simple CNN with GPU acceleration
   python3 simple_cnn.py
   
   # Computer vision inference benchmark
   python3 vision_example.py
   ```

## 📁 Files Description

- **`requirements.txt`** - Python dependencies
- **`setup.sh`** - Automated setup script
- **`device_test.py`** - Device detection and basic tensor operations
- **`simple_cnn.py`** - CNN training example with IPEX optimization
- **`vision_example.py`** - Pre-trained model inference benchmark

## 🎮 Intel Iris Xe Graphics Support

Your Intel Raptor Lake-P [Iris Xe Graphics] with 96 EUs is supported by IPEX. Expected capabilities:

- **Inference**: Excellent performance for computer vision models
- **Training**: Good for smaller models and prototype development
- **Memory**: Shares system RAM (efficient memory usage)
- **Precision**: Supports FP32, FP16, and INT8 optimizations

## 🔧 Prerequisites

### Intel GPU Drivers
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install intel-opencl-icd level-zero level-zero-dev clinfo

# Check installation
clinfo | grep "Intel"
```

### Python Environment
```bash
# Create virtual environment (recommended)
python3 -m venv ipex_env
source ipex_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Expected Performance

For your Intel Iris Xe Graphics (96 EUs):

- **ResNet18 Inference**: ~20-50ms per image
- **Training**: 2-5x faster than CPU for appropriate batch sizes
- **Memory**: Up to 50% of system RAM available for models

## 🐛 Troubleshooting

### Common Issues:

1. **"Intel XPU not available"**
   - Install Intel GPU drivers: `sudo apt install intel-opencl-icd`
   - Install Level Zero: `sudo apt install level-zero level-zero-dev`
   - Reboot after driver installation

2. **"ImportError: intel_extension_for_pytorch"**
   - Install IPEX: `pip install intel-extension-for-pytorch`
   - Use compatible PyTorch version

3. **Poor GPU Performance**
   - Check GPU is not throttling: `intel_gpu_top`
   - Ensure adequate cooling
   - Use appropriate batch sizes (16-64 for inference)

### Check System Status:
```bash
# GPU information
lspci | grep -i vga
intel_gpu_top

# OpenCL devices
clinfo | grep "Device Name"

# IPEX version
python3 -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

## 🔗 Useful Resources

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkit.html)
- [Intel GPU Optimization Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/gpu-optimization-guide.html)
- [Level Zero Programming Guide](https://spec.oneapi.io/level-zero/latest/index.html)

## 💡 Performance Tips

1. **Use mixed precision** (FP16) for inference
2. **Batch operations** when possible
3. **Profile with Intel VTune** for optimization
4. **Consider quantization** for production deployment
5. **Use Intel Model Zoo** optimized models

## 🏆 Next Steps

After running these examples:

1. Try real datasets (CIFAR-10, ImageNet)
2. Experiment with different model architectures
3. Explore Intel Model Zoo for pre-optimized models
4. Profile performance with Intel VTune Profiler
5. Consider quantization for deployment

Happy coding with Intel AI acceleration! 🧠✨