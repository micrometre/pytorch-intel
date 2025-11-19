#!/bin/bash

# Intel Extension for PyTorch Setup Script
echo "🚀 Intel Extension for PyTorch (IPEX) Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Virtual environment setup
VENV_DIR=".venv"

if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo ""
    if [ ! -d "$VENV_DIR" ]; then
        echo "🔧 Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create virtual environment"
            echo "   Make sure python3-venv is installed: sudo apt install python3-venv"
            exit 1
        fi
        echo "✅ Virtual environment created at $VENV_DIR"
    else
        echo "✅ Virtual environment already exists at $VENV_DIR"
    fi
    
    echo "🔄 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "✅ Already in virtual environment: $VIRTUAL_ENV"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip

echo "📦 Installing PyTorch and IPEX..."
pip install -r requirements.txt

# Check Intel GPU drivers
echo ""
echo "🔍 Checking Intel GPU support..."

# Check for Intel GPU
lspci | grep -i "vga\|display" | grep -i intel
if [ $? -eq 0 ]; then
    echo "✅ Intel GPU detected"
else
    echo "⚠️  No Intel GPU detected"
fi

# Check for Level Zero
if command -v level_zero_loader &> /dev/null; then
    echo "✅ Level Zero runtime found"
else
    echo "⚠️  Level Zero runtime not found"
    echo "   Install with: sudo apt install level-zero level-zero-dev intel-opencl-icd"
fi

# Check OpenCL
if command -v clinfo &> /dev/null; then
    echo "✅ OpenCL tools found"
    echo "OpenCL devices:"
    clinfo | grep "Device Name" | head -5
else
    echo "⚠️  OpenCL tools not found"
    echo "   Install with: sudo apt install clinfo ocl-icd-opencl-dev"
fi

echo ""
echo "🧪 Testing IPEX installation..."
python -c "
import torch
import intel_extension_for_pytorch as ipex
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ IPEX {ipex.__version__}')
try:
    xpu_available = ipex.xpu.is_available()
    print(f'Intel XPU available: {xpu_available}')
    if xpu_available:
        device_count = ipex.xpu.device_count()
        print(f'XPU device count: {device_count}')
        for i in range(device_count):
            print(f'  Device {i}: {ipex.xpu.get_device_name(i)}')
except Exception as e:
    print(f'XPU check failed: {e}')
"

echo ""
echo "🎯 Ready to run examples!"
echo "   Remember to activate the virtual environment first:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "   Then run:"
echo "   1. Device detection: python device_test.py"
echo "   2. Simple CNN training: python simple_cnn.py"
echo "   3. Computer vision inference: python vision_example.py"
echo ""
echo "📚 For more examples visit:"
echo "   https://intel.github.io/intel-extension-for-pytorch/"