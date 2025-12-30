#!/bin/bash
# Convenience script to activate the virtual environment

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "🔄 Activating Intel PyTorch environment..."
    source "$VENV_DIR/bin/activate"
    echo "✅ Environment activated!"
    echo ""
    echo "💡 Available examples:"
    echo "   python device_test.py      # Test device detection"
    echo "   python simple_cnn.py       # Train CNN with GPU"
    echo "   python vision_example.py   # Computer vision benchmark"
    echo ""
    echo "   To deactivate: deactivate"
else
    echo "❌ Virtual environment not found."
    echo "   Run ./setup.sh first to create it."
fi