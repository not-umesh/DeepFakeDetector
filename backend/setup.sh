#!/bin/bash

# DeepFake Detector - Pro Analysis Setup Script
echo "========================================"
echo "DeepFake Detector - Pro Backend Setup"
echo "========================================"

# Navigate to backend directory
cd "$(dirname "$0")"

# Check Python version
echo ""
echo "Checking Python..."
python3 --version || { echo "Python 3 is required!"; exit 1; }

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install CLIP
echo ""
echo "Installing OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Create pretrained_weights directory
echo ""
echo "Creating directories..."
mkdir -p pretrained_weights

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To download pretrained weights (optional):"
echo "  Download from: https://github.com/WisconsinAIVision/UniversalFakeDetect"
echo "  Place fc_weights.pth in: backend/pretrained_weights/"
echo ""
echo "To start the server:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  python server.py"
echo "========================================"
