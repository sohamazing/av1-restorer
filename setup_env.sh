#!/bin/bash

# AURA-Net Project Setup Script
# This script creates a conda environment and installs dependencies
# for both macOS (Apple Silicon) and Linux (NVIDIA CUDA).

# --- Configuration ---
ENV_NAME="aura"

# --- Main Logic ---
echo "Starting AURA Environment Setup..."

# 1. Check if the conda environment already exists
if conda env list | grep -q "^\s*$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment. Please check your conda installation."
        exit 1
    fi
fi

echo "Activating environment '$ENV_NAME' for installation..."

# 2. Install PyTorch based on the Operating System
OS_TYPE=$(uname -s)

if [ "$OS_TYPE" == "Darwin" ]; then
    # macOS (Apple Silicon with MPS)
    echo "Detected macOS. Installing PyTorch with MPS support..."
    conda run -n $ENV_NAME pip install torch torchvision torchaudio
elif [ "$OS_TYPE" == "Linux" ]; then
    # Linux (assuming NVIDIA GPU with CUDA 11.8)
    echo "Detected Linux. Installing PyTorch with CUDA 11.8 support..."
    conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Error: Unsupported Operating System '$OS_TYPE'. Please install PyTorch manually."
    exit 1
fi

# Verify PyTorch installation
conda run -n $ENV_NAME python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('GPU/Accelerator available:', torch.cuda.is_available() or torch.backends.mps.is_available())"
if [ $? -ne 0 ]; then
    echo "Error: PyTorch installation failed. Please check the errors above."
    exit 1
fi


# 3. Install all other dependencies from requirements.txt
echo "Installing other dependencies from requirements.txt..."
conda run -n $ENV_NAME pip install -r requirements.txt

echo ""
echo "AURA environment setup complete!"
echo "To activate the environment, run: conda activate $ENV_NAME"

