#!/bin/bash

# Drug-Target Interaction Model Preparation Script
# This script sets up the environment and creates necessary directories

set -e  # Exit on any error

echo "Starting preparation for Drug-Target Interaction model..."

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints

echo "Preparation completed successfully!"
echo "To run the model, use: python src/main.py --config_file cfg/config.yaml"