#!/bin/bash

# Set default config file
CONFIG_FILE=${1:-config.yaml}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# Print training information
echo "Starting training with config: $CONFIG_FILE"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run training
if [ "$DISABLE_SANITY_CHECK" = "true" ]; then
    echo "Running without sanity check..."
    python main.py --config_file $CONFIG_FILE
else
    echo "Running with sanity check..."
    python main.py --config_file $CONFIG_FILE
fi

echo "Training completed!" 