#!/bin/bash

CONFIG_FILE=${1:-cfg/config.yaml}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

echo "Starting training with config: $CONFIG_FILE"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run training
if [ "$DISABLE_SANITY_CHECK" = "true" ]; then
    echo "Running without sanity check..."
    python src/main.py --config_file $CONFIG_FILE
else
    echo "Running with sanity check..."
    python src/main.py --config_file $CONFIG_FILE
fi

echo "Training completed!" 