# Drug-Target Interaction Prediction

A modular and scalable codebase for predicting drug-target interactions using deep learning with PyTorch Lightning.

## Project Structure

- `config.py`: Configuration dataclasses for all aspects of the project
- `config.yaml`: YAML configuration file with all parameters
- `models.py`: Neural network model implementations
- `trainer.py`: PyTorch Lightning trainer for model training and evaluation
- `dataloader.py`: Dataset and PyTorch Lightning DataModule implementations
- `main.py`: Entry point for training and evaluation
- `train.sh`: Bash script to run the training
- `Dockerfile`: Docker configuration for containerization
- `k8s-deployment.yaml`: Kubernetes job configuration

## Features

- **PyTorch Lightning Integration**: Clean, organized code with built-in distributed training support
- **Modular Design**: Clear separation of concerns for easy maintenance and extension
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Comprehensive Logging**: Built-in TensorBoard logging for metrics visualization
- **Flexible Configuration**: YAML-based configuration system
- **Gradient Accumulation**: Support for larger effective batch sizes
- **Early Stopping**: Automatic early stopping with configurable patience
- **Best Model Checkpointing**: Only saves the best model based on validation metrics
- **Docker Support**: Containerized deployment for reproducible environments
- **Kubernetes Integration**: Ready for cloud deployment and scaling

## Model Architecture

The model uses a dual-encoder architecture with:
- Protein language model (ESM2) for protein sequence encoding
- Molecule language model (ChemBERTa) for molecule SMILES encoding
- Residual inception blocks for feature extraction
- Multi-layer prediction head for affinity prediction

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning 2.0+
- Transformers
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- PyYAML
- lifelines

## Usage

### Configuration

All parameters are specified in a YAML configuration file. You can create your own configuration file based on the provided `config.yaml` template.

The configuration file is organized into sections:
- `model`: Model architecture parameters
- `data`: Data paths and loading parameters
- `training`: Training hyperparameters, including early stopping settings
  - `early_stopping_patience`: Number of epochs with no improvement after which training will be stopped
  - `early_stopping_monitor`: Metric to monitor for early stopping (e.g., "val_loss")
  - `early_stopping_mode`: Mode for monitoring ("min" for loss, "max" for accuracy)
- `logging`: Logging and checkpoint settings
- Other top-level parameters like `seed`, `accelerator`, `devices`, and `strategy`

### Basic Training

```bash
python main.py --config_file config.yaml
```

Or using the provided bash script:

```bash
./train.sh config.yaml
```

### Disabling Sanity Check

PyTorch Lightning performs a sanity check before training by running a few validation batches. If your validation data has issues (e.g., all samples have the same target value), you might encounter errors during this phase. You can disable the sanity check with:

```bash
python main.py --config_file config.yaml --disable_sanity_check
```

Or using the bash script:

```bash
./train.sh config.yaml true
```

### Distributed Training with DDP

PyTorch Lightning handles distributed training automatically. Just specify the strategy in your config:

```yaml
# In config.yaml
accelerator: "gpu"
devices: 4
strategy: "ddp"
```

Then run:

```bash
python main.py --config_file config.yaml
```

### Creating Custom Configurations

You can create custom configuration files for different experiments. For example:

1. Create a file `config_experiment1.yaml`:
```yaml
# Inherit from base config.yaml and override only what you need
model:
  protein_model_name: "facebook/esm2_t12_35M_UR50D"
  molecule_model_name: "DeepChem/ChemBERTa-77M-MTR"
  
data:
  batch_size: 64
  
training:
  learning_rate: 5.0e-5
  weight_decay: 1.0e-5
  early_stopping_patience: 15  # Customize early stopping patience
  
logging:
  experiment_name: "custom_experiment"
```

2. Run with your custom config:
```bash
python main.py --config_file config_experiment1.yaml
```

## Docker Usage

### Building the Docker Image

```bash
docker build -t drug-target-interaction:latest .
```

### Running the Docker Container

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/checkpoints:/app/checkpoints drug-target-interaction:latest
```

To use a custom config file:

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/checkpoints:/app/checkpoints drug-target-interaction:latest custom_config.yaml
```

To disable the sanity check:

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/checkpoints:/app/checkpoints drug-target-interaction:latest config.yaml true
```

## Kubernetes Deployment

1. Build and push the Docker image to your container registry:

```bash
docker build -t your-registry/drug-target-interaction:latest .
docker push your-registry/drug-target-interaction:latest
```

2. Update the image name in `k8s-deployment.yaml` to match your registry.

3. Create the necessary persistent volume claims:

```bash
kubectl apply -f data-pvc.yaml
kubectl apply -f output-pvc.yaml
```

4. Deploy the training job:

```bash
kubectl apply -f k8s-deployment.yaml
```

5. Monitor the job:

```bash
kubectl get jobs
kubectl logs job/drug-target-interaction-training
```

## Troubleshooting

### Deterministic Operations Error

If you encounter an error like `RuntimeError: adaptive_max_pool2d_backward_cuda does not have a deterministic implementation`, it's because some operations in PyTorch don't have deterministic implementations on CUDA. The code has been updated to remove the deterministic setting that was causing this issue.

If you need deterministic behavior for reproducibility, you can manually set it for specific operations using:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

This will use deterministic algorithms where available and only warn (not error) for operations without deterministic implementations.

## Data Format

The training and validation data should be CSV files with the following columns:
- `Molecule Sequence`: SMILES representation of the molecule
- `Protein Sequence`: Amino acid sequence of the protein
- `Binding Affinity`: Binding affinity value (target)

## License

MIT 