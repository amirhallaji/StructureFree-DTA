# Drug-Target Interaction Prediction

A modular and scalable codebase for predicting drug-target interactions using deep learning.

## Project Structure

- `config.py`: Configuration dataclasses for all aspects of the project
- `config.yaml`: YAML configuration file with all parameters
- `models.py`: Neural network model implementations
- `datasets.py`: Dataset and dataloader implementations
- `trainer.py`: Modular trainer with support for distributed training
- `utils.py`: Utility functions and metric tracking
- `main.py`: Entry point for training and evaluation

## Features

- **Modular Design**: Clean separation of concerns for easy maintenance and extension
- **Distributed Training**: Support for DDP (DistributedDataParallel) and FSDP (FullyShardedDataParallel)
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Comprehensive Logging**: Detailed metrics tracking and visualization
- **Flexible Configuration**: YAML-based configuration system
- **Gradient Accumulation**: Support for larger effective batch sizes
- **Early Stopping**: Automatic early stopping to prevent overfitting

## Model Architecture

The model uses a dual-encoder architecture with:
- Protein language model (ESM2) for protein sequence encoding
- Molecule language model (ChemBERTa) for molecule SMILES encoding
- Residual inception blocks for feature extraction
- Multi-layer prediction head for affinity prediction

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- PyYAML

## Usage

### Configuration

All parameters are specified in a YAML configuration file. You can create your own configuration file based on the provided `config.yaml` template.

The configuration file is organized into sections:
- `model`: Model architecture parameters
- `data`: Data paths and loading parameters
- `training`: Training hyperparameters
- `logging`: Logging and checkpoint settings
- `distributed`: Distributed training options
- Other top-level parameters like `seed` and `device`

### Basic Training

```bash
python main.py --config_file config.yaml
```

### Distributed Training with DDP

```bash
torchrun --nproc_per_node=4 main.py --config_file config_ddp.yaml
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
  early_stopping_patience: 15
  
logging:
  experiment_name: "custom_experiment"
```

2. Run with your custom config:
```bash
python main.py --config_file config_experiment1.yaml
```

## Data Format

The training and validation data should be CSV files with the following columns:
- `molecule_smiles`: SMILES representation of the molecule
- `protein_sequence`: Amino acid sequence of the protein
- `binding_affinity`: Binding affinity value (target)

## License

MIT 