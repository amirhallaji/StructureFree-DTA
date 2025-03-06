# Drug-Target Interaction Prediction

A modular and scalable codebase for predicting drug-target interactions using deep learning.

## Project Structure

- `config.py`: Configuration dataclasses for all aspects of the project
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
- **Flexible Configuration**: Dataclass-based configuration system
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

## Usage

### Basic Training

```bash
python main.py --train_data data/train.csv --val_data data/val.csv
```

### Distributed Training with DDP

```bash
torchrun --nproc_per_node=4 main.py --distributed_backend ddp --train_data data/train.csv --val_data data/val.csv
```

### Mixed Precision Training

```bash
python main.py --train_data data/train.csv --val_data data/val.csv --mixed_precision
```

### Custom Configuration

```bash
python main.py \
  --train_data data/train.csv \
  --val_data data/val.csv \
  --protein_model facebook/esm2_t12_35M_UR50D \
  --molecule_model DeepChem/ChemBERTa-77M-MTR \
  --batch_size 64 \
  --epochs 200 \
  --lr 5e-5 \
  --weight_decay 1e-5 \
  --patience 15 \
  --grad_accum_steps 2 \
  --mixed_precision \
  --experiment_name custom_experiment
```

## Data Format

The training and validation data should be CSV files with the following columns:
- `molecule_smiles`: SMILES representation of the molecule
- `protein_sequence`: Amino acid sequence of the protein
- `binding_affinity`: Binding affinity value (target)

## License

MIT 