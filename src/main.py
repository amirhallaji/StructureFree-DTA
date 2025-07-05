import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from models import AffinityPredictor, DrugTargetInteractionLoss, count_model_parameters
from dataloader import DrugTargetDataModule
from trainer import Trainer
from utils import MetricTracker, set_seed, get_device, setup_distributed, cleanup_distributed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Drug-Target Interaction Prediction")
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()


def load_yaml_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    return yaml_config


def update_config_from_yaml(config, yaml_config):
    """Update config from YAML configuration with simpler approach"""
    # Get config sections, using empty dict if section is missing
    model_config = yaml_config.get('model', {})
    data_config = yaml_config.get('data', {})
    training_config = yaml_config.get('training', {})
    logging_config = yaml_config.get('logging', {})
    distributed_config = yaml_config.get('distributed', {})
    
    # Update model config
    config.model.protein_model_name = model_config.get('protein_model_name', config.model.protein_model_name)
    config.model.molecule_model_name = model_config.get('molecule_model_name', config.model.molecule_model_name)
    config.model.hidden_sizes = model_config.get('hidden_sizes', config.model.hidden_sizes)
    config.model.inception_out_channels = model_config.get('inception_out_channels', config.model.inception_out_channels)
    config.model.dropout = model_config.get('dropout', config.model.dropout)
    
    # Update data config
    config.data.path = data_config.get('path', config.data.path)
    config.data.test_size = data_config.get('test_size', config.data.test_size)
    config.data.val_size = data_config.get('val_size', config.data.val_size)
    config.data.random_state = data_config.get('random_state', config.data.random_state)
    config.data.batch_size = data_config.get('batch_size', config.data.batch_size)
    config.data.num_workers = data_config.get('num_workers', config.data.num_workers)
    config.data.max_molecule_length = data_config.get('max_molecule_length', config.data.max_molecule_length)
    config.data.max_protein_length = data_config.get('max_protein_length', config.data.max_protein_length)
    
    # Update training config
    config.training.epochs = training_config.get('epochs', config.training.epochs)
    config.training.learning_rate = training_config.get('learning_rate', config.training.learning_rate)
    config.training.weight_decay = training_config.get('weight_decay', config.training.weight_decay)
    config.training.scheduler_factor = training_config.get('scheduler_factor', config.training.scheduler_factor)
    config.training.scheduler_patience = training_config.get('scheduler_patience', config.training.scheduler_patience)
    config.training.scheduler_min_lr = training_config.get('scheduler_min_lr', config.training.scheduler_min_lr)
    config.training.loss_alpha = training_config.get('loss_alpha', config.training.loss_alpha)
    config.training.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', config.training.gradient_accumulation_steps)
    config.training.early_stopping_patience = training_config.get('early_stopping_patience', config.training.early_stopping_patience)
    config.training.mixed_precision = training_config.get('mixed_precision', config.training.mixed_precision)
    config.training.clip_grad_norm = training_config.get('clip_grad_norm', config.training.clip_grad_norm)
    
    # Update logging config
    config.logging.log_dir = logging_config.get('log_dir', config.logging.log_dir)
    config.logging.save_dir = logging_config.get('save_dir', config.logging.save_dir)
    config.logging.experiment_name = logging_config.get('experiment_name', config.logging.experiment_name)
    config.logging.log_interval = logging_config.get('log_interval', config.logging.log_interval)
    
    # Update distributed config
    config.distributed.distributed_backend = distributed_config.get('distributed_backend', config.distributed.distributed_backend)
    config.distributed.find_unused_parameters = distributed_config.get('find_unused_parameters', config.distributed.find_unused_parameters)
    config.distributed.fsdp_config = distributed_config.get('fsdp_config', config.distributed.fsdp_config)
    
    # Update other config
    config.seed = yaml_config.get('seed', config.seed)
    config.device = yaml_config.get('device', config.device)
    
    return config


def main():
    # Add environment variables to avoid GPU memory fragmentation 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Parse arguments
    args = parse_args()
    
    # Load YAML configuration
    yaml_config = load_yaml_config(args.config_file)
    
    # Create default config and update with YAML config
    config = Config()
    config = update_config_from_yaml(config, yaml_config)
    
    # Print configuration
    print("Configuration:")
    print(f"  Model: {config.model.protein_model_name}, {config.model.molecule_model_name}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Device: {config.device}")
    print(f"  Backend: {config.distributed.distributed_backend}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    
    # Initialize distributed training if needed
    if config.distributed.distributed_backend != "none":
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        setup_distributed(rank, world_size, "nccl" if device.type == "cuda" else "gloo")
    
    # Create data module
    data_module = DrugTargetDataModule(
        data_path=config.data.path,
        test_size=getattr(config.data, 'test_size', 0.2),
        val_size=getattr(config.data, 'val_size', 0.1),
        random_state=getattr(config.data, 'random_state', 0),
        molecule_model_name=config.model.molecule_model_name,
        protein_model_name=config.model.protein_model_name,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_molecule_length=config.data.max_molecule_length,
        max_protein_length=config.data.max_protein_length
    )
    
    # Create dataloaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    # Check if validation dataloader exists
    if val_dataloader is None:
        raise ValueError("Validation dataloader is None. Please ensure val_size > 0 in your configuration.")
    
    # Create model
    model = AffinityPredictor(
        protein_model_name=config.model.protein_model_name,
        molecule_model_name=config.model.molecule_model_name,
        hidden_sizes=config.model.hidden_sizes,
        inception_out_channels=config.model.inception_out_channels,
        dropout=config.model.dropout
    )
    
    num_params = count_model_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    loss_fn = DrugTargetInteractionLoss(alpha=config.training.loss_alpha)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs, eta_min=0)
    
    # Create metric tracker
    metric_tracker = MetricTracker(
        log_dir=config.logging.log_dir,
        experiment_name=config.logging.experiment_name
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        config=config,
        metric_tracker=metric_tracker
    )
    
    # Train model
    train_losses, val_losses = trainer.train()
    
    if config.distributed.distributed_backend != "none":
        cleanup_distributed()
    
    return train_losses, val_losses


if __name__ == "__main__":
    main() 
    # Example usage: python main.py --config_file config.yaml
