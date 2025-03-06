import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from models import AffinityPredictor, DrugTargetInteractionLoss, count_model_parameters
from dataloader import DrugTargetDataModule
from trainer import Trainer
from utils import MetricTracker, set_seed, get_device, setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Drug-Target Interaction Prediction")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/train.csv", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="data/val.csv", help="Path to validation data")
    parser.add_argument("--test_data", type=str, default=None, help="Path to test data")
    
    # Model arguments
    parser.add_argument("--protein_model", type=str, default="facebook/esm2_t6_8M_UR50D", help="Protein language model")
    parser.add_argument("--molecule_model", type=str, default="DeepChem/ChemBERTa-77M-MLM", help="Molecule language model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Distributed training arguments
    parser.add_argument("--distributed_backend", type=str, default="none", choices=["none", "ddp", "fsdp"], 
                        help="Distributed training backend")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--experiment_name", type=str, default="drug_target_interaction", help="Experiment name")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update config with command line arguments"""
    # Update model config
    config.model.protein_model_name = args.protein_model
    config.model.molecule_model_name = args.molecule_model
    
    # Update data config
    config.data.train_data_path = args.train_data
    config.data.val_data_path = args.val_data
    config.data.test_data_path = args.test_data
    config.data.batch_size = args.batch_size
    
    # Update training config
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.early_stopping_patience = args.patience
    config.training.gradient_accumulation_steps = args.grad_accum_steps
    config.training.mixed_precision = args.mixed_precision
    
    # Update distributed config
    config.distributed.distributed_backend = args.distributed_backend
    
    # Update logging config
    config.logging.log_dir = args.log_dir
    config.logging.save_dir = args.save_dir
    config.logging.experiment_name = args.experiment_name
    
    # Update other config
    config.seed = args.seed
    config.device = args.device
    
    return config


def main():
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = Config()
    config = update_config_from_args(config, args)
    
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
        train_data_path=config.data.train_data_path,
        val_data_path=config.data.val_data_path,
        test_data_path=config.data.test_data_path,
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
    
    # Create model
    model = AffinityPredictor(
        protein_model_name=config.model.protein_model_name,
        molecule_model_name=config.model.molecule_model_name,
        hidden_sizes=config.model.hidden_sizes,
        inception_out_channels=config.model.inception_out_channels,
        dropout=config.model.dropout
    )
    
    # Print model parameters
    num_params = count_model_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create loss function
    loss_fn = DrugTargetInteractionLoss(alpha=config.training.loss_alpha)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.scheduler_min_lr,
        verbose=True
    )
    
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
    
    # Cleanup distributed training if needed
    if config.distributed.distributed_backend != "none":
        cleanup_distributed()
    
    return train_losses, val_losses


if __name__ == "__main__":
    main() 