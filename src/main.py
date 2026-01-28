import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config
from models import AffinityPredictor, DrugTargetInteractionLoss, count_model_parameters
from dataloader import DrugTargetDataModule
from trainer import Trainer
from utils import MetricTracker, set_seed, get_device, setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Drug-Target Interaction Prediction")
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML configuration file")
    return parser.parse_args()


def load_yaml_config(config_file):
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    return yaml_config


def update_config_from_yaml(config, yaml_config):
    model_config = yaml_config.get('model', {})
    data_config = yaml_config.get('data', {})
    training_config = yaml_config.get('training', {})
    logging_config = yaml_config.get('logging', {})
    distributed_config = yaml_config.get('distributed', {})
    
    config.model.protein_model_name = model_config.get('protein_model_name', config.model.protein_model_name)
    config.model.molecule_model_name = model_config.get('molecule_model_name', config.model.molecule_model_name)
    config.model.hidden_sizes = model_config.get('hidden_sizes', config.model.hidden_sizes)
    config.model.inception_out_channels = model_config.get('inception_out_channels', config.model.inception_out_channels)
    config.model.dropout = model_config.get('dropout', config.model.dropout)
    
    config.data.path = data_config.get('path', config.data.path)
    config.data.n_folds = data_config.get('n_folds', config.data.n_folds)
    config.data.val_size = data_config.get('val_size', config.data.val_size)
    config.data.random_state = data_config.get('random_state', config.data.random_state)
    config.data.batch_size = data_config.get('batch_size', config.data.batch_size)
    config.data.num_workers = data_config.get('num_workers', config.data.num_workers)
    config.data.max_molecule_length = data_config.get('max_molecule_length', config.data.max_molecule_length)
    config.data.max_protein_length = data_config.get('max_protein_length', config.data.max_protein_length)
    
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
    
    config.logging.log_dir = logging_config.get('log_dir', config.logging.log_dir)
    config.logging.save_dir = logging_config.get('save_dir', config.logging.save_dir)
    config.logging.experiment_name = logging_config.get('experiment_name', config.logging.experiment_name)
    config.logging.log_interval = logging_config.get('log_interval', config.logging.log_interval)
    
    config.distributed.distributed_backend = distributed_config.get('distributed_backend', config.distributed.distributed_backend)
    config.distributed.find_unused_parameters = distributed_config.get('find_unused_parameters', config.distributed.find_unused_parameters)
    config.distributed.fsdp_config = distributed_config.get('fsdp_config', config.distributed.fsdp_config)
    
    config.seed = yaml_config.get('seed', config.seed)
    config.device = yaml_config.get('device', config.device)
    
    return config


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    yaml_config = load_yaml_config(args.config_file)
    config = Config()
    config = update_config_from_yaml(config, yaml_config)
    
    print("Configuration:")
    print(f"  Model: {config.model.protein_model_name}, {config.model.molecule_model_name}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Device: {config.device}")
    print(f"  N-Folds: {config.data.n_folds}")
    
    set_seed(config.seed)
    device = get_device(config.device)
    
    if config.distributed.distributed_backend != "none":
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        setup_distributed(rank, world_size, "nccl" if device.type == "cuda" else "gloo")
    
    data_module = DrugTargetDataModule(
        data_path=config.data.path,
        n_folds=config.data.n_folds,
        val_size=getattr(config.data, 'val_size', 0.1),
        random_state=getattr(config.data, 'random_state', 0),
        molecule_model_name=config.model.molecule_model_name,
        protein_model_name=config.model.protein_model_name,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_molecule_length=config.data.max_molecule_length,
        max_protein_length=config.data.max_protein_length
    )
    
    all_fold_results = []
    
    for fold in range(config.data.n_folds):
        print(f"\n{'='*80}")
        print(f"Training Fold {fold + 1}/{config.data.n_folds}")
        print(f"{'='*80}\n")
        
        data_module.setup_fold(fold)
        
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        
        if val_dataloader is None:
            raise ValueError("Validation dataloader is None. Please ensure val_size > 0 in your configuration.")
        
        model = AffinityPredictor(
            protein_model_name=config.model.protein_model_name,
            molecule_model_name=config.model.molecule_model_name,
            hidden_sizes=config.model.hidden_sizes,
            inception_out_channels=config.model.inception_out_channels,
            dropout=config.model.dropout
        )
        
        if fold == 0:
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
        
        metric_tracker = MetricTracker(
            log_dir=config.logging.log_dir,
            experiment_name=f"{config.logging.experiment_name}_fold{fold}"
        )
        
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            config=config,
            metric_tracker=metric_tracker,
            fold=fold
        )
        
        train_losses, val_losses = trainer.train()
        
        fold_results = {
            'fold': fold,
            'best_val_loss': metric_tracker.best_metrics['val_loss'],
            'best_val_r2': metric_tracker.best_metrics['val_r2'],
            'best_val_ci': metric_tracker.best_metrics['val_ci'],
            'best_epoch': metric_tracker.best_metrics['epoch']
        }
        all_fold_results.append(fold_results)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Val Loss: {fold_results['best_val_loss']:.4f}")
        print(f"  Best Val R2: {fold_results['best_val_r2']:.4f}")
        print(f"  Best Val CI: {fold_results['best_val_ci']:.4f}")
        print(f"  Best Epoch: {fold_results['best_epoch']}")
    
    if config.distributed.distributed_backend != "none":
        cleanup_distributed()
    
    print(f"\n{'='*80}")
    print("Cross-Validation Results Summary")
    print(f"{'='*80}\n")
    
    avg_val_loss = np.mean([r['best_val_loss'] for r in all_fold_results])
    std_val_loss = np.std([r['best_val_loss'] for r in all_fold_results])
    avg_val_r2 = np.mean([r['best_val_r2'] for r in all_fold_results])
    std_val_r2 = np.std([r['best_val_r2'] for r in all_fold_results])
    avg_val_ci = np.mean([r['best_val_ci'] for r in all_fold_results])
    std_val_ci = np.std([r['best_val_ci'] for r in all_fold_results])
    
    print(f"Average Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"Average Val R2: {avg_val_r2:.4f} ± {std_val_r2:.4f}")
    print(f"Average Val CI: {avg_val_ci:.4f} ± {std_val_ci:.4f}")
    
    print("\nPer-Fold Results:")
    for result in all_fold_results:
        print(f"  Fold {result['fold'] + 1}: Loss={result['best_val_loss']:.4f}, R2={result['best_val_r2']:.4f}, CI={result['best_val_ci']:.4f}")
    
    return all_fold_results


if __name__ == "__main__":
    main()