import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config
from models import AffinityPredictor
from trainer import Trainer
from dataloader import DrugTargetDataModule


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
    """Update config from YAML configuration"""
    model_config = yaml_config.get('model', {})
    data_config = yaml_config.get('data', {})
    training_config = yaml_config.get('training', {})
    logging_config = yaml_config.get('logging', {})
    
    config.model.protein_model_name = model_config.get('protein_model_name', config.model.protein_model_name)
    config.model.molecule_model_name = model_config.get('molecule_model_name', config.model.molecule_model_name)
    config.model.hidden_sizes = model_config.get('hidden_sizes', config.model.hidden_sizes)
    config.model.inception_out_channels = model_config.get('inception_out_channels', config.model.inception_out_channels)
    config.model.dropout = model_config.get('dropout', config.model.dropout)
    
    config.data.train_data_path = data_config.get('train_data_path', config.data.train_data_path)
    config.data.val_data_path = data_config.get('val_data_path', config.data.val_data_path)
    config.data.test_data_path = data_config.get('test_data_path', config.data.test_data_path)
    config.data.batch_size = data_config.get('batch_size', config.data.batch_size)
    config.data.num_workers = data_config.get('num_workers', config.data.num_workers)
    config.data.max_molecule_length = data_config.get('max_molecule_length', config.data.max_molecule_length)
    config.data.max_protein_length = data_config.get('max_protein_length', config.data.max_protein_length)
    
    config.training.max_epochs = training_config.get('max_epochs', config.training.max_epochs)
    config.training.learning_rate = training_config.get('learning_rate', config.training.learning_rate)
    config.training.weight_decay = training_config.get('weight_decay', config.training.weight_decay)
    config.training.loss_alpha = training_config.get('loss_alpha', config.training.loss_alpha)
    config.training.gradient_clip_val = training_config.get('gradient_clip_val', config.training.gradient_clip_val)
    config.training.accumulate_grad_batches = training_config.get('accumulate_grad_batches', config.training.accumulate_grad_batches)
    config.training.precision = training_config.get('precision', config.training.precision)
    config.training.early_stopping_patience = training_config.get('early_stopping_patience', config.training.early_stopping_patience)
    config.training.early_stopping_monitor = training_config.get('early_stopping_monitor', config.training.early_stopping_monitor)
    config.training.early_stopping_mode = training_config.get('early_stopping_mode', config.training.early_stopping_mode)
    config.training.lr_scheduler_factor = training_config.get('lr_scheduler_factor', config.training.lr_scheduler_factor)
    config.training.lr_scheduler_patience = training_config.get('lr_scheduler_patience', config.training.lr_scheduler_patience)
    config.training.lr_scheduler_min_lr = training_config.get('lr_scheduler_min_lr', config.training.lr_scheduler_min_lr)
    
    config.logging.log_dir = logging_config.get('log_dir', config.logging.log_dir)
    config.logging.save_dir = logging_config.get('save_dir', config.logging.save_dir)
    config.logging.experiment_name = logging_config.get('experiment_name', config.logging.experiment_name)
    config.logging.log_every_n_steps = logging_config.get('log_every_n_steps', config.logging.log_every_n_steps)
    
    config.seed = yaml_config.get('seed', config.seed)
    config.accelerator = yaml_config.get('accelerator', config.accelerator)
    config.devices = yaml_config.get('devices', config.devices)
    config.strategy = yaml_config.get('strategy', config.strategy)
    
    return config


def main():
    """Main function to run the training pipeline"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    yaml_config = load_yaml_config(args.config_file)
    config = Config()
    config = update_config_from_yaml(config, yaml_config)
    
    print(f"Model: {config.model.protein_model_name}, {config.model.molecule_model_name}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Accelerator: {config.accelerator}")
    print(f"Strategy: {config.strategy}")
    print(f"Early stopping patience: {config.training.early_stopping_patience}")
    
    pl.seed_everything(config.seed)
    
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
    
    model = AffinityPredictor(
        protein_model_name=config.model.protein_model_name,
        molecule_model_name=config.model.molecule_model_name,
        hidden_sizes=config.model.hidden_sizes,
        inception_out_channels=config.model.inception_out_channels,
        dropout=config.model.dropout
    )
    
    trainer_module = Trainer(model=model, config=config)
    
    callbacks = [
        # Save only the best model checkpoint
        ModelCheckpoint(
            dirpath=os.path.join(config.logging.save_dir, config.logging.experiment_name),
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor=config.training.early_stopping_monitor,
            mode=config.training.early_stopping_mode,
            save_top_k=1,  # Only save the best model
            save_last=False,  # Don't save the last model
            verbose=True
        ),
        # Early stopping with configurable patience
        EarlyStopping(
            monitor=config.training.early_stopping_monitor,
            mode=config.training.early_stopping_mode,
            patience=config.training.early_stopping_patience,
            verbose=True
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    logger = TensorBoardLogger(
        save_dir=config.logging.log_dir,
        name=config.logging.experiment_name,
        default_hp_metric=False
    )
    
    pl_trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        max_epochs=config.training.max_epochs,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        precision=config.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.logging.log_every_n_steps,
    )
    
    pl_trainer.fit(trainer_module, data_module)
    
    if data_module.test_dataloader() is not None:
        pl_trainer.test(trainer_module, data_module)
    
    return model


if __name__ == "__main__":
    main()
    # Example usage: python main.py --config_file config.yaml
