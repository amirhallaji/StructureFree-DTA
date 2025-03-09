import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path

from tqdm import tqdm

from utils import MetricTracker


class Trainer:
    """
    A modular trainer class for drug-target interaction models.
    Supports various distributed training methods (DDP, FSDP) and mixed precision.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Any,
        metric_tracker: MetricTracker,
    ):
        self.config = config
        self.device = device
        self.rank = 0
        self.world_size = 1
        self.is_distributed = config.distributed.distributed_backend != "none"
        
        # Set up distributed training if needed
        if self.is_distributed:
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.is_main_process = self.rank == 0
        else:
            self.is_main_process = True
        
        # Set up model based on distributed backend
        self.model = self._setup_model(model)
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metric_tracker = metric_tracker
        
        # Mixed precision setup
        self.use_mixed_precision = config.training.mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training parameters
        self.epochs = config.training.epochs
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.clip_grad_norm = config.training.clip_grad_norm
        self.early_stopping_patience = config.training.early_stopping_patience
        
        # Logging parameters
        self.log_interval = config.logging.log_interval
        self.save_dir = Path(config.logging.save_dir)
        
        # Create save directory if it doesn't exist
        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Set up model based on distributed backend"""
        model = model.to(self.device)
        
        if not self.is_distributed:
            return model
        
        if self.config.distributed.distributed_backend == "ddp":
            return DDP(
                model,
                device_ids=[self.rank] if self.device.type == "cuda" else None,
                output_device=self.rank if self.device.type == "cuda" else None,
                find_unused_parameters=self.config.distributed.find_unused_parameters
            )
        elif self.config.distributed.distributed_backend == "fsdp":
            fsdp_config = self.config.distributed.fsdp_config or {}
            return FSDP(model, **fsdp_config)
        else:
            return model
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metrics": self.metric_tracker.best_metrics,
        }
        
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint (overwrite)
        latest_path = self.save_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Reset gradients for first step
        self.optimizer.zero_grad()
        
        # use tqdm to iterate over the train_dataloader
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels")
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs.view(-1), labels)
                    loss = loss / self.grad_accum_steps  # Normalize loss for gradient accumulation
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    # Gradient clipping
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                outputs = self.model(batch)
                loss = self.loss_fn(outputs.view(-1), labels)
                loss = loss / self.grad_accum_steps  # Normalize loss for gradient accumulation
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    # Gradient clipping
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * self.grad_accum_steps * batch_size  # Denormalize loss
            total_samples += batch_size
            
            # Log step metrics only at specified intervals
            if self.is_main_process and batch_idx % self.log_interval == 0:
                step_loss = loss.item() * self.grad_accum_steps  # Denormalize loss
                self.metric_tracker.update_train_metrics(step_loss, step=True)
                # Keep the terminal output minimal
                if batch_idx % (self.log_interval * 10) == 0:  # Show less frequent terminal output
                    print(f"Batch {batch_idx}/{len(self.train_dataloader)} | Loss: {step_loss:.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        return avg_loss
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = batch.pop("labels")
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.loss_fn(outputs.view(-1), labels)
                
                # Update metrics
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Store predictions and labels for metrics calculation
                all_preds.append(outputs.detach())
                all_labels.append(labels.detach())
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Update validation metrics
        val_metrics = self.metric_tracker.update_val_metrics(avg_loss, all_preds, all_labels, epoch)
        
        return val_metrics
    
    def train(self) -> Tuple[List[float], List[float]]:
        """Train the model for the specified number of epochs"""
        # Log model architecture and hyperparameters ONCE at the beginning of training
        if self.is_main_process:
            print("Logging model architecture and hyperparameters...")
            self.metric_tracker.log_model_architecture(self.model)
            self.metric_tracker.log_hyperparameters(self.config)
            print("Model architecture and hyperparameters logged successfully")
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        print(f"Starting training for {self.epochs} epochs")
        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self._train_epoch(epoch)
            
            # Validate
            val_metrics = self._validate_epoch(epoch)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log epoch metrics
            if self.is_main_process:
                self.metric_tracker.update_train_metrics(train_loss)
                self.metric_tracker.log_epoch(epoch, train_loss, val_metrics, current_lr)
            
            # Check for improvement
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                if self.is_main_process:
                    print(f"Best model saved at epoch {epoch}")
            else:
                patience_counter += 1
                if self.is_main_process:
                    print(f"Did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    if self.is_main_process:
                        print("Early stopping triggered.")
                    break
            
            # Save regular checkpoint
            self._save_checkpoint(epoch)
        
        # Save final model
        final_path = self.save_dir / "final_model.pt"
        if self.is_main_process:
            torch.save(
                self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                final_path
            )
            print("Final model saved.")
            
            # Log summary
            self.metric_tracker.log_summary()
        
        return self.metric_tracker.metrics["train"]["loss"], self.metric_tracker.metrics["val"]["loss"] 