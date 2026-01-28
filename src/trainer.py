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
        fold: int = 0,
    ):
        self.config = config
        self.device = device
        self.rank = 0
        self.world_size = 1
        self.is_distributed = config.distributed.distributed_backend != "none"
        self.fold = fold
        
        if self.is_distributed:
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.is_main_process = self.rank == 0
        else:
            self.is_main_process = True
        
        self.model = self._setup_model(model)
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metric_tracker = metric_tracker
        
        self.use_mixed_precision = config.training.mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        self.epochs = config.training.epochs
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.clip_grad_norm = config.training.clip_grad_norm
        self.early_stopping_patience = config.training.early_stopping_patience
        
        self.log_interval = config.logging.log_interval
        self.save_dir = Path(config.logging.save_dir) / f"fold_{fold}"
        
        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
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
        if not self.is_main_process:
            return
        
        checkpoint = {
            "epoch": epoch,
            "fold": self.fold,
            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metrics": self.metric_tracker.best_metrics,
        }
        
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        latest_path = self.save_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.epochs}"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop("labels")
            
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs.view(-1), labels)
                    loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs.view(-1), labels)
                loss = loss / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            batch_size = labels.size(0)
            total_loss += loss.item() * self.grad_accum_steps * batch_size
            total_samples += batch_size
            
            if self.is_main_process and batch_idx % self.log_interval == 0:
                step_loss = loss.item() * self.grad_accum_steps
                self.metric_tracker.update_train_metrics(step_loss, step=True)
                if batch_idx % (self.log_interval * 10) == 0:
                    print(f"Batch {batch_idx}/{len(self.train_dataloader)} | Loss: {step_loss:.4f}")
        
        avg_loss = total_loss / total_samples
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return avg_loss
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = batch.pop("labels")
                
                outputs = self.model(batch)
                loss = self.loss_fn(outputs.view(-1), labels)
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                all_preds.append(outputs.detach())
                all_labels.append(labels.detach())
        
        avg_loss = total_loss / total_samples
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        val_metrics = self.metric_tracker.update_val_metrics(avg_loss, all_preds, all_labels, epoch)
        
        return val_metrics
    
    def train(self) -> Tuple[List[float], List[float]]:
        if self.is_main_process:
            print("Logging model architecture and hyperparameters...")
            self.metric_tracker.log_model_architecture(self.model)
            self.metric_tracker.log_hyperparameters(self.config)
            print("Model architecture and hyperparameters logged successfully")
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        print(f"Starting training for {self.epochs} epochs")
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            
            val_metrics = self._validate_epoch(epoch)
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            if self.is_main_process:
                self.metric_tracker.update_train_metrics(train_loss)
                self.metric_tracker.log_epoch(epoch, train_loss, val_metrics, current_lr)
            
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
                
                if patience_counter >= self.early_stopping_patience:
                    if self.is_main_process:
                        print("Early stopping triggered.")
                    break
            
            self._save_checkpoint(epoch)
        
        final_path = self.save_dir / "final_model.pt"
        if self.is_main_process:
            torch.save(
                self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                final_path
            )
            print("Final model saved.")
            
            self.metric_tracker.log_summary()
        
        return self.metric_tracker.metrics["train"]["loss"], self.metric_tracker.metrics["val"]["loss"]