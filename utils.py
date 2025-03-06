import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score


class MetricTracker:
    """Class to track metrics during training and validation"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.metrics = {
            "train": {"loss": [], "step_loss": []},
            "val": {"loss": [], "r2": [], "mse": [], "ci": []}
        }
        self.best_metrics = {
            "val_loss": float("inf"),
            "val_r2": float("-inf"),
            "val_ci": float("-inf"),
            "epoch": 0
        }
        self.start_time = time.time()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create experiment directory
        self.experiment_dir = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file_path = self.experiment_dir / "training_log.txt"
        with open(self.log_file_path, "w") as f:
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters to a file"""
        config_path = self.experiment_dir / "config.json"
        
        # Convert config to a JSON-serializable format
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, "__dict__"):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        
        # Also log to the training log
        with open(self.log_file_path, "a") as f:
            f.write("Hyperparameters:\n")
            f.write(json.dumps(config_dict, indent=4))
            f.write("\n\n")
    
    def log_model_architecture(self, model: torch.nn.Module):
        """Log model architecture to a file"""
        model_path = self.experiment_dir / "model_architecture.txt"
        with open(model_path, "w") as f:
            f.write(str(model))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log to training log
        with open(self.log_file_path, "a") as f:
            f.write(f"Model Architecture:\n{str(model)}\n\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n\n")
    
    def update_train_metrics(self, loss: float, step: bool = False):
        """Update training metrics"""
        if step:
            self.metrics["train"]["step_loss"].append(loss)
        else:
            self.metrics["train"]["loss"].append(loss)
    
    def update_val_metrics(self, loss: float, predictions: torch.Tensor, targets: torch.Tensor, epoch: int):
        """Update validation metrics"""
        # Convert tensors to numpy arrays
        preds = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Calculate metrics
        mse = mean_squared_error(targets, preds)
        r2 = r2_score(targets, preds)
        ci = self.concordance_index(targets, preds)
        
        # Update metrics
        self.metrics["val"]["loss"].append(loss)
        self.metrics["val"]["mse"].append(mse)
        self.metrics["val"]["r2"].append(r2)
        self.metrics["val"]["ci"].append(ci)
        
        # Update best metrics
        if loss < self.best_metrics["val_loss"]:
            self.best_metrics["val_loss"] = loss
            self.best_metrics["val_r2"] = r2
            self.best_metrics["val_ci"] = ci
            self.best_metrics["epoch"] = epoch
            
            # Log best metrics
            with open(self.log_file_path, "a") as f:
                f.write(f"New best model at epoch {epoch}:\n")
                f.write(f"  Val Loss: {loss:.4f}\n")
                f.write(f"  Val R2: {r2:.4f}\n")
                f.write(f"  Val CI: {ci:.4f}\n\n")
        
        return {
            "loss": loss,
            "mse": mse,
            "r2": r2,
            "ci": ci
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_metrics: Dict[str, float], lr: float):
        """Log epoch metrics"""
        elapsed_time = time.time() - self.start_time
        
        log_str = (
            f"Epoch {epoch} | "
            f"Time: {elapsed_time:.2f}s | "
            f"LR: {lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f} | "
            f"Val CI: {val_metrics['ci']:.4f}"
        )
        
        print(log_str)
        
        with open(self.log_file_path, "a") as f:
            f.write(log_str + "\n")
    
    def plot_metrics(self):
        """Plot training and validation metrics"""
        # Create figures directory
        figures_dir = self.experiment_dir / "figures"
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["train"]["loss"], label="Train Loss")
        plt.plot(self.metrics["val"]["loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(figures_dir / "loss.png")
        plt.close()
        
        # Plot validation metrics
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["val"]["r2"], label="R2 Score")
        plt.plot(self.metrics["val"]["ci"], label="CI Score")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(True)
        plt.savefig(figures_dir / "val_metrics.png")
        plt.close()
        
        # Plot step losses if available
        if len(self.metrics["train"]["step_loss"]) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics["train"]["step_loss"])
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Step Loss")
            plt.grid(True)
            plt.savefig(figures_dir / "step_loss.png")
            plt.close()
    
    def save_metrics(self):
        """Save metrics to a JSON file"""
        metrics_path = self.experiment_dir / "metrics.json"
        
        metrics_dict = {
            "train": self.metrics["train"],
            "val": self.metrics["val"],
            "best": self.best_metrics
        }
        
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    
    def log_summary(self):
        """Log summary of training"""
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        summary = (
            f"\nTraining Summary:\n"
            f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"Best epoch: {self.best_metrics['epoch']}\n"
            f"Best validation loss: {self.best_metrics['val_loss']:.4f}\n"
            f"Best validation R2: {self.best_metrics['val_r2']:.4f}\n"
            f"Best validation CI: {self.best_metrics['val_ci']:.4f}\n"
        )
        
        print(summary)
        
        with open(self.log_file_path, "a") as f:
            f.write(summary)
        
        # Save metrics and plots
        self.save_metrics()
        self.plot_metrics()
    
    @staticmethod
    def concordance_index(y_true, y_pred):
        """Calculate concordance index (CI)"""
        n = len(y_true)
        pairs = 0
        concordant = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if y_true[i] != y_true[j]:  # Only consider pairs with different true values
                    pairs += 1
                    if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                       (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                        concordant += 1
        
        return concordant / pairs if pairs > 0 else 0.5


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda"):
    """Get the device to use"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training"""
    torch.distributed.destroy_process_group() 