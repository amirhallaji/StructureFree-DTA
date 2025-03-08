import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index


class Trainer(pl.LightningModule):
    """PyTorch Lightning trainer for drug-target interaction models"""
    
    def __init__(self, model: nn.Module, config):
        """Initialize trainer with model and configuration"""
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.config = config
        
        # Create loss function
        self.loss_fn = self._create_loss_function()
        
        # Log model parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log("model/parameters", trainable_params)
        print(f"Model has {trainable_params:,} trainable parameters")
    
    def _create_loss_function(self):
        """Create loss function based on configuration"""
        from models import DrugTargetInteractionLoss
        return DrugTargetInteractionLoss(alpha=self.config.training.loss_alpha)
    
    def forward(self, batch):
        """Forward pass through the model"""
        return self.model(batch)
    
    def _calculate_metrics(self, predictions, targets):
        """Calculate evaluation metrics"""
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        mse = F.mse_loss(predictions, targets).item()
        
        # Calculate R2 score if we have more than 1 sample
        if len(predictions) > 1:
            r2 = r2_score(targets_np, predictions_np)
        else:
            r2 = 0.0
        
        try:
            unique_targets = set(targets_np.flatten())
            if len(unique_targets) > 1:
                ci = concordance_index(targets_np, predictions_np)
            else:
                # No different target values, can't calculate CI
                ci = 0.5
                print(f"Warning: All target values are the same ({unique_targets}). Cannot calculate concordance index.")
        except Exception as e:
            # Handle any other errors in concordance index calculation
            ci = 0.5
            print(f"Warning: Error calculating concordance index: {str(e)}. Using default value of 0.5.")
        
        return {"mse": mse, "r2": r2, "ci": ci}
    
    def training_step(self, batch, batch_idx):
        """Execute one training step"""
        labels = batch.pop("labels")
        outputs = self(batch)
        loss = self.loss_fn(outputs, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Execute one validation step"""
        labels = batch.pop("labels")
        outputs = self(batch)
        loss = self.loss_fn(outputs, labels)
        metrics = self._calculate_metrics(outputs, labels)
        
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mse", metrics["mse"], on_epoch=True)
        self.log("val/r2", metrics["r2"], on_epoch=True)
        self.log("val/ci", metrics["ci"], on_epoch=True)
        
        return {"loss": loss, "preds": outputs, "targets": labels}
    
    def test_step(self, batch, batch_idx):
        """Execute one test step"""
        labels = batch.pop("labels")
        outputs = self(batch)
        loss = self.loss_fn(outputs, labels)
        metrics = self._calculate_metrics(outputs, labels)
        
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/mse", metrics["mse"], on_epoch=True)
        self.log("test/r2", metrics["r2"], on_epoch=True)
        self.log("test/ci", metrics["ci"], on_epoch=True)
        
        return {"loss": loss, "preds": outputs, "targets": labels}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Execute one prediction step"""
        labels = batch.pop("labels") if "labels" in batch else None
        outputs = self(batch)
        return {"preds": outputs, "targets": labels}
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.training.lr_scheduler_factor,
                patience=self.config.training.lr_scheduler_patience,
                min_lr=self.config.training.lr_scheduler_min_lr,
                verbose=True
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 