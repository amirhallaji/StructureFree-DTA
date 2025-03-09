from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class ModelConfig:
    protein_model_name: str = "facebook/esm2_t6_8M_UR50D"
    molecule_model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    hidden_sizes: List[int] = field(default_factory=lambda: [1024, 768, 512, 256, 1])
    inception_out_channels: int = 256
    dropout: float = 0.05


@dataclass
class DataConfig:
    train_data_path: str = "data/train.csv"
    val_data_path: str = "data/val.csv"
    test_data_path: Optional[str] = None
    batch_size: int = 2
    num_workers: int = 1
    max_molecule_length: int = 32
    max_protein_length: int = 128


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-7
    loss_alpha: float = 0.5
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    clip_grad_norm: Optional[float] = 1.0


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    experiment_name: str = "drug_target_interaction"
    log_interval: int = 10


@dataclass
class DistributedConfig:
    distributed_backend: str = "none"  # Options: none, ddp, fsdp
    find_unused_parameters: bool = False
    fsdp_config: Optional[Dict[str, Any]] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    seed: int = 42
    device: str = "cuda"
    
    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead")
            self.device = "cpu" 

