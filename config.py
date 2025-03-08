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
    batch_size: int = 32
    num_workers: int = 4
    max_molecule_length: int = 128
    max_protein_length: int = 1024


@dataclass
class TrainingConfig:
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    loss_alpha: float = 0.5
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"  # Options: "32", "16-mixed", "bf16-mixed"
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-7


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    experiment_name: str = "drug_target_interaction"
    log_every_n_steps: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    accelerator: str = "auto"  # Options: "auto", "cpu", "gpu", "tpu", "mps"
    devices: Union[int, str, List[int]] = "auto"  # Number of devices or "auto"
    strategy: str = "auto"  # Options: "auto", "ddp", "deepspeed", "fsdp"
    
    def __post_init__(self):
        import torch
        if self.accelerator == "gpu" and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead")
            self.accelerator = "cpu"

