import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import KFold, train_test_split
from typing import Dict, List, Tuple, Optional, Union


class DrugTargetDataset(Dataset):
    def __init__(self, molecules, proteins, labels):
        self.molecules = molecules
        self.proteins = proteins
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "molecule": self.molecules[idx],
            "protein": self.proteins[idx],
            "label": self.labels[idx]
        }


def collate_fn(batch, molecule_tokenizer, protein_tokenizer, max_molecule_length=128, max_protein_length=1024):
    molecules = [item["molecule"] for item in batch]
    proteins = [item["protein"] for item in batch]
    labels = [item["label"] for item in batch]
    
    molecule_encodings = molecule_tokenizer(
        molecules,
        padding="max_length",
        truncation=True,
        max_length=max_molecule_length,
        return_tensors="pt"
    )
    
    protein_encodings = protein_tokenizer(
        proteins,
        padding="max_length",
        truncation=True,
        max_length=max_protein_length,
        return_tensors="pt"
    )
    
    batch_dict = {
        "molecule_input_ids": molecule_encodings.input_ids,
        "molecule_attention_mask": molecule_encodings.attention_mask,
        "protein_input_ids": protein_encodings.input_ids,
        "protein_attention_mask": protein_encodings.attention_mask,
        "labels": torch.tensor(labels, dtype=torch.float)
    }
    
    return batch_dict


class CollateManager:
    def __init__(self, molecule_tokenizer, protein_tokenizer, max_molecule_length, max_protein_length):
        self.molecule_tokenizer = molecule_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_molecule_length = max_molecule_length
        self.max_protein_length = max_protein_length
    
    def __call__(self, batch):
        return collate_fn(
            batch,
            self.molecule_tokenizer,
            self.protein_tokenizer,
            self.max_molecule_length,
            self.max_protein_length
        )


class DrugTargetDataModule:
    def __init__(
        self,
        data_path: str,
        n_folds: int = 5,
        val_size: float = 0.1,
        random_state: int = 0,
        molecule_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        protein_model_name: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 32,
        num_workers: int = 4,
        max_molecule_length: int = 128,
        max_protein_length: int = 1024,
    ):
        self.data_path = data_path
        self.n_folds = n_folds
        self.val_size = val_size
        self.random_state = random_state
        self.molecule_model_name = molecule_model_name
        self.protein_model_name = protein_model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_molecule_length = max_molecule_length
        self.max_protein_length = max_protein_length
        
        self.molecule_tokenizer = AutoTokenizer.from_pretrained(molecule_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
        
        self.collate_manager = CollateManager(
            self.molecule_tokenizer,
            self.protein_tokenizer,
            self.max_molecule_length,
            self.max_protein_length
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.molecules = None
        self.proteins = None
        self.labels = None
        self.fold_indices = None
        
        self._load_all_data()
        self._create_folds()
    
    def _load_data(self, file_path: str) -> Tuple[List[str], List[str], List[float]]:
        df = pd.read_csv(file_path)
        molecules = df["Molecule Sequence"].tolist()
        proteins = df["Protein Sequence"].tolist()
        labels = df["Binding Affinity"].astype(float).tolist()
        return molecules, proteins, labels
    
    def _load_all_data(self):
        self.molecules, self.proteins, self.labels = self._load_data(self.data_path)
    
    def _create_folds(self):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        indices = np.arange(len(self.labels))
        self.fold_indices = list(kf.split(indices))
    
    def setup_fold(self, fold: int):
        if fold < 0 or fold >= self.n_folds:
            raise ValueError(f"Fold must be between 0 and {self.n_folds-1}")
        
        train_val_idx, test_idx = self.fold_indices[fold]
        
        train_val_molecules = [self.molecules[i] for i in train_val_idx]
        train_val_proteins = [self.proteins[i] for i in train_val_idx]
        train_val_labels = [self.labels[i] for i in train_val_idx]
        
        test_molecules = [self.molecules[i] for i in test_idx]
        test_proteins = [self.proteins[i] for i in test_idx]
        test_labels = [self.labels[i] for i in test_idx]
        
        self.test_dataset = DrugTargetDataset(test_molecules, test_proteins, test_labels)
        
        if self.val_size > 0:
            train_molecules, val_molecules, train_proteins, val_proteins, train_labels, val_labels = train_test_split(
                train_val_molecules, train_val_proteins, train_val_labels,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=None
            )
            self.train_dataset = DrugTargetDataset(train_molecules, train_proteins, train_labels)
            self.val_dataset = DrugTargetDataset(val_molecules, val_proteins, val_labels)
        else:
            self.train_dataset = DrugTargetDataset(train_val_molecules, train_val_proteins, train_val_labels)
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_manager,
            pin_memory=True
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_manager,
                pin_memory=True
            )
        return None
    
    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_manager,
                pin_memory=True
            )
        return None