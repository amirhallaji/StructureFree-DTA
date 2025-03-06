import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
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
    
    # Tokenize molecules
    molecule_encodings = molecule_tokenizer(
        molecules,
        padding="max_length",
        truncation=True,
        max_length=max_molecule_length,
        return_tensors="pt"
    )
    
    # Tokenize proteins
    protein_encodings = protein_tokenizer(
        proteins,
        padding="max_length",
        truncation=True,
        max_length=max_protein_length,
        return_tensors="pt"
    )
    
    # Create batch dictionary
    batch_dict = {
        "molecule_input_ids": molecule_encodings.input_ids,
        "molecule_attention_mask": molecule_encodings.attention_mask,
        "protein_input_ids": protein_encodings.input_ids,
        "protein_attention_mask": protein_encodings.attention_mask,
        "labels": torch.tensor(labels, dtype=torch.float)
    }
    
    return batch_dict


class DrugTargetDataModule:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: Optional[str] = None,
        molecule_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        protein_model_name: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 32,
        num_workers: int = 4,
        max_molecule_length: int = 128,
        max_protein_length: int = 1024,
    ):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.molecule_model_name = molecule_model_name
        self.protein_model_name = protein_model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_molecule_length = max_molecule_length
        self.max_protein_length = max_protein_length
        
        # Load tokenizers
        self.molecule_tokenizer = AutoTokenizer.from_pretrained(molecule_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Load data
        self._setup()
    
    def _load_data(self, file_path: str) -> Tuple[List[str], List[str], List[float]]:
        """Load data from CSV file"""
        df = pd.read_csv(file_path)
        
        # Assuming the CSV has columns: molecule_smiles, protein_sequence, binding_affinity
        molecules = df["molecule_smiles"].tolist()
        proteins = df["protein_sequence"].tolist()
        labels = df["binding_affinity"].astype(float).tolist()
        
        return molecules, proteins, labels
    
    def _setup(self):
        """Setup datasets"""
        # Load training data
        train_molecules, train_proteins, train_labels = self._load_data(self.train_data_path)
        self.train_dataset = DrugTargetDataset(train_molecules, train_proteins, train_labels)
        
        # Load validation data
        val_molecules, val_proteins, val_labels = self._load_data(self.val_data_path)
        self.val_dataset = DrugTargetDataset(val_molecules, val_proteins, val_labels)
        
        # Load test data if available
        if self.test_data_path:
            test_molecules, test_proteins, test_labels = self._load_data(self.test_data_path)
            self.test_dataset = DrugTargetDataset(test_molecules, test_proteins, test_labels)
    
    def get_collate_fn(self):
        """Return the collate function with tokenizers"""
        return lambda batch: collate_fn(
            batch,
            self.molecule_tokenizer,
            self.protein_tokenizer,
            self.max_molecule_length,
            self.max_protein_length
        )
    
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.get_collate_fn(),
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.get_collate_fn(),
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Return the test dataloader if available"""
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.get_collate_fn(),
                pin_memory=True
            )
        return None 