import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], dropout=0.05):
        super(ResidualInceptionBlock, self).__init__()

        self.out_channels = out_channels
        num_branches = len(kernel_sizes)
        branch_out_channels = out_channels // num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, branch_out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(branch_out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])

        self.residual_adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        residual = self.residual_adjust(x)
        output = self.relu(concatenated + residual)
        return output

class AffinityPredictor(nn.Module):
    def __init__(self, 
                 protein_model_name="facebook/esm2_t6_8M_UR50D", 
                 molecule_model_name="DeepChem/ChemBERTa-77M-MLM",
                 hidden_sizes=[1024,768,512,256,1], 
                 inception_out_channels=256,
                 dropout=0.05):
        super(AffinityPredictor, self).__init__()

        self.protein_model = AutoModel.from_pretrained(protein_model_name)
        self.molecule_model = AutoModel.from_pretrained(molecule_model_name)

        self.protein_model.config.gradient_checkpointing = True
        self.protein_model.gradient_checkpointing_enable()

        self.molecule_model.config.gradient_checkpointing = True
        self.molecule_model.gradient_checkpointing_enable()

        prot_embedding_dim = self.protein_model.config.hidden_size
        mol_embedding_dim = self.molecule_model.config.hidden_size
        combined_dim = prot_embedding_dim + mol_embedding_dim

        self.inc1 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        self.inc2 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)

        layers = []
        input_dim = combined_dim  # After Inception block
        for output_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, output_dim))
            if output_dim != 1:
                layers.append(Mish())
            input_dim = output_dim
        self.regressor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        protein_input = {
            "input_ids": batch["protein_input_ids"],
            "attention_mask": batch["protein_attention_mask"]
        }
        molecule_input = {
            "input_ids": batch["molecule_input_ids"],
            "attention_mask": batch["molecule_attention_mask"]
        }
        protein_embedding = self.protein_model(**protein_input).last_hidden_state.mean(dim=1)  # (batch_size, hidden_dim)
        molecule_embedding = self.molecule_model(**molecule_input).last_hidden_state.mean(dim=1)  # (batch_size, hidden_dim)
        combined_features = torch.cat((protein_embedding, molecule_embedding), dim=1).unsqueeze(2)  # (batch_size, combined_dim, 1)
        combined_features = self.inc1(combined_features)  # (batch_size, combined_dim)
        combined_features = self.inc2(combined_features)
        combined_features = combined_features.squeeze(2)
        output = self.regressor(self.dropout(combined_features))  # (batch_size, 1)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class DrugTargetInteractionLoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super(DrugTargetInteractionLoss, self).__init__()
        self.alpha = alpha  # Weighting factor between cosine similarity loss and MSE loss
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, pred , interaction_score):
        """
        Computes the combined loss for drug-target interaction.
        :param drug_emb: Tensor of shape (batch_size, feature_dim)
        :param target_emb: Tensor of shape (batch_size, feature_dim)
        :param interaction_score: Tensor of shape (batch_size,) with interaction scores (continuous values)
        """
        cosine_sim = F.cosine_similarity(pred, interaction_score, dim=-1)
        cosine_loss = 1 - cosine_sim  # Encourage similarity between interacting pairs
        
        mse_loss = self.mse_loss(pred, interaction_score)
        total_loss = self.alpha * cosine_loss + (1 - self.alpha) * mse_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        return total_loss


def count_model_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 