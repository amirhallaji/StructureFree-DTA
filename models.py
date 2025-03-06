import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ResidualInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], dropout=0.05):
        super(ResidualInceptionBlock, self).__init__()
        self.conv_branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                Mish(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                Mish(),
                nn.Dropout(dropout)
            )
            self.conv_branches.append(branch)
        
        # Residual connection if dimensions don't match
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * len(kernel_sizes), kernel_size=1),
            nn.BatchNorm1d(out_channels * len(kernel_sizes))
        ) if in_channels != out_channels * len(kernel_sizes) else nn.Identity()
    
    def forward(self, x):
        branch_outputs = []
        for branch in self.conv_branches:
            branch_outputs.append(branch(x))
        
        branch_output = torch.cat(branch_outputs, dim=1)
        residual_x = self.residual(x)
        
        return F.relu(branch_output + residual_x)


class AffinityPredictor(nn.Module):
    def __init__(self, 
                 protein_model_name="facebook/esm2_t6_8M_UR50D", 
                 molecule_model_name="DeepChem/ChemBERTa-77M-MLM",
                 hidden_sizes=[1024, 768, 512, 256, 1], 
                 inception_out_channels=256,
                 dropout=0.05):
        super(AffinityPredictor, self).__init__()
        
        # Load pre-trained models
        self.protein_model = AutoModel.from_pretrained(protein_model_name)
        self.molecule_model = AutoModel.from_pretrained(molecule_model_name)
        
        # Get embedding dimensions
        self.protein_dim = self.protein_model.config.hidden_size
        self.molecule_dim = self.molecule_model.config.hidden_size
        
        # Inception blocks for feature extraction
        self.protein_inception = ResidualInceptionBlock(
            self.protein_dim, inception_out_channels, dropout=dropout
        )
        self.molecule_inception = ResidualInceptionBlock(
            self.molecule_dim, inception_out_channels, dropout=dropout
        )
        
        # Calculate input size for the prediction head
        inception_output_size = inception_out_channels * 2 * 2  # 2 branches, 2 modalities
        
        # Prediction head
        layers = []
        input_size = inception_output_size
        
        for i, output_size in enumerate(hidden_sizes):
            if i == len(hidden_sizes) - 1:  # Last layer
                layers.append(nn.Linear(input_size, output_size))
            else:
                layers.extend([
                    nn.Linear(input_size, output_size),
                    nn.BatchNorm1d(output_size),
                    Mish(),
                    nn.Dropout(dropout)
                ])
                input_size = output_size
        
        self.prediction_head = nn.Sequential(*layers)
    
    def forward(self, batch):
        # Unpack batch
        molecule_input_ids = batch["molecule_input_ids"]
        molecule_attention_mask = batch["molecule_attention_mask"]
        protein_input_ids = batch["protein_input_ids"]
        protein_attention_mask = batch["protein_attention_mask"]
        
        # Get embeddings from pre-trained models
        molecule_outputs = self.molecule_model(
            input_ids=molecule_input_ids,
            attention_mask=molecule_attention_mask,
            return_dict=True
        )
        protein_outputs = self.protein_model(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            return_dict=True
        )
        
        # Extract sequence representations
        molecule_embeddings = molecule_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        protein_embeddings = protein_outputs.last_hidden_state    # [batch_size, seq_len, hidden_size]
        
        # Transpose for Conv1D: [batch_size, hidden_size, seq_len]
        molecule_embeddings = molecule_embeddings.transpose(1, 2)
        protein_embeddings = protein_embeddings.transpose(1, 2)
        
        # Apply inception blocks
        molecule_features = self.molecule_inception(molecule_embeddings)
        protein_features = self.protein_inception(protein_embeddings)
        
        # Global pooling
        molecule_features = torch.cat([
            F.adaptive_max_pool1d(molecule_features, 1).squeeze(-1),
            F.adaptive_avg_pool1d(molecule_features, 1).squeeze(-1)
        ], dim=1)
        
        protein_features = torch.cat([
            F.adaptive_max_pool1d(protein_features, 1).squeeze(-1),
            F.adaptive_avg_pool1d(protein_features, 1).squeeze(-1)
        ], dim=1)
        
        # Concatenate features
        combined_features = torch.cat([molecule_features, protein_features], dim=1)
        
        # Predict affinity
        affinity = self.prediction_head(combined_features)
        
        return affinity.squeeze(-1)


class DrugTargetInteractionLoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super(DrugTargetInteractionLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, interaction_score):
        # MSE Loss
        mse_loss = self.mse(pred, interaction_score)
        
        # Ranking Loss (Pairwise)
        batch_size = pred.size(0)
        if batch_size <= 1:
            return mse_loss
        
        # Create all possible pairs
        pred_i = pred.unsqueeze(1).repeat(1, batch_size)
        pred_j = pred.unsqueeze(0).repeat(batch_size, 1)
        
        target_i = interaction_score.unsqueeze(1).repeat(1, batch_size)
        target_j = interaction_score.unsqueeze(0).repeat(batch_size, 1)
        
        # Calculate ranking loss only for pairs with different targets
        mask = (target_i != target_j).float()
        
        # Concordance loss: if target_i > target_j then pred_i should be > pred_j
        concordance = torch.sign(target_i - target_j) * torch.sign(pred_i - pred_j)
        concordance = (1.0 - concordance) / 2.0  # Convert to loss (0 if concordant, 1 if discordant)
        
        ranking_loss = (concordance * mask).sum() / (mask.sum() + 1e-8)
        
        # Combine losses
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ranking_loss
        
        return total_loss


def count_model_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 