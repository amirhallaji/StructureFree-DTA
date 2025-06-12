# Structure-Free Drug–Target Affinity Prediction  
_A sequence-based, language-model-driven framework for drug–target binding regression_

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Method Overview](#method-overview)
  - [Model Architecture](#model-architecture)
  - [Datasets](#datasets)
  - [Sequence Embedding](#sequence-embedding)
  - [Fusion and Regression](#fusion-and-regression)
  - [Loss Function](#loss-function)
- [Results](#results)
  - [Performance on Davis](#performance-on-davis)
  - [Performance on KIBA](#performance-on-kiba)
  - [Ablation and Negative Results](#ablation-and-negative-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Configuration](#configuration)
  - [Reproducibility](#reproducibility)
- [Figures](#figures)
- [Limitations](#limitations)
- [Citation](#citation)
- [References](#references)

---

## Introduction

Accurate **drug–target affinity (DTA) prediction** is critical for modern drug discovery, replacing costly and slow experimental screening. Most prior models require 3D structures, handcrafted features, or complex graphs, making them hard to generalize or scale.  
This project introduces a **structure-agnostic, sequence-centric approach** that combines **pretrained language models (LLMs)** for proteins and molecules with a **lightweight Residual Inception regressor** for end-to-end affinity prediction, using only SMILES and FASTA sequences.

Key features:
- **No need for 3D structures or molecular graphs**
- **Efficient fusion via Residual Inception blocks**
- **Hybrid loss: regression + ranking**
- **SOTA results on Davis and KIBA datasets**

---

## Method Overview

### Model Architecture

The framework consists of:

1. **Sequence Embedding**  
   - **Proteins:** [ESM2](https://huggingface.co/facebook/esm2_t6_8M_UR50D) (t6-8M), transformer-based, pretrained on protein FASTA.
   - **Molecules:** [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1), transformer-based, pretrained on SMILES.
   - Both encoders are fine-tuned on the DTA task.

2. **Feature Fusion**  
   - **Residual Inception Blocks:**  
     - Concatenate protein and molecule embeddings.
     - Pass through two parallel 1D-convolutional branches with residual connections (kernel sizes 1 and 3).
     - Outputs are concatenated, residual projected, and passed through ReLU.

3. **Regression Head**  
   - Flattened output → 4-layer MLP with Mish activation.
   - Output: single affinity value (e.g., Kd, Ki, or IC50).

4. **Hybrid Loss Function**  
   - **Mean Squared Error (MSE)**
   - **Cosine Similarity Loss** (to encourage correct ranking, i.e., high Concordance Index)
   - Combined as: `L = α * L_cos + (1-α) * L_mse` (α=0.5 by default)

![Model Architecture](./figures/architecture1.png)

---

### Datasets

- **Davis**  
  - 442 kinase proteins, 68 inhibitors, 30,056 affinity measurements  
  - Nearly complete interaction matrix
  - Target: Dissociation constant (Kd)

- **KIBA**  
  - 229 proteins, 2,111 ligands, 118,254 affinity samples  
  - Highly sparse (25% populated)
  - Target: Unified affinity score from Ki/Kd/IC50

| Dataset | Proteins | Ligands | Samples  | Sparsity   |
|---------|----------|---------|----------|------------|
| Davis   | 442      | 68      | 30,056   | 0%         |
| KIBA    | 229      | 2,111   | 118,254  | ~75%       |

---

### Sequence Embedding

- **Protein FASTA**:  
  - Tokenized with ESM2, capped at 1024 tokens  
  - Global mean pooling for embedding  
  - Full fine-tuning

- **SMILES**:  
  - Tokenized with ChemBERTa, capped at 128 tokens  
  - Global mean pooling  
  - Full fine-tuning

- **Empirical findings**:  
  - Freezing encoders = worse results  
  - Larger encoders = more overfitting on small/sparse datasets

---

### Fusion and Regression

- **Residual Inception Blocks:**  
  - 2 parallel Conv1d branches (kernel sizes 1 and 3) + residual
  - 2 blocks used (more → overfitting, less → underfitting)
  - Multi-scale feature fusion, low parameter count

- **Feedforward head:**  
  - Hidden layers: 1024 → 768 → 512 → 256 → 1  
  - Mish activations, Dropout, ReLU

---

### Loss Function

- **Hybrid Loss**  
  - **MSE** (numerical accuracy)  
  - **Cosine similarity** (ranking, to optimize Concordance Index)  
  - Default α = 0.5

```python
# Pseudocode for hybrid loss:
L_mse = ((y_pred - y_true) ** 2).mean()
L_cos = 1 - cosine_similarity(y_pred, y_true)
loss = 0.5 * L_cos + 0.5 * L_mse
```


## Results

All results averaged over 5-fold CV, random split, seed=0.

### Performance on Davis

| Method         | MSE   | CI    |
|----------------|-------|-------|
| **Our Method** | 0.182 | 0.920 |
| 3DProtDTA      | 0.184 | 0.917 |
| MGraphDTA      | 0.207 | 0.900 |
| BiFormerDTA    | 0.211 | 0.901 |
| SSM-DTA        | 0.219 | 0.890 |
| GraphDTA       | 0.229 | 0.893 |
| MT-DTI         | 0.245 | 0.887 |
| DeepCDA        | 0.248 | 0.891 |
| DeepDTA        | 0.262 | 0.870 |

**Best MSE and CI on Davis**  
Residual Inception blocks are sample-efficient; full fine-tuning helps the most.

---

### Performance on KIBA

| Method         | MSE   | CI    |
|----------------|-------|-------|
| MGraphDTA      | 0.128 | 0.902 |
| **Our Method** | 0.135 | 0.902 |
| 3DProtDTA      | 0.138 | 0.893 |
| GraphDTA       | 0.139 | 0.891 |
| MT-DTI         | 0.152 | 0.882 |
| SSM-DTA        | 0.154 | 0.895 |
| BiFormerDTA    | 0.174 | 0.893 |
| DeepCDA        | 0.176 | 0.889 |
| DeepDTA        | 0.196 | 0.864 |

**Matched the best CI on highly sparse KIBA**  
Training ~30% faster per epoch than GNN-based baselines.

---

### Ablation and Negative Results

- **Freezing encoders:** Underfits, CI drops <0.90
- **Cross-attention:** Overfits, lower CI
- **More than 2 Inception blocks:** Overfitting
- **Larger encoders:** No improvement, more GPU/memory
- **Self-attention after fusion:** Attention weights collapse, worse CI


## Limitations

- **Interpretability**: Sequence-based, so less interpretable than graph-based models (no explicit atom/residue attention)
- **Dataset-specific tuning**: Current hyperparameters tuned for **Davis/KIBA**; new datasets may need further tuning
- **Resource constraints**: Large-scale/large-encoder experiments limited by hardware; current setup is SOTA on Davis/KIBA size


## Citation

If you use this code or architecture in your work, please cite:

```bibtex
@article{your2025dti,
  title={Structure-Free Drug–Target Affinity Prediction Using Protein and Molecule Language Models},
  author={Your Name},
  journal={},
  year={2025}
}

