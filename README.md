# Structure-Free Drug–Target Affinity Prediction with Protein and Molecule Language Models

> **State-of-the-art sequence-centric DTA regression using ChemBERTa, ESM2, and a novel Residual Inception regressor.**  
> Davis: **MSE = 0.182, CI = 0.920** · KIBA: **CI = 0.902**

---

## Table of Contents

- [Overview](#overview)
- [Motivation & Background](#motivation--background)
- [Key Contributions](#key-contributions)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
    - [Data & Preprocessing](#data--preprocessing)
    - [Sequence Embedding: ChemBERTa & ESM2](#sequence-embedding-chemberta--esm2)
    - [Residual Inception Fusion](#residual-inception-fusion)
    - [Prediction Head & Hybrid Loss](#prediction-head--hybrid-loss)
    - [Training Procedure](#training-procedure)
    - [Architecture Diagram](#architecture-diagram)
- [Experiments](#experiments)
    - [Datasets](#datasets)
    - [Baselines](#baselines)
    - [Results](#results)
    - [Ablation & Negative Results](#ablation--negative-results)
- [Limitations](#limitations)
- [How to Run](#how-to-run)
    - [Setup](#setup)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Docker Support](#docker-support)
- [References](#references)
- [Citation](#citation)

---

## Overview

This repository presents a **structure-free, sequence-centric framework** for drug–target affinity (DTA) prediction, leveraging:

- **Protein language models** (ESM2)
- **Molecule language models** (ChemBERTa)
- A lightweight **Residual Inception regressor**

Our method operates **entirely on SMILES and FASTA** sequences—**no 3D structures, no graphs**—and achieves SOTA on Davis and KIBA benchmarks, matching or surpassing the best deep graph/transformer approaches with less complexity.

---

## Motivation & Background

Predicting DTA is essential for modern drug discovery, yet:

- **Traditional methods** (docking, 3D-based) require expensive, often unavailable structural data.
- **Deep learning approaches** (CNNs, GNNs, Transformers) brought progress but can be *data-hungry*, complex, and overfit on sparse datasets.

Recent **large language models (LLMs)** pre-trained on biomolecular sequences have shown that purely sequence-based approaches can compete with or outperform structure-based ones—if embeddings are fused effectively.

---

## Key Contributions

- **Fully Sequence-Based**: No explicit structural/graph input—only SMILES and FASTA, encoded by ChemBERTa & ESM2.
- **Novel Residual Inception Fusion**: Custom, efficient regressor fusing embeddings via parallel 1D convolutions + residual connections (multi-scale, regularized).
- **Hybrid Loss**: Joint regression (MSE) + ranking (cosine similarity) loss for accurate values **and** correct order.
- **Rigorous Evaluation**: SOTA on **Davis** (MSE 0.182, CI 0.920) and **KIBA** (CI 0.902), outperforming or matching best GNNs/transformers.
- **Extensive Ablations**: Justify every design choice; report and discuss negative results for full transparency.

---

## Project Structure

├── cfg/               # YAML/JSON configuration files (model, train, etc.)
├── src/               # All main source code (modules, trainers, utils)
├── data/              # Data loaders, scripts, processed datasets (not included)
├── requirements.txt   # Python dependencies
├── train.sh           # Shell script for running training (CLI entry)
├── Dockerfile         # Containerization (optional)
├── build_push.sh      # Docker build/push script (infra, not user-facing)
├── README.md          # This file
├── .dockerignore
└── .gitignore
