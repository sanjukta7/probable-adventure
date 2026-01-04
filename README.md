# MergeDNA

Implementing an adaptive DNA tokenization and pre-training framework based on token merging.

This is inspired by the following research papers:
- [MergeDNA](https://arxiv.org/pdf/2511.14806) - Adaptive DNA tokenization via token merging
- [Token Merging (ToMe)](https://arxiv.org/pdf/2210.09461) - Token merging for efficient vision transformers

## Overview

MergeDNA learns to compress DNA sequences by merging similar adjacent tokens, enabling:
- **Efficient representation** - Reduces sequence length while preserving biological information
- **Adaptive tokenization** - Learns which regions are important vs. repetitive
- **Anomaly detection** - Uses reconstruction error to detect unusual sequences

## Repository Structure

```
probable-adventure/
├── mergedna/                    # Core library
│   ├── backbone.py              # MergeDNA model architecture (encoder, decoder, attention)
│   ├── merging.py               # Token merging layer (bipartite matching, source matrix)
│   ├── dataloader.py            # DNA sequence data loading utilities
│   └── utils.py                 # Transformer blocks, positional embeddings
│
├── notebooks/                   # Jupyter notebooks
│   ├── _01_promoter.ipynb       # Promoter prediction task setup
│   └── _02_pretrain.ipynb       # Full pre-training pipeline with visualization
│
├── scripts/                     # Standalone scripts
│   ├── download_data.py         # Download genomic benchmark datasets
│   ├── pretrain.py              # Pre-training script
│   └── inference.py             # Inference utilities
│
├── data/                        # Datasets (gitignored)
│   └── human_nontata_promoters/ # Promoter classification dataset
│       ├── train/
│       │   ├── positive/        # Promoter sequences
│       │   └── negative/        # Non-promoter sequences
│       └── test/
│
├── checkpoints/                 # Saved models and plots
│   ├── mergedna_pretrain.pt     # Pre-trained model checkpoint
│   ├── training_curves.png      # Loss curves visualization
│   └── combined_losses.png      # Combined loss plot
│
├── pyproject.toml               # Project dependencies (uv)
└── README.md
```

## Pre-training Objectives

The model is trained with three unsupervised objectives:

| Loss | Description |
|------|-------------|
| **MTR** | Merged Token Reconstruction - reconstruct original DNA from merged tokens |
| **Latent MTR** | Adaptive global token selection with compression |
| **AMTM** | Adaptive Masked Token Modeling - focus on high-information regions |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/probable-adventure.git
cd improbable-adventure

# Install dependencies with uv
uv sync
```

## Quick Start

### 1. Download the dataset; steps to download and use the genomic benchmark datasets 

```bash
uv run scripts/data
```
The dataset is primarily composed of classifying if a genomic sequence is a promoter or not. 

Open and run `notebooks/_02_pretrain.ipynb` for the full training pipeline example on a reconstruction task. 


Ownership is calculated dynamically. 
Data training is done. 
