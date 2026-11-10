# Transformer Machine Translation from Scratch in PyTorch

A complete PyTorch implementation of the **Encoder-Decoder Transformer architecture** from scratch (based on Vaswani et al., *"Attention Is All You Need"*), applied to English-to-French Neural Machine Translation (NMT).

## Overview

This repository contains a modular implementation of the Transformer model without relying on high-level prebuilt transformer libraries (`nn.Transformer`). Every component—including scaled dot-product attention, multi-head attention projections, sinusoidal positional encodings, masked autoregressive decoding, and position-wise feed-forward networks—is built directly using core PyTorch tensor operations.

---

## Architecture Design

```
                     [Target Output (French)]
                               │
                       [Shifted Right]
                               │
                       [Output Embedding]
                               │
                 ┌─────────────▼─────────────┐
                 │    + Positional Encoding  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Masked Multi-Head   │
                    │   Self-Attention    │
                    └──────────┬──────────┘
                               │ (Add & Norm)
                               ▼
[Source (English)]  ┌─────────────────────┐
        │           │ Cross Multi-Head    │<─── [Encoder Output]
 [Input Embedding]  │      Attention      │
        │           └──────────┬──────────┘
┌───────▼────────┐             │ (Add & Norm)
│+ Pos Encoding  │             ▼
└───────┬────────┘  ┌─────────────────────┐
        │           │ Feed-Forward Net    │
┌───────▼────────┐  └──────────┬──────────┘
│ Encoder Layer  │             │ (Add & Norm)
│  (Nx = 6)      │             ▼
└───────┬────────┘       [Linear Head]
        │                      │
        └────────────────> [Softmax] ──> Output Tokens
```

### Key Components

1. **Scaled Dot-Product Multi-Head Attention** (`utils.py`):
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   Projects queries, keys, and values across $h = 8$ parallel attention heads ($d_k = d_v = 64$).
2. **Sinusoidal Positional Encoding** (`utils.py`):
   Injects deterministic sequence order information into input embeddings using alternating sine and cosine functions across dimensions.
3. **Dual Masking Mechanism** (`utils.py`):
   - **Padding Mask**: Zeroes out attention weights for padding tokens (`<pad>`) across source and target batches.
   - **Look-Ahead Mask**: Lower-triangular masking ensuring autoregressive causality in the decoder during training.
4. **Deep 6-Layer Stacks** (`encoder.py`, `decoder.py`):
   - 6 Encoder layers and 6 Decoder layers with Layer Normalization and Residual Skip Connections.
   - Position-Wise Feed-Forward Networks with hidden dimension $d_{ff} = 2048$.

---

## Model Configuration & Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Dimension ($d_{\text{model}}$)** | 512 | Embedding and hidden representation dimension |
| **Number of Layers ($N$)** | 6 | Identical encoder and decoder stacked layers |
| **Attention Heads ($h$)** | 8 | Parallel multi-head attention projections |
| **Feed-Forward Dim ($d_{ff}$)** | 2048 | Inner dense layer expansion dimension |
| **Max Sequence Length** | 100 | Maximum input/target sequence cutoff |
| **Dropout** | 0.2 | Regularization across attention and residual connections |
| **Batch Size** | 64 | Training mini-batch size |
| **Optimizer** | Adam | $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$ |
| **Learning Rate** | 1e-4 | Peak learning rate with warmup |

---

## Repository Structure

```
├── encoder.py              # EncoderLayer and 6-layer Encoder stack
├── decoder.py              # DecoderLayer with masked self-attention & cross-attention
├── utils.py                # MultiHeadAttention, PositionalEncoding, Masking utilities
├── train.py                # Training loop, data preprocessing & tokenization
├── test.py                 # Greedy autoregressive inference & BLEU evaluation
├── transformers_best.ipynb # Interactive end-to-end training & evaluation notebook
├── testbleu.txt            # Generated sample translations vs. ground truth references
├── Report.pdf              # Comprehensive technical report
└── README.md               # Project documentation
```

---

## Getting Started

### 1. Requirements
```bash
pip install torch torchvision nltk numpy tqdm matplotlib
```

### 2. Training
```bash
python train.py
```

### 3. Evaluation (BLEU Score)
```bash
python test.py
```
Evaluates candidate translations against reference French sentences using NLTK BLEU score with smoothing functions.
