# GPT from Scratch on WikiText-103

A clean, minimal PyTorch implementation of a decoder-only GPT-style Transformer (~51.2M parameters) trained from scratch on the **WikiText-103** dataset.

## Overview

- **Architecture**: Pre-LayerNorm causal Transformer with tied input/output token embeddings, learned positional embeddings, GELU feed-forward networks, and hardware-accelerated causal attention via `F.scaled_dot_product_attention` (FlashAttention).
- **Dataset**: [Salesforce/wikitext (`wikitext-103-raw-v1`)](https://huggingface.co/datasets/Salesforce/wikitext) tokenized with OpenAI's GPT-2 BPE tokenizer (`tiktoken`, vocab size `50,257`).
- **Training**: Mixed-precision (`bfloat16`) training with fused `AdamW`, gradient clipping, linear warmup + cosine annealing learning rate schedule (`train.py`), followed by constant-LR continuation (`extended.py`).

---

## Repository Structure

```text
.
├── prepare.py               # Downloads WikiText-103 and tokenizes splits into .pt tensors
├── model.py                 # GPT Transformer architecture (MultiHeadSelfAttention, FeedForward, Block, Transformer)
├── train.py                 # Main training loop (0 -> 30,000 steps with warmup + cosine decay)
├── extended.py              # Extended training loop (30,000 -> 40,000 steps at constant min LR)
└── Checkpoints/
    └── training.log         # Training and validation perplexity log across 40,000 steps
```

---

## Model Architecture & Hyperparameters

### Architecture (`model.py`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `num_embeddings` | `50,257` | GPT-2 BPE vocabulary size (`tiktoken`) |
| `block_size` | `512` | Context window length |
| `embedding_dim` | `512` | Hidden dimension ($d_{\text{model}}$) |
| `num_heads` | `8` | Attention heads ($d_{\text{head}} = 64$) |
| `num_layers` | `8` | Number of Pre-LN Transformer blocks |
| `ffn_dim` | `2048` | Feed-forward inner dimension ($4 \times d_{\text{model}}$, GELU) |
| `dropout` | `0.1` | Dropout probability |
| **Total Parameters** | **~51.2M** | Exact: `51,201,536` (with weight tying between token embeddings and LM head) |
| **Non-Embedding Parameters** | **~25.2M** | Exact: `25,207,808` (excluding token and positional embeddings) |

### Dataset & Token-to-Parameter Ratio

Tokenized with GPT-2 BPE (`tiktoken`), the WikiText-103 splits contain:
- **Train (`train.pt`)**: `118,496,151` tokens (~118.5M)
- **Validation (`val.pt`)**: `248,461` tokens (~248.5K)
- **Test (`test.pt`)**: `284,614` tokens (~284.6K)

| Metric | Tokens | vs. Total Params (`51.2M`) | vs. Non-Embedding Params (`25.2M`) |
| :--- | ---: | ---: | ---: |
| **Unique Train Dataset (`train.pt`)** | `118.5M` | **2.31 tokens / param** | **4.70 tokens / param** |
| **Phase 1 Training (`30k` steps, ~4.15 epochs)** | `491.5M` | **9.60 tokens / param** | **19.50 tokens / param** |
| **Phase 2 Extended (`40k` steps, ~5.53 epochs)** | `655.4M` | **12.80 tokens / param** | **26.00 tokens / param** |

### Optimization (`train.py` & `extended.py`)

| Setting | Phase 1 (`train.py`) | Phase 2 (`extended.py`) |
| :--- | :--- | :--- |
| Steps | `0` $\rightarrow$ `30,000` | `30,000` $\rightarrow$ `40,000` |
| Batch Size | `32` (`16,384` tokens/step) | `32` (`16,384` tokens/step) |
| Optimizer | `AdamW` ($\beta_1=0.9, \beta_2=0.95$, `fused=True`) | `AdamW` ($\beta_1=0.9, \beta_2=0.95$, `fused=True`) |
| Peak / Min LR | `6e-4` $\rightarrow$ `6e-5` | Constant `6e-5` |
| LR Schedule | `1,000`-step linear warmup + cosine annealing | Fixed learning rate |
| Precision | `bfloat16` (`torch.autocast`) | `bfloat16` (`torch.autocast`) |
| Grad Clipping | Max L2 norm `1.0` | Max L2 norm `1.0` |

---

## Training Results

Logged training and validation perplexity ($\text{PPL} = e^{\text{loss}}$) on WikiText-103 from `Checkpoints/training.log`:

| Iteration | Train Perplexity | Validation Perplexity | Phase |
| ---: | ---: | ---: | :--- |
| `0` | `52092.88` | `51846.01` | Initialization |
| `1,000` | `186.63` | `178.34` | End of Warmup |
| `5,000` | `48.30` | `48.53` | Cosine Decay |
| `10,000` | `37.06` | `37.83` | Cosine Decay |
| `20,000` | `29.42` | `31.09` | Cosine Decay |
| `30,000` | `26.65` | `27.84` | End of Phase 1 (`train.py`) |
| `39,000` | `26.66` | **`27.31`** | Best Validation PPL (`extended.py`) |
| `39,500` | `26.57` | `27.56` | Extended Phase |

---

## Getting Started

### 1. Install Dependencies

```bash
pip install torch tiktoken datasets
```

### 2. Prepare the Dataset

Download `wikitext-103-raw-v1` and serialize the GPT-2 token IDs into `train.pt`, `val.pt`, and `test.pt`:

```bash
python prepare.py
```

### 3. Train the Model

Run the primary 30,000-step training phase (automatically selects `cuda`, `mps`, or `cpu`):

```bash
python train.py
```

> **Note:** By default, `train.py` and `extended.py` write checkpoints (`best.pt`, `final.pt`, and `checkpoint_<step>.pt`) to `save_dir = "/workspace/GPT/checkpoints"`. Update `save_dir` in `train.py` / `extended.py` if running in a local directory such as `./Checkpoints`.

### 4. Extended Training (Optional)

Resume training from `final.pt` at step `30,000` to `40,000` with a constant learning rate of `6e-5`:

```bash
python extended.py
```
