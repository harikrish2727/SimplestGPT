# SimplestGPT — Phase 0

A minimal character-level GPT implementation built from scratch using PyTorch.

This project is focused on learning the core building blocks behind modern Large Language Models (LLMs) through clean and simple code rather than abstraction-heavy frameworks.

Phase 0 is the foundation stage of the project:

* Character-level tokenization
* Transformer decoder architecture
* Multi-Head Self Attention
* Pre-Norm blocks
* Feed Forward Network (FFN)
* Residual connections
* Weight tying
* Cosine learning rate scheduling with warmup
* Gradient clipping and evaluation loop
* Text generation with temperature and top-k sampling

The model is trained on the Shakespeare dataset.

---

## Goals of This Project

The purpose of SimplestGPT is to deeply understand:

* How GPT-style decoder-only transformers work
* Tensor shapes through the full forward pass
* Attention implementation from scratch
* Training dynamics of autoregressive language models
* Tokenization and data pipelines
* Sampling strategies during inference
* How modern LLM training loops are structured

This project prioritizes clarity and learning over optimization.

---

## Project Structure

```bash
SimplestGPT/
│
├── main.py                 # Entry point
├── gpt.py                  # GPT model implementation
├── train.py                # Training loop
├── dataset.py              # Dataset and sequence creation
├── data_preparation.py     # Data loading utilities
├── .env                    # Dataset URL or configuration
└── README.md
```

---

## Current Architecture

### Model Configuration

| Component           | Value              |
| ------------------- | ------------------ |
| Vocabulary Size     | 65                 |
| Embedding Dimension | 64                 |
| Context Length      | 64                 |
| Attention Heads     | 4                  |
| Transformer Blocks  | 4                  |
| Optimizer           | AdamW              |
| LR Scheduler        | Cosine with Warmup |

---

## Features Implemented

### Data Pipeline

* Character-level vocabulary creation
* Train / validation split
* Sequence dataset generation
* PyTorch DataLoader integration

### Transformer Components

* Token embeddings
* Positional embeddings
* Multi-head causal self-attention
* Feed Forward Network (FFN)
* Residual connections
* Layer normalization (Pre-Norm)
* Weight tying

### Training

* Cross entropy loss
* AdamW optimizer
* Cosine learning rate scheduler
* Warmup steps
* Gradient clipping
* Validation loop
* Model checkpoint saving

### Inference

* Temperature sampling
* Top-k sampling
* Autoregressive text generation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/harikrish2727/SimplestGPT.git
cd SimplestGPT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
url=<dataset_url>
```

---

## Running Training

```bash
python main.py
```

The model checkpoint will be saved as:

```bash
shakespere_gpt.pt
```

---

## Example Training Flow

```text
Raw Text
   ↓
Character Tokenization
   ↓
Input / Target Sequence Creation
   ↓
Embedding Layer
   ↓
Transformer Decoder Blocks
   ↓
Logits
   ↓
Cross Entropy Loss
   ↓
Backpropagation
```

---

## Learning Focus

This repository intentionally avoids:

* High-level training frameworks
* Hidden abstractions
* Excessive optimization tricks

The idea is to understand every tensor and every operation.

---

## Planned Future Phases

### Phase 1

* Rotary Positional Embeddings (RoPE)
* KV Cache for inference
* Better tokenizer
* Config system
* Improved generation pipeline

### Phase 2

* Subword tokenization (BPE / SentencePiece)
* Mixed precision training
* Flash Attention
* Larger datasets
* Checkpoint loading and resuming

### Phase 3

* Instruction tuning
* Fine-tuning pipelines
* Multi-language support
* Scaled training experiments

---

## Technologies Used

* Python
* PyTorch
* Transformers (scheduler utilities)
* dotenv

---

## Why This Exists

Most tutorials hide important implementation details.

This repository is an attempt to build a GPT-style language model step-by-step while understanding:

* the mathematics,
* the tensor dimensions,
* the training mechanics,
* and the reasoning behind each design choice.

---

## Acknowledgements

Inspired by:

* GPT architecture papers
* Andrej Karpathy's educational content
* Open-source LLM implementations

---

## License

MIT License
