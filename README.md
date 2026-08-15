# Classical Transformer Implementation

This repository contains a PyTorch implementation of a Transformer language model trained on the Tiny Shakespeare dataset.

## 🏗️ Architecture Overview

This implementation follows the classic Transformer architecture from the "Attention Is All You Need" paper:

### **Classical Positional Embeddings**

- **Word Embeddings:** Learned embeddings for each token in the vocabulary
- **Positional Embeddings:** Learnable absolute positional embeddings added to word embeddings
- **Multi-Head Attention:** Standard scaled dot-product attention with causal masking
- **Feed-Forward Networks:** GELU-activated MLP layers

### **Key Design Choices**

1. **Weight Tying:** The token embedding matrix shares weights with the final linear layer
2. **Layer Normalization:** Applied before each attention and MLP layer (pre-norm architecture)
3. **Residual Connections:** Around both attention and MLP blocks
4. **Dropout:** Applied to attention scores and MLP outputs for regularization

## 📂 Project Structure

- `architecture/`: Contains the model definition
- `tokenizer/`: Handles BPE tokenizer training and data splitting
- `train.py`: Training loop with perplexity logging
- `test.py`: Inference and test set evaluation
- `eval_baseline.py`: Full-dataset, deterministic perplexity evaluation (val + test)
- `hyperparameter_search.py`: Lightweight random hyperparameter search bounded by fixed training steps

## 🛠️ Installation & Usage

### 1. Environment Setup

1. First clone the repository:

```bash
git clone https://github.com/Cheralia/transformer-model.git
```

2. Configure the environment

- It is recommended to use a virtual environment.

**Mac/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windos:**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data & Tokenizer

The tokenizer script will download the dataset, train a BPE tokenizer, and save train.pt, val.pt, and test.pt to the data/ folder.

```bash
python3 tokenizer/tokenizer.py
```

> **Note on data splitting:** The train/val/test split is built by dividing the
> tokenized corpus into 512-token chunks and shuffling their order (fixed seed)
> before assigning 80/10/10 to each split. This avoids a naive contiguous split,
> which would make the test set just the tail of the source file (a different
> distribution than training, since Tiny Shakespeare is many plays concatenated
> together).

### 3. Training

Run the training loop. This will print the Perplexity (PPL) for every batch,
and save the trained model weights.

```bash
python3 train.py
```

Hyperparameters can be set via CLI args (defaults shown):

```bash
python3 train.py --lr 3e-4 --n_layer 4 --n_head 4 --n_embd 256 \
                  --dropout 0.1 --batch_size 32 --epochs 5 --out model.pth
```

The script also tracks validation loss each epoch and saves a separate
`<name>_best.pth` checkpoint whenever it improves — useful if training runs
long enough to start overfitting.

### 4. Testing

After training, run evaluation on the held-out test set and generate text:

```bash
python3 test.py
```

### 5. Full-Dataset Evaluation

`test.py` reports test perplexity from a single random batch, which can be
noisy. For a stable, reproducible number across the entire val/test sets:

```bash
python3 eval_baseline.py --model model.pth --n_layer 4 --n_embd 256 --n_head 4 --dropout 0.1
```

### 6. Hyperparameter Search

A lightweight random search over learning rate, model size, dropout, and
batch size, bounded by a fixed number of training steps per trial so runtime
stays predictable on CPU:

```bash
python3 hyperparameter_search.py --n_trials 8 --trial_steps 200
```