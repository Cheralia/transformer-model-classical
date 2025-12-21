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
### 3. Training
Run the training loop. This will print the Perplexity (PPL) for every batch.

```bash
python3 train.py
```
### 4. Testing
After training, run evaluation on the held-out test set and generate text:

```bash
python3 test.pyy
```
