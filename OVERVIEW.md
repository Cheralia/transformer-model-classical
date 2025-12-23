# Project Overview

This file provides a brief overview of the **Classical Transformer Implementation** project, including the workflow and the purpose of each component.

## Tokenizer 

The first step is `tokenizing`  
A `tokenizer` is like a text chopper that converts raw text into smaller pieces (tokens) that machine learning models can understand.
Computers don't understand words, they understand numbers!
```
"I love pizza" → Tokenizer → [45, 128, 891] → Computer: ✅ "Ah, numbers I can work with!"
```

The workflow for `tokenizer.py` involves the following components:

1. **DataHandler Class Initialization (__init__)**: 
Inside main `dh = DataHandler()` call the __init__ function of DataHandler class:
   - **Purpose**: Sets up file paths and creates directory structure
        - **data_path**: Directory where all data/tokenizer files are stored
        - **input_file**: Path to raw text data (data/input.txt)
        - **tokenizer_file**: Path to save trained tokenizer (data/tokenizer.json)

2. **Tokenize**: Using the object dh `dh.prepare_tensors()` call the prepare_tensors() inside DataHandler class:

    prepare_tensors()
     - self.download_data() 
       - Checks if input.txt already exists
       - Downloads TinyShakespeare dataset (1.1 MB text, Shakespeare's works)
       - Saves as UTF-8 text file

     - self.train_tokenizer()
         - Step 2.1: Tokenizer Initialization

            ```bash
            Tokenizer(BPE(unk_token="[UNK]"))
            ```
            - **Algorithm**: Byte-Pair Encoding (BPE) :- BPE is like learning vocabulary by merging frequent pairs. 
            - **Example**:
                - Step 2.1.1: Start with letters (bytes)

                ```bash 
                Initial: l o w e r l o w e s t 
                ```
                - Step 2.1.2: Find most frequent pair
                    ```bash
                    Pairs: lo(2), ow(2), we(1), er(1), r_(space)(1), _l(1), lo(again), etc.
                    "lo" appears most (2 times)
                    ```
                - Step 2.1.3: Merge them
                    ```bash
                    After merge: lo w e r lo w e s t
                    Vocabulary adds: "lo"
                    ```
                - Step 2.1.4: Repeat
                    ```bash
                    Next frequent: "low" (appears 2 times)
                    Merge: low e r low e s t
                    Vocabulary adds: "low"
                    ```
                - **Final vocabulary** might have: l, o, w, e, r, s, t, lo, low, etc.
                - **Mathematical Formula**
                    ```bash
                    While vocabulary_size < target_size:
                        1. Count all adjacent pairs in corpus
                        2. Find pair (A, B) with highest frequency
                        3. Replace all (A, B) with new token AB
                        4. Add AB to vocabulary
                    ```
            - unk_token="[UNK]": Token for unknown/out-of-vocabulary words
        - Step 2.2: Pre-tokenizer Setup
            ```
            tokenizer.pre_tokenizer = Whitespace()
            ```
            - **Purpose**: Splits text into words before BPE
            - **Example**: "hello world!" → ["hello", "world!"]
            - Preserves punctuation attached to words
        - Step 2.3: Trainer Configuration
            ```
            BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size) 
            ```
            - **special_tokens**:
                - [UNK]: The Unknown Word Handler
                    What it means: "I don't know this word!"
                    When it's used: When the tokenizer encounters a word it has never seen before during training.
                    ```bash
                    # Tokenizer was trained on English, now sees:
                    text = "I love schlorpflingen"  # Made-up word
                    # Encoding result:
                    ids = [45, 128, [UNK]]  # [UNK] gets a specific ID like 1
                    ``` 
                - [CLS]: Classification token "The Summarizer Token"
                    What it means: "This represents the whole sentence!"
                    When it's used: At the beginning of every sequence in models like BERT.
                    ```bash
                    # For classification tasks (like sentiment analysis):
                    text = "This movie was amazing!"
                    # With [CLS] token:
                    encoded = [[CLS], "This", "movie", "was", "amazing", "!"]
                    ids = [101, 45, 128, 67, 8914, 27]  # 101 is often [CLS] ID
                    # The model uses the [CLS] token's representation 
                    # to make predictions about the whole sentence
                    ```
                - [SEP]: Separation token
                    What it means: "This is where one thing ends and another begins!"
                    When it's used:
                        - Between two sentences
                        - At the end of a single sentence
                    ```bash
                    # For question-answering or two-sentence tasks:
                    question = "What is AI?"
                    context = "Artificial Intelligence is..."
                    # Encoding with [SEP]:
                    tokens = [[CLS], "What", "is", "AI", "?", [SEP], 
                            "Artificial", "Intelligence", "is", "...", [SEP]]
                    ids = [101, 45, 67, 891, 15, 102, 2451, 5421, 67, 99, 102]
                    ```
                - [PAD]: Padding for batch alignment
                    What it means: "Fill empty spaces so everything is the same length!"
                    When it's used: When making batches of different-length sequences.
                    ```
                    # Three sentences of different lengths:
                    sentences = [
                        "Hi",                    # 1 word
                        "Hello there",           # 2 words  
                        "Good morning to you"    # 4 words
                    ]
                    # Without padding (can't make a matrix):
                    [45]
                    [128, 245]
                    [891, 542, 67, 99]
                    # WITH padding to length 4:
                    [[45, [PAD], [PAD], [PAD]],
                    [128, 245, [PAD], [PAD]],
                    [891, 542, 67, 99]]          
                    # Now we can make a nice 3×4 matrix!
                    ```
                - [MASK]: The Blank to Fill
                    What it means: "Guess what word goes here!"
                    When it's used: During Masked Language Model (MLM) training (like BERT).
                    ```
                    # Original sentence:
                    text = "The cat sat on the mat"
                    # For training, we randomly mask 15% of words:
                    masked = "The [MASK] sat on the [MASK]"
                    # Model's task: Predict the masked words
                    # Input: [CLS] The [MASK] sat on the [MASK] [SEP]
                    # Output should be: "cat" and "mat"
                    ```
            - **vocab_size**: Maximum vocabulary size (50,000 tokens)
        - Step 2.4: Training Process
            ```bash
            tokenizer.train(files, trainer)
            ```
            BPE Training algorithm with example
            ```bash
            Corpus: "low lower lowest"
            Step 2.4.1: ["l", "o", "w", " ", "l", "o", "w", "e", "r", " ", "l", "o", "w", "e", "s", "t"]
            Step 2.4.2: Frequency Counting: Count all adjacent symbol pairs in corpus
                    Merge "l"+"o" → "lo" (appears 3x)
                    Merge "lo"+"w" → "low" (appears 3x)
                    Merge "e"+"r" → "er" (appears 1x)
            Step 2.4.3: Merge Iterations:
                    - Find most frequent pair (e.g., "t" + "h" → "th")
                    - Merge them into new token
                    - Repeat until reaching vocab_size
            Special Tokens: Always included regardless of frequency
            ```
        - Step 2.5: Saving & Loading
            ```bash
            tokenizer.save(self.tokenizer_file)  # Saves as JSON
            tokenizer = Tokenizer.from_file(self.tokenizer_file)  # Loads from JSON
            ```
            Tokenizer JSON Structure
            ```bash
                {
                "model": {
                    "type": "BPE",
                    "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                    "[MASK]": 4,
                    "the": 5,
                    "and": 6,
                    // ... up to 50,000 entries
                    },
                    "merges": [
                    "t h",    # First merge: t + h → th
                    "th e",   # Second merge: th + e → the
                    "e space" # Third merge: e + space → e_
                    // ... thousands of merges
                    ]
                },
                "pre_tokenizer": {"type": "Whitespace"}
                }
           ```
     - **tokenizer.encode(text).ids** "Encoding Process"
         - Flow:
            - pre_tokenizer: Split by whitespace → ["Hello", "world!"]
            - BPE: Apply merges → ["Hello", "world", "!"] (if "world" in vocab)
            - Convert to IDs using vocabulary mapping
     - **Train/Val/Test** Split
        ```bash
        n = len(data)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        train_data = data[:train_end]    # 80%
        val_data = data[train_end:val_end]  # 10%
        test_data = data[val_end:]       # 10%
        ```
     - **File Saving**
        ```
        torch.save(train_data, "train.pt")  # PyTorch tensor format
        ```
        Format: PyTorch tensor (.pt file) of dtype torch.long
            - Memory efficient binary format
            - Directly loadable with torch.load()
     - **Directory Structure After Execution**
        ```
        data/
        ├── input.txt              # Raw Shakespeare text (1MB)
        ├── tokenizer.json         # BPE vocabulary + merges (JSON)
        ├── train.pt              # Training tensor (80% of tokens)
        ├── val.pt                # Validation tensor (10%)
        └── test.pt               # Test tensor (10%)
        ```

## Complete Training Flow 

Here the `train.py` flow is briefly explained. The training flow prepares the data tensors produced by the tokenizer (if not exist produce new) and runs batch-based optimization on the model defined in `architecture.py`.
 
### Step 1. Prepare Data

```
dh = DataHandler()
vocab_size = dh.prepare_tensors()
train_data = torch.load("data/train.pt")
val_data = torch.load("data/val.pt")
```
- The first two lines do tokenization if not done before.
- The last two line Load tensors: `train.pt`and `val.pt` (dtype: torch.long). In PyTorch, `dtype: torch.long` refers to the data type of a tensor. 
    - In PyTorch, a **tensor** is a multi-dimensional array used to store data. Tensors are the primary data structure in PyTorch, similar to arrays in NumPy, but they come with additional capabilities optimized for deep learning, such as GPU acceleration. 
    -  Tensors can hold various data types, in this case `torch.long` is used, corresponds to a 64-bit signed integer (int64). Thus, the range of values that can be represented is from -2^63 to 2^63 - 1.

### Step 2. Initiate Model

```bash
config = ModelConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE)
model = Transformer(config).to(DEVICE)  
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```
- **config** create ModelConfig object, which is find inside `architecture/architecture.py`.
    ```bash
        config = ModelConfig(
                vocab_size=vocab_size,  # ~50,000 (from tokenizer)
                block_size=128,         # Context window
                n_layer=4,              # 4 transformer blocks
                n_head=4,               # 4 attention heads
                n_embd=256,             # 256 embedding dimensions
                dropout=0.1             # 10% dropout
                )
    ```
    - `block_size=128`: Model can see 128 tokens at once (balance of context vs memory)
    - `n_embd=256`: Each token represented as 256-dimensional vector
    - `n_head=4`: 4 parallel attention heads (256/4 = 64 dimensions per head)
    - `n_layer=4`: 4 transformer blocks (deeper = more complex patterns)

- **Model instantiation**: `model = TransformerModel(...)` (see Model Architecture explanation of `architecture.py`, explained next section)
- **Optimizer**: commonly `AdamW` with weight decay; optional LR scheduler
    - An **optimizer** is an algorithm that modifies the attributes of the model, such as weights and biases, to reduce the loss during training. It adjusts the learning parameters based on the gradients computed from backpropagation.
    - **Adam** stands for **Adaptive Moment Estimation**, which combines the benefits of two other popular optimizers: AdaGrad and RMSProp. It adjusts the learning rate for each parameter individually based on the first and second moments of the gradients.
    - The **W** in **AdamW** stands for Weight Decay. This method adds an extra term to the loss function to prevent overfitting by penalizing large weights, effectively regularizing the model.
    - **model.parameters()**: This retrieves the parameters (weights and biases) of the model to be optimized.
    - **lr=LEARNING_RATE**: This sets the learning rate, a hyperparameter controlling the step size at each iteration while moving toward a minimum of the loss function.

### Step 3. Training Loop

```bash
model.train()
```

- Batching: create mini-batches of shape (batch_size, seq_len) and move to device (CPU/GPU)
- Loss: Cross-entropy between model logits and target token IDs (PyTorch: `nn.CrossEntropyLoss()`)
- 
- Training loop (per step):
    1. zero gradients
    2. forward pass: outputs = model(input_ids) → logits of shape (B, S, V)
    3. compute loss: loss = CE(logits.view(-1, V), targets.view(-1))
    4. backward: loss.backward() (optionally gradient clip)
    5. optimizer.step(); scheduler.step() (if used)
    6. checkpointing: save model state_dict, optimizer state, and tokenizer periodically

Important training details: use a causal (autoregressive) attention mask for next-token prediction so each position can only attend to previous positions; evaluate with perplexity = exp(average_loss).

## Model architecture (file: `architecture.py`)

Below are the high-level components of the typical small autoregressive Transformer implemented in this repo and what each computes (shapes use B=batch, S=seq_len, D=model_dim, H=num_heads, V=vocab_size):

```bash
# Word Embeddings
self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
# Position Embeddings (classical approach - learnable absolute positional embeddings)
self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
# Transformer blocks
self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
self.ln_f = nn.LayerNorm(config.n_embd)
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

- **Token Embeddings**
    - the **Embedding class** uses the input indices to form a 3D tensor where each token ID is replaced by its corresponding embedding vector, resulting in the output shape (B, S, D).
    - Purpose: map input token ids to continuous vectors.
    - Input: token ID of vocabulary (e.g., 1834) and embedding dimension
    - Output: 256-dim vector
    - Computation: Emb = Embedding(V, D); input_ids -> X of shape (B, S, D)
        - B is batch size.
        - S is sequence length.
        - D is dimension of each embedding vector.

- **Positional Encoding** 
    - Positional encoding adds information about token order (position in the sequence) to a Transformer. Transformers process all tokens in parallel, so by default they do not know which token comes first, second, etc.
    - The nn.Embedding layer creates a lookup table that maps each position index (from 0 to block_size - 1) to a continuous vector of size n_embd. This allows the model to learn a unique embedding for each position in the input sequence, making it aware of the relative or absolute positions of tokens.
    - Effect: makes self-attention position-aware.

        ```bash
        Without positional information:
        “dog bites man” and “man bites dog” look identical to the model.Positional embeddings let the model distinguish sequence order.
        ```
- Positional embeddings tell the Transformer `where` a token appears, while token embeddings tell it `what` the token is.

- **Transformer Block** (stacked N times)
    - Create 4 transformer blocks in our case, since n_layer is 4.
    - Each block contains:
        ```bash
        class Block(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.ln1 = nn.LayerNorm(config.n_embd)
                self.attn = CausalSelfAttention(config)
                self.ln2 = nn.LayerNorm(config.n_embd)
                self.mlp = MLP(config)
        ```
        1. **Layer Normalization**
            - Layer normalization applied before the attention mechanism, stabilizing the input, which helps in training deep networks.
            - The main idea is to standardize the inputs of a layer by removing the mean and scaling by the variance. It ensures that the features have a mean of 0 and a standard deviation of 1, which can lead to more stable and efficient training.
            - For a given input 𝑥 (for example, from the previous layer's output): ![alt text](image.png) Where 𝜖 is a small constant added to avoid division by zero.
            - Benefits
                - Stability: Reduces the risk of exploding or vanishing gradients.
                - Training Speed: Can lead to faster convergence.
                - Independence: Works effectively with varying batch sizes.

        2. **Causal Multi-Head Self-Attention**
        - It called causal because: Token at position t can attend only to positions ≤ t never to future tokens.
        - First check that the embedding size (n_embd) is divisible by the number of attention heads (n_head). This is necessary to evenly split the input features among the attention heads.
            ```bash
            assert config.n_embd % config.n_head == 0
            ```
        -  Calculates the dimensionality of each attention head by dividing the total embedding size by the number of heads. In our case it is 256/4 = 64.
            ```bash
            self.head_dim = config.n_embd // config.n_head
            ```
        - Stores the number of heads and the embedding dimension in instance variables for later use.
            ```bash 
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            ```
        - Initializes a linear layer for computing query, key, and value (QKV) projections in one matrix multiplication. The output size is three times the embedding dimension, as each input generates a query, key, and value. `bias=False` disables the bias vector in this linear layer.
            ```bash
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
          ```
        - Initializes another linear layer for projecting the output of the attention mechanism back into the embedding space. Final projection after combining all heads. The idea is Mix information across heads. 
            ```bash
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
            ```  
        - Initializes dropout layers for regularization, using the dropout rate specified in the config.
            - attn_dropout: applied to attention weights
            - resid_dropout: applied to output projection
            ```bash
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            ```
        - Creates a lower triangular matrix (bias) and registers it as a buffer. This mask is used to prevent attending to future tokens, essential for causal attention. 
             ```bash
             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
             ```

        3. **Residual + LayerNorm (post-attention)**
             - Purpose: stabilizes training and eases gradient flow.
             - Computation: X = LayerNorm(X + attn_out)

        4. **MLP - Position-wise Feed-Forward Network (FFN)**
             - Purpose: apply non-linearity and expand per-position representation.
             - Computation: FFN(x) = W_2(activation(W_1(x) + b1)) + b2
                 - Typical shapes: W_1: D → D_ff (e.g., 4×D), W_2: D_ff → D
             - Activation: GELU or ReLU

- **Residual + LayerNorm (post-FFN)**
    - Computation: X = LayerNorm(X + FFN_out)

- **Final linear & output**
    - Purpose: map model hidden states to vocabulary logits for each position.
    - Computation: logits = Linear(D → V) applied to each position → shape (B, S, V)
    - Prediction: probabilities = softmax(logits, dim=-1)

### Loss and metrics
    - Cross-entropy loss computed between logits and shifted target token ids (next-token prediction).
    - Perplexity = exp(mean_loss) is commonly reported on validation set.

````
