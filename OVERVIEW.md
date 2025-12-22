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
    |
     - self.download_data() 
       - Checks if input.txt already exists
       - Downloads TinyShakespeare dataset (1.1 MB text, Shakespeare's works)
       - Saves as UTF-8 text file

     - self.train_tokenizer()        |
         - Step 1: Tokenizer Initialization

            ```bash
            Tokenizer(BPE(unk_token="[UNK]"))
            ```
            - **Algorithm**: Byte-Pair Encoding (BPE) :- BPE is like learning vocabulary by merging frequent pairs. 
            - **Example**:
                - Step 1: Start with letters (bytes)

                ``` 
                Initial: l o w e r l o w e s t 
                ```
                - Step 2: Find most frequent pair
                    ```
                    Pairs: lo(2), ow(2), we(1), er(1), r_(space)(1), _l(1), lo(again), etc.
                    "lo" appears most (2 times)
                    ```
                - Step 3: Merge them
                    ```
                    After merge: lo w e r lo w e s t
                    Vocabulary adds: "lo"
                    ```
                - Step 4: Repeat
                    ```
                    Next frequent: "low" (appears 2 times)
                    Merge: low e r low e s t
                    Vocabulary adds: "low"
                    ```
                - **Final vocabulary** might have: l, o, w, e, r, s, t, lo, low, etc.
                - **Mathematical Formula**
                    ```
                    While vocabulary_size < target_size:
                        1. Count all adjacent pairs in corpus
                        2. Find pair (A, B) with highest frequency
                        3. Replace all (A, B) with new token AB
                        4. Add AB to vocabulary
                    ```
            - unk_token="[UNK]": Token for unknown/out-of-vocabulary words
        - Step 2: Pre-tokenizer Setup
            ```
            tokenizer.pre_tokenizer = Whitespace()
            ```
            - **Purpose**: Splits text into words before BPE
            - **Example**: "hello world!" → ["hello", "world!"]
            - Preserves punctuation attached to words
        - Step 3: Trainer Configuration
            ```
            BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size) 
            ```
            - **special_tokens**:
                - [UNK]: The Unknown Word Handler
                    What it means: "I don't know this word!"
                    When it's used: When the tokenizer encounters a word it has never seen before during training.
                    ```
                    # Tokenizer was trained on English, now sees:
                    text = "I love schlorpflingen"  # Made-up word
                    # Encoding result:
                    ids = [45, 128, [UNK]]  # [UNK] gets a specific ID like 1
                    ``` 
                - [CLS]: Classification token "The Summarizer Token"
                    What it means: "This represents the whole sentence!"
                    When it's used: At the beginning of every sequence in models like BERT.
                    ```
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
                    ```
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
        - Step 4: Training Process
            ```
            tokenizer.train(files, trainer)
            ```
            BPE Training algorithm with example
            ```
            Corpus: "low lower lowest"
            Step 1: ["l", "o", "w", " ", "l", "o", "w", "e", "r", " ", "l", "o", "w", "e", "s", "t"]
            Step 2: Frequency Counting: Count all adjacent symbol pairs in corpus
                    Merge "l"+"o" → "lo" (appears 3x)
                    Merge "lo"+"w" → "low" (appears 3x)
                    Merge "e"+"r" → "er" (appears 1x)
            Step 3: Merge Iterations:
                    - Find most frequent pair (e.g., "t" + "h" → "th")
                    - Merge them into new token
                    - Repeat until reaching vocab_size
            Special Tokens: Always included regardless of frequency
            ```
        - Step 5: Saving & Loading
            ```
            tokenizer.save(self.tokenizer_file)  # Saves as JSON
            tokenizer = Tokenizer.from_file(self.tokenizer_file)  # Loads from JSON
            ```
            Tokenizer JSON Structure
            ```
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
         Flow:
            - pre_tokenizer: Split by whitespace → ["Hello", "world!"]
            - BPE: Apply merges → ["Hello", "world", "!"] (if "world" in vocab)
            - Convert to IDs using vocabulary mapping
     - **Train/Val/Test** Split
        ```
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



3. **Architecture**: 
   - **File**: `architecture.py`
