import os
import requests
import torch
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class DataHandler:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self.input_file = os.path.join(data_path, "input.txt")
        self.tokenizer_file = os.path.join(data_path, "tokenizer.json")

    def download_data(self):
        if not os.path.exists(self.input_file):
            print("Downloading Tiny Shakespeare...")
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(self.input_file, 'w') as f:
                f.write(requests.get(url).text)

    def train_tokenizer(self, vocab_size=50000):
        if not os.path.exists(self.tokenizer_file):
            print("Training BPE Tokenizer...")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
            
            with open(self.input_file, 'r', encoding='utf-8') as f:
                files = [self.input_file]
                tokenizer.train(files, trainer)
            tokenizer.save(self.tokenizer_file)
            return tokenizer
        return Tokenizer.from_file(self.tokenizer_file)

    def prepare_tensors(self):
        self.download_data()
        tokenizer = self.train_tokenizer()
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print("Encoding data...")
        ids = tokenizer.encode(text).ids
        data = torch.tensor(ids, dtype=torch.long)

        # Split into fixed-size chunks, then shuffle chunk ORDER (with a fixed
        # seed for reproducibility) before assigning to train/val/test. This
        # avoids the previous contiguous 80/10/10 cut, where val/test were
        # simply whichever play(s) happened to sit at the end of the file --
        # a different distribution than most of train, not just "held out".
        # Chunks themselves stay internally contiguous so local context/coherence
        # within a chunk is preserved; only the assignment order is shuffled.
        chunk_size = 512
        n = len(data)
        n_chunks = n // chunk_size
        chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

        rng = random.Random(42)
        rng.shuffle(chunks)

        train_end = int(0.8 * n_chunks)
        val_end = int(0.9 * n_chunks)

        train_data = torch.cat(chunks[:train_end])
        val_data = torch.cat(chunks[train_end:val_end])
        test_data = torch.cat(chunks[val_end:])
        
        torch.save(train_data, os.path.join(self.data_path, "train.pt"))
        torch.save(val_data, os.path.join(self.data_path, "val.pt"))
        torch.save(test_data, os.path.join(self.data_path, "test.pt"))
        
        print(f"Data saved to {self.data_path}/ (Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)})")
        return tokenizer.get_vocab_size()

if __name__ == "__main__":
    dh = DataHandler()
    dh.prepare_tensors()