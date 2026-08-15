import torch
import math
import argparse
from tokenizers import Tokenizer
from architecture.architecture import Transformer, ModelConfig

DEVICE = 'cpu'
BLOCK_SIZE = 128
BATCH_SIZE = 32

def evaluate_full(model, data, block_size, batch_size):
    """
    Evaluates perplexity over the ENTIRE dataset by walking through it in
    non-overlapping chunks, instead of a single random batch. This gives a
    stable, reproducible number to use as a baseline / comparison point.
    """
    model.eval()
    total_loss = 0.0
    n_chunks = 0

    # Non-overlapping windows across the whole dataset
    max_start = len(data) - block_size - 1
    starts = list(range(0, max_start, block_size))

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            if not batch_starts:
                continue
            x = torch.stack([data[s:s + block_size] for s in batch_starts]).to(DEVICE)
            y = torch.stack([data[s + 1:s + block_size + 1] for s in batch_starts]).to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item() * len(batch_starts)
            n_chunks += len(batch_starts)

    avg_loss = total_loss / n_chunks
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, n_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.pth", help="Path to the trained model weights.")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = Transformer(config).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    val_data = torch.load("data/val.pt")
    test_data = torch.load("data/test.pt")

    val_loss, val_ppl, val_n = evaluate_full(model, val_data, BLOCK_SIZE, BATCH_SIZE)
    test_loss, test_ppl, test_n = evaluate_full(model, test_data, BLOCK_SIZE, BATCH_SIZE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 60)
    print(f"BASELINE MODEL PERFORMANCE — {args.model}")
    print("=" * 60)
    print(f"Params: {n_params:.2f}M | Vocab size: {vocab_size} | Block size: {BLOCK_SIZE}")
    print(f"Config: n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} dropout={args.dropout}")
    print(f"Validation: {val_n} windows | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
    print(f"Test:       {test_n} windows | Loss: {test_loss:.4f} | Perplexity: {test_ppl:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()