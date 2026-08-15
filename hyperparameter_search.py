"""
Lightweight random hyperparameter search for the classical Transformer.

Design choices for a CPU-only, short-run budget:
- Each trial is bounded by a fixed number of TRAINING STEPS (not epochs), so
  trial time is predictable regardless of batch size / model size.
- Search order of importance: learning rate > model capacity (n_layer/n_embd)
  > dropout > batch size.
- Each trial is scored on validation perplexity, computed over several
  non-overlapping windows (not one random batch) for a less noisy signal.

Usage:
    python3 hyperparameter_search.py --n_trials 8 --trial_steps 200

After it finishes, it prints the best config and the exact train.py command
to run a full training run with it.
"""
import argparse
import json
import math
import random
import time
import torch

from tokenizer.tokenizer import DataHandler
from architecture.architecture import Transformer, ModelConfig

DEVICE = 'cpu'
BLOCK_SIZE = 128

SEARCH_SPACE = {
    "lr":       [1e-4, 3e-4, 6e-4, 1e-3],
    "n_layer":  [4, 6],
    "n_embd":   [256, 384],
    "dropout":  [0.0, 0.1, 0.2],
    "batch_size": [32, 64],
}
# n_head is derived to keep head_dim reasonable (n_embd // n_head == 64)
def n_head_for(n_embd):
    return max(1, n_embd // 64)

def sample_config(rng):
    n_embd = rng.choice(SEARCH_SPACE["n_embd"])
    return {
        "lr": rng.choice(SEARCH_SPACE["lr"]),
        "n_layer": rng.choice(SEARCH_SPACE["n_layer"]),
        "n_embd": n_embd,
        "n_head": n_head_for(n_embd),
        "dropout": rng.choice(SEARCH_SPACE["dropout"]),
        "batch_size": rng.choice(SEARCH_SPACE["batch_size"]),
    }

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def eval_windows(model, data, block_size, batch_size, max_windows=200):
    """Deterministic perplexity over up to max_windows non-overlapping chunks."""
    model.eval()
    max_start = len(data) - block_size - 1
    starts = list(range(0, max_start, block_size))[:max_windows]
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i+batch_size]
            if not batch_starts:
                continue
            x = torch.stack([data[s:s+block_size] for s in batch_starts]).to(DEVICE)
            y = torch.stack([data[s+1:s+block_size+1] for s in batch_starts]).to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item() * len(batch_starts)
            n += len(batch_starts)
    model.train()
    avg_loss = total_loss / n
    return avg_loss, math.exp(avg_loss)

def run_trial(cfg, vocab_size, train_data, val_data, trial_steps):
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    )
    model = Transformer(model_cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    model.train()
    t0 = time.time()
    for step in range(trial_steps):
        xb, yb = get_batch(train_data, BLOCK_SIZE, cfg["batch_size"])
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss, val_ppl = eval_windows(model, val_data, BLOCK_SIZE, cfg["batch_size"])
    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return {
        "config": cfg,
        "n_params_M": round(n_params, 2),
        "val_loss": round(val_loss, 4),
        "val_ppl": round(val_ppl, 2),
        "seconds": round(elapsed, 1),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=8)
    p.add_argument("--trial_steps", type=int, default=200,
                    help="Training steps per trial. ~200 steps keeps each trial "
                         "under ~10-15 min on CPU for these model sizes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="search_results.json")
    args = p.parse_args()

    rng = random.Random(args.seed)

    dh = DataHandler()
    vocab_size = dh.prepare_tensors()
    train_data = torch.load("data/train.pt")
    val_data = torch.load("data/val.pt")

    results = []
    print(f"Running {args.n_trials} trials x {args.trial_steps} steps each...\n")

    for t in range(args.n_trials):
        cfg = sample_config(rng)
        print(f"[Trial {t+1}/{args.n_trials}] {cfg}")
        result = run_trial(cfg, vocab_size, train_data, val_data, args.trial_steps)
        results.append(result)
        print(f"  -> val_ppl={result['val_ppl']} val_loss={result['val_loss']} "
              f"params={result['n_params_M']}M time={result['seconds']}s\n")

    results.sort(key=lambda r: r["val_ppl"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    best = results[0]
    print("=" * 60)
    print("SEARCH COMPLETE — ranked by validation perplexity (lower = better)")
    print("=" * 60)
    for r in results:
        print(f"val_ppl={r['val_ppl']:>8} | val_loss={r['val_loss']:>7} | "
              f"{r['config']} | {r['n_params_M']}M | {r['seconds']}s")

    print("\nBest config:")
    print(json.dumps(best["config"], indent=2))

    c = best["config"]
    print("\nRun a full training with this config:")
    print(f"python3 train.py --lr {c['lr']} --n_layer {c['n_layer']} "
          f"--n_embd {c['n_embd']} --n_head {c['n_head']} "
          f"--dropout {c['dropout']} --batch_size {c['batch_size']} --epochs 5")

if __name__ == "__main__":
    main()
