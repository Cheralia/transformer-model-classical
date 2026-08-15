import os
import time
import math
import logging
import argparse
import torch
from tokenizer.tokenizer import DataHandler
from architecture.architecture import Transformer, ModelConfig 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

DEVICE = 'cpu'

def parse_args():
    p = argparse.ArgumentParser(description="Train the classical Transformer.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_batches", type=int, default=5,
                    help="Number of random val batches averaged for the epoch-end validation metric (reduces noise vs a single batch).")
    p.add_argument("--out", type=str, default="model.pth", help="Path to save the trained model weights.")
    return p.parse_args()

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def evaluate_val(model, val_data, block_size, batch_size, n_batches):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            vx, vy = get_batch(val_data, block_size, batch_size)
            _, vloss = model(vx, vy)
            total += vloss.item()
    model.train()
    return total / n_batches

def main():
    args = parse_args()
    logger.info("Starting preparation...")
    
    dh = DataHandler()
    vocab_size = dh.prepare_tensors()
    train_data = torch.load("data/train.pt")
    val_data = torch.load("data/val.pt")
    
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = Transformer(config).to(DEVICE)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    logger.info(f"Model initialized on {DEVICE}. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M Vocab Size: {vocab_size}")
    logger.info(f"Config: {config}")
    logger.info(f"lr={args.lr} batch_size={args.batch_size} epochs={args.epochs}")

    model.train()
    start_time = time.time()
    
    iters_per_epoch = len(train_data) // (args.batch_size * args.block_size)
    best_val_loss = float("inf")
    best_epoch = -1
    
    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        for i in range(iters_per_epoch):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size)
            _, loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                perplexity = math.exp(loss.item())
                logger.info(f"Epoch {epoch+1} | Batch {i+1}/{iters_per_epoch} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")

        val_loss = evaluate_val(model, val_data, args.block_size, args.batch_size, args.val_batches)
        val_ppl = math.exp(val_loss)
        logger.info(f"End of Epoch {epoch+1} VALIDATION (avg over {args.val_batches} batches) | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_path = args.out.replace(".pth", "_best.pth") if args.out.endswith(".pth") else args.out + "_best"
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best val loss ({val_loss:.4f}) — checkpoint saved to {best_path}")

    torch.save(model.state_dict(), args.out)
    logger.info(f"Training complete in {time.time()-start_time:.2f}s. Final-epoch model saved to {args.out}")
    logger.info(f"Best model was at epoch {best_epoch} (val loss {best_val_loss:.4f}) — saved to {best_path}")

if __name__ == "__main__":
    main()
    # PR practice: added code to track and plot perplexity per epoch

