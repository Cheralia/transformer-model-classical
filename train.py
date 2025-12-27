import os
import time
import math
import logging
import torch
from tokenizer.tokenizer import DataHandler
from architecture.architecture import Transformer, ModelConfig 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 32
BLOCK_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 5 
DEVICE = 'cpu'

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def main():
    logger.info("Starting preparation...")
    
    dh = DataHandler()
    vocab_size = dh.prepare_tensors()
    train_data = torch.load("data/train.pt")
    val_data = torch.load("data/val.pt")
    
    config = ModelConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = Transformer(config).to(DEVICE)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Model initialized on {DEVICE}. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M Vocab Size: {vocab_size}")
    logger.info("Classical Transformer with learnable positional embeddings.")

    model.train()
    start_time = time.time()
    
    iters_per_epoch = len(train_data) // (BATCH_SIZE * BLOCK_SIZE)
    
    for epoch in range(EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        
        for i in range(iters_per_epoch):
            xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
            _, loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                perplexity = math.exp(loss.item())
                logger.info(f"Epoch {epoch+1} | Batch {i+1}/{iters_per_epoch} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")

        model.eval()
        with torch.no_grad():
            vx, vy = get_batch(val_data, BLOCK_SIZE, BATCH_SIZE)
            _, vloss = model(vx, vy)
            val_ppl = math.exp(vloss.item())
            logger.info(f"End of Epoch {epoch+1} VALIDATION | Loss: {vloss.item():.4f} | Perplexity: {val_ppl:.2f}")
        model.train()

    torch.save(model.state_dict(), "model.pth")
    logger.info(f"Training complete in {time.time()-start_time:.2f}s. Model saved to model.pth")

if __name__ == "__main__":
    main()