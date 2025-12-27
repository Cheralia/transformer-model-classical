import torch
import math
import sys
from tokenizers import Tokenizer
from architecture.architecture import Transformer, ModelConfig  # Changed import

DEVICE = 'cpu'

def generate_text(model, tokenizer, start_text, max_new_tokens=200):
    """
    Generates text starting from a prompt.
    """
    model.eval()
    try:
        ids = tokenizer.encode(start_text).ids
    except Exception as e:
        print(f"Error encoding text: {e}")
        return ""

    if not ids:
        ids = [0] 
        
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= 128 else idx[:, -128:]
        
        logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] 
        
        logits = logits / 1.0 
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
        
    return tokenizer.decode(idx[0].tolist())

def main():
    print("Loading tokenizer and model...")
    try:
        tokenizer = Tokenizer.from_file("data/tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()
        
        # Init config and load weights
        config = ModelConfig(vocab_size=vocab_size, block_size=128)
        model = Transformer(config).to(DEVICE)  # Changed to Transformer
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Could not find 'model.pth' or 'data/tokenizer.json'. Did you run train.py?")
        sys.exit(1)
    
    try:
        test_data = torch.load("data/test.pt")
        model.eval()
        ix = torch.randint(len(test_data) - 128, (32,))
        x = torch.stack([test_data[i:i+128] for i in ix]).to(DEVICE)
        y = torch.stack([test_data[i+1:i+128+1] for i in ix]).to(DEVICE)
        with torch.no_grad():
            _, loss = model(x, y)
            ppl = math.exp(loss.item())
        print(f"Current Test Perplexity: {ppl:.2f}\n")
    except:
        print("Could not load test.pt, skipping perplexity check.\n")

    print("----------------------------------------------------------------")
    print("INTERACTIVE MODE")
    print("Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("----------------------------------------------------------------")

    while True:
        user_input = input("\nYour Prompt >> ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if not user_input.strip():
            continue

        print("\nGenerating...", end=" ", flush=True)
        generated = generate_text(model, tokenizer, user_input, max_new_tokens=200)
        
        print("\n--- RESULT ---")
        print(generated)
        print("--------------")

if __name__ == "__main__":
    main()