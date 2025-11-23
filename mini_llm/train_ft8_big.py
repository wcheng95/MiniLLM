# mini_llm/train_ft8_big.py

import torch
from torch.utils.data import DataLoader
from mini_llm.tokenizer import ByteTokenizer
from mini_llm.dataset import FT8Dataset
from mini_llm.model import TinyGPT
from mini_llm.config import Config
import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1) Load corpus
    corpus_path = "data/processed/ft8_corpus.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[INFO] Loaded corpus from {corpus_path}, length={len(text)} chars")

    # 2) Tokenizer (byte-level)
    tokenizer = ByteTokenizer()

    # 3) Bigger model config
    cfg = Config()
    cfg.vocab_size = 256
    cfg.block_size = 256      # larger context
    cfg.n_embd = 256          # larger embedding
    cfg.n_head = 4            # you can try 8 later (256/8=32 per head)
    cfg.n_layer = 4           # deeper model
    cfg.dropout = 0.1

    print("[INFO] Model config:")
    print("  vocab_size =", cfg.vocab_size)
    print("  block_size =", cfg.block_size)
    print("  n_embd     =", cfg.n_embd)
    print("  n_head     =", cfg.n_head)
    print("  n_layer    =", cfg.n_layer)

    # 4) Dataset & DataLoader
    dataset = FT8Dataset("data/processed/ft8_corpus.txt",
                     tokenizer, block_size=cfg.block_size)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    # 5) Model
    model = TinyGPT(cfg).to(device)
    print("[INFO] Model parameters:",
          sum(p.numel() for p in model.parameters()) // 1000, "K params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    num_epochs = 5   # you can bump this later
    print("[INFO] Training for", num_epochs, "epochs")

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=float(loss))

        # quick checkpoint each epoch
        ckpt_name = f"phase4_ft8_big_model_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"[INFO] Saved checkpoint: {ckpt_name}")

    # final save
    torch.save(model.state_dict(), "phase4_ft8_big_model.pt")
    print("[INFO] Saved final model: phase4_ft8_big_model.pt")


if __name__ == "__main__":
    main()
