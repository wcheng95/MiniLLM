# mini_llm/train_ft8.py

import torch
from torch.utils.data import DataLoader
import tqdm

from mini_llm.tokenizer import ByteTokenizer
from mini_llm.dataset import FT8Dataset
from mini_llm.model import TinyGPT
from mini_llm.config import Config


def main():
    corpus_path = "data/processed/ft8_corpus.txt"

    # --- Config for Phase 2 ---
    cfg = Config()
    cfg.vocab_size = 256         # byte-level
    cfg.block_size = 128         # context length
    cfg.n_embd = 128             # hidden size
    cfg.n_head = 4               # attention heads
    cfg.n_layer = 2              # transformer blocks
    cfg.dropout = 0.1

    tokenizer = ByteTokenizer()
    dataset = FT8Dataset(corpus_path, tokenizer, block_size=cfg.block_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    max_epochs = 3
    global_step = 0
    max_steps = 3000  # you can increase later if training is fast

    for epoch in range(max_epochs):
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            model.train()
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=float(loss))

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    torch.save(model.state_dict(), "phase2_ft8_model.pt")
    print("[INFO] Saved model to phase2_ft8_model.pt")


if __name__ == "__main__":
    main()
