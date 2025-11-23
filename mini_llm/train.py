import torch
from torch.utils.data import DataLoader
from mini_llm.dataset import CharTokenizer, ToyDataset
from mini_llm.model import TinyGPT
from mini_llm.config import Config
import tqdm

def main():

    text = "hello world! " * 100  # toy corpus
    tok = CharTokenizer(text)
    
    cfg = Config()
    cfg.vocab_size = tok.vocab_size

    ds = ToyDataset(text, tok, block_size=cfg.block_size)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = TinyGPT(cfg).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(50):
        pbar = tqdm.tqdm(dl, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

    torch.save(model.state_dict(), "phase1_model.pt")

if __name__ == "__main__":
    main()
