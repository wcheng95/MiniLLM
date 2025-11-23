import torch
from mini_llm.dataset import CharTokenizer
from mini_llm.model import TinyGPT
from mini_llm.config import Config

def sample(model, idx, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

def main():
    text = "hello world! " * 100
    tok = CharTokenizer(text)

    cfg = Config()
    cfg.vocab_size = tok.vocab_size

    model = TinyGPT(cfg).cuda()
    model.load_state_dict(torch.load("phase1_model.pt"))

    x = torch.tensor([[tok.stoi["h"]]], dtype=torch.long).cuda()
    out = sample(model, x, 100)[0].tolist()
    print(tok.decode(out))

if __name__ == "__main__":
    main()
