# mini_llm/sample_ft8_big.py

import torch

from mini_llm.tokenizer import ByteTokenizer
from mini_llm.model import TinyGPT
from mini_llm.config import Config


def sample(
    model,
    tokenizer,
    start_text,
    max_new_tokens,
    block_size,
    device,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
):
    model.eval()
    with torch.no_grad():
        idx = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1).squeeze(0)

            if top_k is not None:
                values, _ = torch.topk(probs, top_k)
                threshold = values.min()
                probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)

            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                mask = cumulative <= top_p
                mask[0] = True
                probs = torch.zeros_like(probs).scatter(0, sorted_idx[mask], sorted_probs[mask])

            probs = probs / probs.sum()

            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(idx[0].tolist())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    cfg = Config()
    cfg.vocab_size = 256
    cfg.block_size = 256
    cfg.n_embd = 256
    cfg.n_head = 4
    cfg.n_layer = 4
    cfg.dropout = 0.1

    tokenizer = ByteTokenizer()
    model = TinyGPT(cfg).to(device)

    state_dict = torch.load("phase4_ft8_big_model.pt", weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    for prompt in [
        "CQ AG6AQ ",
        "CQ DX ",
        "AG6AQ ",
        "CQ POTA ",
    ]:
        out = sample(
            model,
            tokenizer,
            start_text=prompt,
            max_new_tokens=80,
            block_size=cfg.block_size,
            device=device,
        )
        print("=" * 60)
        print("PROMPT:", repr(prompt))
        print(out)


if __name__ == "__main__":
    main()
