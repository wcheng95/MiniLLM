import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb   = nn.Embedding(config.block_size, config.n_embd)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        # self-attention with causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).bool()
        attn_output, _ = self.attn(x, x, x, attn_mask=~mask)

        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.ff(x))

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        return logits, loss
