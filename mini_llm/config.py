# Tiny config for Phase 1
class Config:
    vocab_size = None          # set after building tokenizer
    block_size = 32            # how many tokens of context
    n_embd = 64                # embedding dimension
    n_head = 1                 # single attention head
    n_layer = 1                # one transformer block
    dropout = 0.0              # keep things simple