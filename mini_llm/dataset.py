import torch

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, block_size=32):
        self.block_size = block_size
        self.tokenizer = tokenizer
        data = tokenizer.encode(text)
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# add to mini_llm/dataset.py

class FT8Dataset(torch.utils.data.Dataset):
    """
    Phase 2 dataset:
    - Loads entire FT8 corpus from a text file
    - Uses a byte-level tokenizer
    - Produces (x, y) next-token sequences
    """

    def __init__(self, corpus_path: str, tokenizer, block_size: int = 128):
        self.block_size = block_size
        self.tokenizer = tokenizer

        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = self.tokenizer.encode(text)
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
