# mini_llm/tokenizer.py

class ByteTokenizer:
    """
    Simple byte-level tokenizer.
    - vocab_size is always 256 (0..255)
    - encode: str -> list[int] (bytes)
    - decode: list[int] -> str
    """

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        # UTF-8 encode string to bytes, then to list of ints
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: list[int]) -> str:
        # Convert list of ints back to bytes, then decode UTF-8
        return bytes(ids).decode("utf-8", errors="ignore")
