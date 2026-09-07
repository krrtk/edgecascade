import tiktoken


class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)