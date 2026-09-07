# GPT-2 Small / 124M configuration
# Used for Phase 0 pretrained weight loading.

GPT2_124M_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}