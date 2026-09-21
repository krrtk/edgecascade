import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = linear
        self.lora_dropout = nn.Dropout(dropout)
        
        self.lora_A = nn.Parameter(torch.empty(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = alpha / rank
        
        # Freeze the underlying pretrained linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.scaling * (self.lora_dropout(x) @ self.lora_A @ self.lora_B)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, (
            "d_out must be divisible by num_heads"
        )

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = LoRALinear(
            nn.Linear(d_in, d_out, bias=qkv_bias),
            rank=8,
            alpha=16,
            dropout=0.05,
        )

        self.W_key = nn.Linear(
            d_in,
            d_out,
            bias=qkv_bias,
        )

        self.W_value = LoRALinear(
            nn.Linear(d_in, d_out, bias=qkv_bias),
            rank=8,
            alpha=16,
            dropout=0.05,
        )

        self.out_proj = nn.Linear(
            d_out,
            d_out,
        )

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1,
            ),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        values = values.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        queries = queries.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[
            :num_tokens,
            :num_tokens,
        ]

        attn_scores.masked_fill_(
            mask_bool,
            -torch.inf,
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )

        attn_weights = self.dropout(attn_weights)

        context_vec = (
            attn_weights @ values
        ).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b,
            num_tokens,
            self.d_out,
        )

        context_vec = self.out_proj(context_vec)

        return context_vec