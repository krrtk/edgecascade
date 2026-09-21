import numpy as np
import torch

from models.transformer.config import GPT2_124M_CONFIG
from models.transformer.model import GPTModel


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            "Shape mismatch. "
            f"Left: {left.shape}, "
            f"Right: {right.shape}"
        )

    return torch.nn.Parameter(
        torch.tensor(
            right,
            dtype=left.dtype,
        )
    )


def load_weights_into_gpt(gpt, params):
    # Positional embeddings
    gpt.pos_emb.weight = assign(
        gpt.pos_emb.weight,
        params["wpe"],
    )

    # Token embeddings
    gpt.tok_emb.weight = assign(
        gpt.tok_emb.weight,
        params["wte"],
    )

    for b in range(len(params["blocks"])):

        # ------------------------------------------------
        # Attention: combined GPT-2 QKV -> separate Q/K/V
        # ------------------------------------------------

        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"],
            3,
            axis=-1,
        )

        gpt.trf_blocks[b].att.W_query.linear.weight = assign(
            gpt.trf_blocks[b].att.W_query.linear.weight,
            q_w.T,
        )

        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight,
            k_w.T,
        )

        gpt.trf_blocks[b].att.W_value.linear.weight = assign(
            gpt.trf_blocks[b].att.W_value.linear.weight,
            v_w.T,
        )

        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"],
            3,
            axis=-1,
        )

        gpt.trf_blocks[b].att.W_query.linear.bias = assign(
            gpt.trf_blocks[b].att.W_query.linear.bias,
            q_b,
        )

        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias,
            k_b,
        )

        gpt.trf_blocks[b].att.W_value.linear.bias = assign(
            gpt.trf_blocks[b].att.W_value.linear.bias,
            v_b,
        )

        # ------------------------------------------------
        # Attention output projection
        # ------------------------------------------------

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )

        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # ------------------------------------------------
        # Feed-forward network
        # ------------------------------------------------

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )

        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )

        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        # ------------------------------------------------
        # LayerNorm 1
        # ------------------------------------------------

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"],
        )

        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"],
        )

        # ------------------------------------------------
        # LayerNorm 2
        # ------------------------------------------------

        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"],
        )

        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"],
        )

    # ----------------------------------------------------
    # Final LayerNorm
    # ----------------------------------------------------

    gpt.final_norm.scale = assign(
        gpt.final_norm.scale,
        params["g"],
    )

    gpt.final_norm.shift = assign(
        gpt.final_norm.shift,
        params["b"],
    )

    # ----------------------------------------------------
    # Output head
    #
    # The notebook copies GPT-2 token embedding weights here.
    # ----------------------------------------------------

    gpt.out_head.weight = assign(
        gpt.out_head.weight,
        params["wte"],
    )


def load_pretrained_gpt2_124m(device="cpu"):
    from gpt_download3 import download_and_load_gpt2

    settings, params = download_and_load_gpt2(
        model_size="124M",
        models_dir="models/checkpoints/pretrained_gpt2",
    )

    model = GPTModel(GPT2_124M_CONFIG)

    load_weights_into_gpt(
        model,
        params,
    )

    model.to(device)
    model.eval()

    return model