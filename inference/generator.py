import torch


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(
            logits,
            dim=-1,
        )

        idx_next = torch.argmax(
            probas,
            dim=-1,
            keepdim=True,
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

    return idx


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    device,
):
    model.eval()

    encoded = tokenizer.encode(prompt)

    idx = torch.tensor(
        encoded,
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    context_size = model.pos_emb.weight.shape[0]

    token_ids = generate_text_simple(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
    )

    return tokenizer.decode(
        token_ids.squeeze(0).tolist()
    )