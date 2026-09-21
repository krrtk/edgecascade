import torch


def generate_text_simple(
    model,
    idx,
    max_new_tokens,
    context_size,
    return_probs=False,
    temperature=1.0,
    top_k=None,
):
    log_probs = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        if temperature > 0.0 and temperature != 1.0:
            logits = logits / temperature
            
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probas = torch.softmax(logits, dim=-1)
        
        if top_k is not None or (temperature > 0.0 and temperature != 1.0):
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        if return_probs:
            token_prob = torch.gather(probas, -1, idx_next).squeeze(-1).item()
            import math
            log_probs.append(math.log(token_prob + 1e-10))

        idx = torch.cat((idx, idx_next), dim=1)

    if return_probs:
        return idx, log_probs
    return idx


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    device,
    return_probs=False,
    temperature=1.0,
    top_k=None,
):
    model.eval()
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    context_size = model.pos_emb.weight.shape[0]

    if return_probs:
        token_ids, log_probs = generate_text_simple(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            return_probs=True,
            temperature=temperature,
            top_k=top_k,
        )
        decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
        return decoded, log_probs
    else:
        token_ids = generate_text_simple(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            return_probs=False,
            temperature=temperature,
            top_k=top_k,
        )
        decoded = tokenizer.decode(token_ids.squeeze(0).tolist())
        return decoded