import torch


def cross_entropy(logits, targets):
    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    log_probs = logits - log_sum_exp
    n = logits.shape[0]

    return -log_probs[torch.arange(n, device=logits.device), targets].mean()


def gradient_clipping(parameters, max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.detach().mul_(clip_coef)
