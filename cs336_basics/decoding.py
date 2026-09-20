import torch

from cs336_basics.nn import softmax


def sample_from_distribution(prob_dist: torch.Tensor, top_p: float | None = None) -> int:

    if top_p is not None and top_p < 1.0:
        
        sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        crossings = (cumsum > top_p).nonzero(as_tuple=False)
        keep = 1 if len(crossings) == 0 else crossings[0].item() + 1
        keep = max(keep, 1)  

        prob_dist = prob_dist.clone()
        keep_mask = torch.zeros_like(prob_dist, dtype=torch.bool)
        keep_mask[sorted_indices[:keep]] = True
        prob_dist[~keep_mask] = 0.0
        prob_dist = prob_dist / prob_dist.sum()

    return torch.multinomial(prob_dist, num_samples=1).item()


def _lookup_eos_id(tokenizer) -> int | None:
    special = getattr(tokenizer, "_special_token_ids", None)
    if special and "<|endoftext|>" in special:
        return special["<|endoftext|>"]
    return None


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    max_seq_len: int = 256,
    device: str = "cpu",
    suppress_eos: bool = False,
) -> list[int]:

    model.eval()
    eos_id = _lookup_eos_id(tokenizer)

    ids = list(tokenizer.encode(prompt))
    with torch.no_grad():
        for _ in range(max_tokens):
            context = ids[-max_seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)                      
            next_logits = logits[0, -1, :]         

            if suppress_eos and eos_id is not None:
                next_logits = next_logits.clone()
                next_logits[eos_id] = -float("inf")

            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = softmax(next_logits, dim=-1)

            next_token = sample_from_distribution(probs, top_p=top_p)
            ids.append(next_token)
            if not suppress_eos and eos_id is not None and next_token == eos_id:
                break
    return ids
