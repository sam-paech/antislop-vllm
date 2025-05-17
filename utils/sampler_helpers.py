# utils/sampler_helpers.py
import math
import torch

# ── probability helpers ───────────────────────────────────────────────
def _get_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)

def _apply_min_p_filter(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p is None:
        return probs
    keep = probs >= (probs.max() * min_p)
    if not torch.any(keep):
        keep[probs.argmax()] = True
    filt = probs * keep
    return filt / filt.sum()

# ── tail-selection (lowest-prob among top-k) ──────────────────────────
def select_tail_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    min_p: float,
    top_k: int,
    max_tokens: int,
) -> list[int]:
    """
    Return <= max_tokens ids: the *lowest*-probability survivors after
    (temperature → softmax → min-p → top-k).  Ordered from lowest→higher.
    """
    probs = _get_probs(logits, temperature)
    filt  = _apply_min_p_filter(probs, min_p)

    # keep only the k highest prob after filtering, then take the tail
    k = min(top_k, (filt > 0).sum().item())
    vals, idx = torch.topk(filt, k)                   # descending
    tail_ids  = idx.tolist()[::-1][:max_tokens]       # ascending prob
    return tail_ids
