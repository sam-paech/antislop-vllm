# utils/sampler_helpers.py
import math
from typing import List, Optional, Tuple

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


def apply_sampling_filters(
    weighted_tokens: List[Tuple[str, float]],
    *,
    min_p: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
) -> List[Tuple[str, float]]:
    """Apply truncation filters to an unconstrained token distribution.

    Callers must apply bans and other post-sampling constraints *after* this
    function.  In particular, min-p's reference probability must come from
    the model's original top token, even when that token is later rejected.
    """
    total = sum(weight for _, weight in weighted_tokens)
    if total <= 0:
        return []

    pairs = [(token, weight / total) for token, weight in weighted_tokens]

    if min_p is not None and pairs:
        floor = min_p * max(prob for _, prob in pairs)
        pairs = [(token, prob) for token, prob in pairs if prob >= floor]

    if top_p is not None and pairs:
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        nucleus, cumulative = [], 0.0
        for token, prob in pairs:
            nucleus.append((token, prob))
            cumulative += prob
            if cumulative >= top_p:
                break
        pairs = nucleus

    if top_k is not None and len(pairs) > top_k:
        pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:top_k]

    return pairs

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
