"""One normalized rank relation per sample, shared by both consumers."""
import torch

def normalized_descending_rank(scores):
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError('finite [B,K] scores required')
    ranks = torch.empty_like(scores, dtype=torch.float32)
    order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
    ranks.scatter_(1, order, torch.arange(scores.shape[-1], device=scores.device, dtype=torch.float32).expand_as(scores))
    return ranks / max(1, scores.shape[-1] - 1)

def compute_ocr_compatibility(stable_scores, dynamic_scores, tau_rank=.15):
    if tau_rank <= 0 or stable_scores.shape != dynamic_scores.shape:
        raise ValueError('positive tau and matching shapes required')
    stable_rank = normalized_descending_rank(stable_scores)
    dynamic_rank = normalized_descending_rank(dynamic_scores)
    displacement = (stable_rank - dynamic_rank).abs()
    compatibility = torch.exp(-displacement / tau_rank).clamp_min(torch.finfo(torch.float32).tiny)
    return dict(compatibility=compatibility, stable_rank=stable_rank, dynamic_rank=dynamic_rank, rank_displacement=displacement)
