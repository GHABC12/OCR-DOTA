"""One explicit rank compatibility, consumed by prediction and update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F


SCHEMA_VERSION = "ocr_dota_v3_rank_compatibility.v1"


@dataclass(frozen=True)
class RankCompatibilityConfig:
    tau_rank: float = 0.15
    prediction_strength: float = 1.0
    update_power: float = 1.0
    eps: float = 1e-12

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RankCompatibilityConfig":
        cfg = value.get("rank", value)
        result = cls(
            tau_rank=float(cfg.get("tau_rank", 0.15)),
            prediction_strength=float(cfg.get("prediction_strength", 1.0)),
            update_power=float(cfg.get("update_power", 1.0)),
            eps=float(cfg.get("eps", 1e-12)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.tau_rank <= 0.0:
            raise ValueError("tau_rank must be positive")
        if self.prediction_strength < 0.0:
            raise ValueError("prediction_strength must be non-negative")
        if self.update_power < 0.0:
            raise ValueError("update_power must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def normalized_descending_rank(scores: torch.Tensor) -> torch.Tensor:
    """Return deterministic descending ranks normalized to [0,1]."""

    values = scores.float()
    if values.ndim != 2:
        raise ValueError("scores must be [B,K]")
    if not torch.isfinite(values).all():
        raise ValueError("scores contain NaN or infinity")
    batch, classes = values.shape
    if classes == 1:
        return torch.zeros_like(values)
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(values)
    ordinal = torch.arange(classes, device=values.device, dtype=values.dtype).view(1, -1)
    ranks.scatter_(1, order, ordinal.expand(batch, -1))
    return ranks / float(classes - 1)


def compute_rank_compatibility(
    stable_scores: torch.Tensor,
    dynamic_scores: torch.Tensor,
    *,
    tau_rank: float = 0.15,
    eps: float = 1e-12,
    return_parts: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compare stable semantic order with the current distribution order.

    This is O(B*K*logK), avoiding an O(B*K^2) pairwise-rank tensor on
    ImageNet-scale class sets.  A class receives compatibility one when its two
    normalized ranks match and decays exponentially with rank displacement.
    """

    stable = stable_scores.float()
    dynamic = dynamic_scores.to(device=stable.device, dtype=torch.float32)
    if stable.shape != dynamic.shape or stable.ndim != 2:
        raise ValueError("stable_scores and dynamic_scores must share [B,K] shape")
    if tau_rank <= 0.0 or eps <= 0.0:
        raise ValueError("tau_rank and eps must be positive")
    stable_rank = normalized_descending_rank(stable)
    dynamic_rank = normalized_descending_rank(dynamic)
    rank_displacement = (stable_rank - dynamic_rank).abs()
    compatibility = torch.exp(-rank_displacement / float(tau_rank)).clamp_min(float(eps))
    if return_parts:
        return compatibility, {
            "stable_rank": stable_rank,
            "dynamic_rank": dynamic_rank,
            "rank_displacement": rank_displacement,
        }
    return compatibility


def prediction_rank_prior(
    base_logits: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    prediction_strength: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Put the explicit rank prior back into prediction posterior."""

    base = base_logits.float()
    comp = compatibility.to(device=base.device, dtype=torch.float32)
    if base.shape != comp.shape or base.ndim != 2:
        raise ValueError("base_logits and compatibility must share [B,K] shape")
    if prediction_strength < 0.0:
        raise ValueError("prediction_strength must be non-negative")
    rank_log_prior = float(prediction_strength) * torch.log(comp.clamp_min(float(eps)))
    return base + rank_log_prior, rank_log_prior


def sample_update_compatibility(
    base_posterior: torch.Tensor,
    compatibility: torch.Tensor,
) -> torch.Tensor:
    """Aggregate the same class compatibility into one update-suitability gate."""

    posterior = base_posterior.float()
    comp = compatibility.to(device=posterior.device, dtype=torch.float32)
    if posterior.shape != comp.shape or posterior.ndim != 2:
        raise ValueError("base_posterior and compatibility must share [B,K] shape")
    if (comp <= 0.0).any() or (comp > 1.0 + 1e-6).any():
        raise ValueError("compatibility must be in (0,1]")
    return (posterior * comp).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)


def use_rank_compatibility(
    base_logits: torch.Tensor,
    compatibility: torch.Tensor,
    *,
    prediction_strength: float = 1.0,
    update_power: float = 1.0,
    update_posterior: torch.Tensor | None = None,
    eps: float = 1e-12,
    return_parts: bool = False,
):
    """Expose the two consumers of one already-computed compatibility tensor."""

    rank_logits, rank_log_prior = prediction_rank_prior(
        base_logits, compatibility, prediction_strength=prediction_strength, eps=eps
    )
    p_rank = F.softmax(rank_logits, dim=-1)
    p_update = F.softmax(base_logits.float(), dim=-1) if update_posterior is None else update_posterior.float()
    sample_compat = sample_update_compatibility(p_update, compatibility)
    update_gate = sample_compat.pow(float(update_power))
    if return_parts:
        return rank_logits, p_rank, update_gate, {
            "rank_log_prior": rank_log_prior,
            "sample_compatibility": sample_compat,
            "compatibility": compatibility,
        }
    return rank_logits, p_rank, update_gate


__all__ = [
    "SCHEMA_VERSION",
    "RankCompatibilityConfig",
    "compute_rank_compatibility",
    "normalized_descending_rank",
    "prediction_rank_prior",
    "sample_update_compatibility",
    "use_rank_compatibility",
]
