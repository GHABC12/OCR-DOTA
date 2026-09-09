"""Isolated three-risk diagnostic model for OCR-DOTA-V3.

This file deliberately lives outside ``OCR-DOTA-V3``.  It keeps V3's single
rank-compatibility calculation, while replacing the geometry correction by an
explicit, switchable sum of the three historical risks for factorial analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from ocr_dota_v3.model import OCRDOTAV3
from ocr_dota_v3.rank_compatibility import (
    compute_rank_compatibility,
    prediction_rank_prior,
    sample_update_compatibility,
)


@dataclass(frozen=True)
class RiskSwitches:
    residual: bool
    semantic: bool
    order: bool

    @property
    def key(self) -> str:
        return f"D{int(self.residual)}E{int(self.semantic)}O{int(self.order)}"


@dataclass(frozen=True)
class RiskStrengths:
    residual: float = 0.25
    semantic: float = 0.05
    order: float = 0.05

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value < 0.0:
                raise ValueError(f"{name} strength must be non-negative")


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def _anchors(model: OCRDOTAV3) -> torch.Tensor:
    text = _normalize(model.text_prototypes, model.rank_cfg.eps)
    means = _normalize(model.mu, model.rank_cfg.eps)
    return _normalize(
        (1.0 - model.anchor_beta) * text + model.anchor_beta * means,
        model.rank_cfg.eps,
    )


def three_risk_parts(
    model: OCRDOTAV3,
    z: torch.Tensor,
    switches: RiskSwitches | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute the three historical risks exactly, without labels."""

    eps = model.rank_cfg.eps
    values = _normalize(z.to(model.device), eps)
    anchors = _anchors(model)
    text = _normalize(model.text_prototypes, eps)
    cosine = (values @ anchors.t()).clamp(-1.0, 1.0)
    residual = (1.0 - cosine.square()).clamp_min(0.0)

    # Historical semantic leakage: maximum projection of the normalized
    # class residual onto a different text prototype.
    classes = int(cosine.shape[-1])
    zeros = torch.zeros_like(residual)
    if classes == 1:
        semantic = zeros
        order = zeros
    else:
        need_semantic = switches is None or switches.semantic
        need_order = switches is None or switches.order
        if need_semantic:
            residual_vectors = values.unsqueeze(1) - cosine.unsqueeze(-1) * anchors.unsqueeze(0)
            residual_vectors = _normalize(residual_vectors, eps)
            similarity = torch.einsum("bkd,jd->bkj", residual_vectors, text)
            diagonal = torch.eye(classes, device=similarity.device, dtype=torch.bool).unsqueeze(0)
            semantic = similarity.masked_fill(diagonal, float("-inf")).max(dim=-1).values
            semantic = torch.where(torch.isfinite(semantic), semantic, zeros).clamp_min(0.0)
        else:
            semantic = zeros
        if need_order:
            top_values, top_indices = cosine.topk(2, dim=-1)
            best = top_values[:, :1].expand(-1, classes)
            second = top_values[:, 1:2].expand(-1, classes)
            best_index = top_indices[:, :1].expand(-1, classes)
            class_index = torch.arange(classes, device=cosine.device).view(1, -1).expand_as(best_index)
            max_other = torch.where(best_index == class_index, second, best)
            order = (max_other - cosine).clamp_min(0.0)
        else:
            order = zeros
    return {"residual": residual, "semantic": semantic, "order": order}


class OCRDOTAV3RiskAblation(OCRDOTAV3):
    """V3 state with a frozen three-risk factorial diagnostic head."""

    def __init__(self, *args, switches: RiskSwitches, strengths: RiskStrengths, **kwargs):
        super().__init__(*args, **kwargs)
        strengths.validate()
        self.risk_switches = switches
        self.risk_strengths = strengths
        # The parent residual branch is replaced, not added a second time.
        self.residual_strength = 0.0

    def risk_adjusted_geometry(self, z: torch.Tensor, return_parts: bool = False):
        gaussian = self.gaussian_logits(z)
        parts = three_risk_parts(self, z, self.risk_switches)
        total = torch.zeros_like(gaussian)
        if self.risk_switches.residual:
            total = total + self.risk_strengths.residual * parts["residual"]
        if self.risk_switches.semantic:
            total = total + self.risk_strengths.semantic * parts["semantic"]
        if self.risk_switches.order:
            total = total + self.risk_strengths.order * parts["order"]
        logits = gaussian - total
        if return_parts:
            return logits, total, parts
        return logits

    def posterior_and_responsibility(self, z: torch.Tensor, return_parts: bool = False):
        z = z.to(self.device, dtype=torch.float32)
        gaussian = self.gaussian_logits(z)
        stable = self._stable_scores(z)
        compatibility, rank_parts = compute_rank_compatibility(
            stable,
            gaussian,
            tau_rank=self.rank_cfg.tau_rank,
            eps=self.rank_cfg.eps,
            return_parts=True,
        )
        geometry, total_risk, risks = self.risk_adjusted_geometry(z, return_parts=True)
        p_geometry = F.softmax(geometry, dim=-1)
        rank_logits, rank_log_prior = prediction_rank_prior(
            geometry,
            compatibility,
            prediction_strength=self.rank_cfg.prediction_strength,
            eps=self.rank_cfg.eps,
        )
        p_rank = F.softmax(rank_logits, dim=-1)
        sample_compatibility = sample_update_compatibility(p_geometry, compatibility)
        update_gate = sample_compatibility.pow(self.rank_cfg.update_power)
        result = {
            "dota_logits": gaussian,
            "geometry_logits": geometry,
            "rank_logits": rank_logits,
            "ocr_logits": rank_logits,
            "p_geometry": p_geometry,
            "p_rank": p_rank,
            "p_ocr": p_rank,
            "rank_compatibility": compatibility,
            "sample_compatibility": sample_compatibility,
            "update_gate": update_gate,
            "omega": update_gate * p_geometry,
            "total_risk": total_risk,
            "residual_magnitude": risks["residual"],
            "semantic_leakage": risks["semantic"],
            "order_violation": risks["order"],
        }
        if return_parts:
            result.update(rank_parts)
            result["rank_log_prior"] = rank_log_prior
        return result


__all__ = [
    "OCRDOTAV3RiskAblation",
    "RiskStrengths",
    "RiskSwitches",
    "three_risk_parts",
]
