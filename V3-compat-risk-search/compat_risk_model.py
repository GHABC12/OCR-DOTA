"""Isolated compatibility-source variants for OCR-DOTA-V3."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from ocr_dota_v3.model import OCRDOTAV3
from ocr_dota_v3.rank_compatibility import (
    prediction_rank_prior,
    sample_update_compatibility,
)

SOURCE_COMPONENTS = {
    "D": ("D",), "E": ("E",), "O": ("O",),
    "DO": ("D", "O"), "EO": ("E", "O"),
}


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def historical_risks(model: OCRDOTAV3, z: torch.Tensor, needed: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Compute only requested historical classwise risks."""
    needed = set(needed)
    eps = model.rank_cfg.eps
    values = _normalize(z.to(model.device), eps)
    text = _normalize(model.text_prototypes, eps)
    means = _normalize(model.mu, eps)
    anchors = _normalize((1.0 - model.anchor_beta) * text + model.anchor_beta * means, eps)
    cosine = (values @ anchors.t()).clamp(-1.0, 1.0)
    out: Dict[str, torch.Tensor] = {}
    if "D" in needed:
        out["D"] = (1.0 - cosine.square()).clamp_min(0.0)
    classes = cosine.shape[-1]
    if "E" in needed:
        if classes == 1:
            out["E"] = torch.zeros_like(cosine)
        else:
            residual = _normalize(values.unsqueeze(1) - cosine.unsqueeze(-1) * anchors.unsqueeze(0), eps)
            similarity = torch.einsum("bkd,jd->bkj", residual, text)
            diagonal = torch.eye(classes, device=similarity.device, dtype=torch.bool).unsqueeze(0)
            out["E"] = similarity.masked_fill(diagonal, float("-inf")).max(-1).values.clamp(0.0, 1.0)
    if "O" in needed:
        if classes == 1:
            out["O"] = torch.zeros_like(cosine)
        else:
            top_values, top_indices = cosine.topk(2, dim=-1)
            best, second = top_values[:, :1], top_values[:, 1:2]
            index = torch.arange(classes, device=cosine.device).view(1, -1)
            max_other = torch.where(top_indices[:, :1] == index, second, best)
            out["O"] = 0.5 * (max_other - cosine).clamp_min(0.0)
    return out


class OCRDOTAV3CompatibilitySearch(OCRDOTAV3):
    """Keep V3 geometry intact and replace only compatibility source/consumers."""

    def __init__(self, *args, source: str, prediction_strength: float, update_power: float, **kwargs):
        super().__init__(*args, **kwargs)
        if source not in SOURCE_COMPONENTS:
            raise ValueError(f"unsupported source: {source}")
        self.compat_source = source
        self.compat_prediction_strength = float(prediction_strength)
        self.compat_update_power = float(update_power)

    def _compatibility(self, z: torch.Tensor, gaussian: torch.Tensor):
        components = SOURCE_COMPONENTS[self.compat_source]
        risks = historical_risks(self, z, components)
        combined_risk = torch.stack([risks[name] for name in components]).mean(0)
        compatibility = torch.exp(-combined_risk / 0.15).clamp_min(self.rank_cfg.eps)
        return compatibility, {"combined_risk": combined_risk, **{f"risk_{k}": v for k, v in risks.items()}}

    def posterior_and_responsibility(self, z: torch.Tensor, return_parts: bool = False):
        z = z.to(self.device, dtype=torch.float32)
        gaussian = self.gaussian_logits(z)
        residual = self._residual_magnitude(z)
        geometry = gaussian - self.residual_strength * residual
        p_geometry = F.softmax(geometry, dim=-1)
        compatibility, source_parts = self._compatibility(z, gaussian)
        rank_logits, rank_log_prior = prediction_rank_prior(
            geometry, compatibility,
            prediction_strength=self.compat_prediction_strength,
            eps=self.rank_cfg.eps,
        )
        sample_compat = sample_update_compatibility(p_geometry, compatibility)
        gate = sample_compat.pow(self.compat_update_power)
        result = {
            "dota_logits": gaussian, "geometry_logits": geometry,
            "rank_logits": rank_logits, "ocr_logits": rank_logits,
            "p_geometry": p_geometry, "p_rank": F.softmax(rank_logits, -1),
            "rank_compatibility": compatibility,
            "sample_compatibility": sample_compat, "update_gate": gate,
            "omega": gate * p_geometry, "residual_magnitude": residual,
        }
        if return_parts:
            result.update(source_parts)
            result["rank_log_prior"] = rank_log_prior
        return result


__all__ = ["OCRDOTAV3CompatibilitySearch", "SOURCE_COMPONENTS", "historical_risks"]
