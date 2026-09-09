"""OCR-DOTA model with one rank compatibility and two consumers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from ocr_dota_model import OCRDOTA

from .rank_compatibility import (
    RankCompatibilityConfig,
    compute_rank_compatibility,
    prediction_rank_prior,
    sample_update_compatibility,
)


SCHEMA_VERSION = "ocr_dota_v3_model.v1"
REMOVED_RISK_KEYS = frozenset({"eta_e", "eta_o", "e_mode", "o_mode", "semantic_leakage"})


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(float(eps))


class OCRDOTAV3(OCRDOTA):
    """Reuse DOTA statistics while replacing the risk composition contract.

    There are only two signals:
      * residual magnitude: the retained geometric reliability term;
      * rank compatibility: computed once and used for prediction prior plus
        sample-level update gating.

    Semantic leakage and the previous independent order-risk branch are not
    accepted by the configuration schema.
    """

    def __init__(
        self,
        base_cfg: Mapping[str, Any],
        v3_cfg: Mapping[str, Any],
        input_shape: int,
        num_classes: int,
        text_prototypes: torch.Tensor,
        device: str | None = None,
    ) -> None:
        raw = v3_cfg.get("ocr_dota_v3", v3_cfg)
        update_cfg = deepcopy(dict(raw.get("update", {})))
        rank_cfg = raw.get("rank", {})
        forbidden = REMOVED_RISK_KEYS.intersection(raw) | REMOVED_RISK_KEYS.intersection(update_cfg)
        if forbidden:
            raise ValueError(f"removed risk keys are not supported: {sorted(forbidden)}")

        parsed_rank_cfg = RankCompatibilityConfig.from_mapping(rank_cfg)
        anchor_beta = float(update_cfg.pop("beta", 0.5))
        residual_strength = float(update_cfg.pop("residual_strength", 0.25))
        if not 0.0 <= anchor_beta <= 1.0:
            raise ValueError("update.beta must be in [0,1]")
        if residual_strength < 0.0:
            raise ValueError("update.residual_strength must be non-negative")

        # Parent class supplies stable sufficient-statistic state.  Its composite
        # risk path is disabled and never called by this subclass.
        parent_update = {
            **update_cfg,
            "beta": anchor_beta,
            "eta_e": 0.0,
            "eta_o": 0.0,
            "eta_d": 0.0,
            "e_mode": "off",
            "o_mode": "off",
            "delta": 1.0,
        }
        super().__init__(
            dict(base_cfg),
            {"ocr": parent_update},
            input_shape,
            num_classes,
            text_prototypes,
            device=device,
        )
        self.rank_cfg = parsed_rank_cfg
        self.anchor_beta = anchor_beta
        self.residual_strength = residual_strength

    def _stable_scores(self, z: torch.Tensor) -> torch.Tensor:
        return _normalize(z.to(self.device), self.rank_cfg.eps) @ self.text_prototypes.t()

    def _residual_magnitude(self, z: torch.Tensor) -> torch.Tensor:
        text = self.text_prototypes
        means = _normalize(self.mu, self.rank_cfg.eps)
        anchors = _normalize(
            (1.0 - self.anchor_beta) * text + self.anchor_beta * means,
            self.rank_cfg.eps,
        )
        cosine = _normalize(z.to(self.device), self.rank_cfg.eps) @ anchors.t()
        return (1.0 - cosine.clamp(-1.0, 1.0).pow(2)).clamp_min(0.0)

    def posterior_and_responsibility(self, z: torch.Tensor, return_parts: bool = False) -> Dict[str, torch.Tensor]:
        z = z.to(self.device, dtype=torch.float32)
        gaussian_logits = self.gaussian_logits(z)
        stable_scores = self._stable_scores(z)
        compatibility, rank_parts = compute_rank_compatibility(
            stable_scores,
            gaussian_logits,
            tau_rank=self.rank_cfg.tau_rank,
            eps=self.rank_cfg.eps,
            return_parts=True,
        )

        residual = self._residual_magnitude(z)
        geometry_logits = gaussian_logits - self.residual_strength * residual
        p_geometry = F.softmax(geometry_logits, dim=-1)

        # Use 1: explicit prediction rank prior.
        rank_logits, rank_log_prior = prediction_rank_prior(
            geometry_logits,
            compatibility,
            prediction_strength=self.rank_cfg.prediction_strength,
            eps=self.rank_cfg.eps,
        )
        p_rank = F.softmax(rank_logits, dim=-1)

        # Use 2: the very same class compatibility determines whether this
        # sample is suitable for updating the distribution.
        sample_compatibility = sample_update_compatibility(p_geometry, compatibility)
        update_gate = sample_compatibility.pow(self.rank_cfg.update_power)
        responsibility = update_gate * p_geometry

        result: Dict[str, torch.Tensor] = {
            "dota_logits": gaussian_logits,
            "geometry_logits": geometry_logits,
            "rank_logits": rank_logits,
            "ocr_logits": rank_logits,
            "p_geometry": p_geometry,
            "p_rank": p_rank,
            "p_ocr": p_rank,
            "rank_compatibility": compatibility,
            "sample_compatibility": sample_compatibility,
            "update_gate": update_gate,
            "omega": responsibility,
        }
        if return_parts:
            result["rank_log_prior"] = rank_log_prior
            result["rank_displacement"] = rank_parts["rank_displacement"]
            result["stable_rank"] = rank_parts["stable_rank"]
            result["dynamic_rank"] = rank_parts["dynamic_rank"]
            result["residual_magnitude"] = residual
        return result

    def fused_prediction(
        self,
        z: torch.Tensor,
        clip_logits: torch.Tensor,
        fusion_weight: float | torch.Tensor,
        return_parts: bool = False,
    ):
        """Apply the rank prior inside the DOTA Gaussian branch."""

        parts = self.posterior_and_responsibility(z, return_parts=return_parts)
        weight = torch.as_tensor(fusion_weight, device=self.device, dtype=torch.float32)
        final_logits = clip_logits.to(self.device, dtype=torch.float32) + weight * parts["rank_logits"]
        if return_parts:
            return final_logits, parts
        return final_logits

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.posterior_and_responsibility(z)["rank_logits"]


__all__ = ["OCRDOTAV3", "REMOVED_RISK_KEYS", "SCHEMA_VERSION"]
