"""Posterior consumers for OCR-DOTA PaperCore V2.

The V2 posterior deliberately uses the class-level rank compatibility directly.
The sample-level clean mass is reserved for the update responsibility consumer.
This module is independent from the V1 posterior implementation so that V1
results remain reproducible and cannot be silently reinterpreted.
"""
from __future__ import annotations

import torch


def _validate(g: torch.Tensor, c: torch.Tensor) -> None:
    if g.ndim < 1 or c.shape != g.shape:
        raise ValueError("gaussian_logits and compatibility must have matching shapes")
    if not torch.isfinite(g).all() or not torch.isfinite(c).all():
        raise ValueError("posterior inputs must be finite")
    if bool((c < 0).any()) or bool((c > 1).any()):
        raise ValueError("compatibility must lie in [0, 1]")


def compute_ocr_posterior_v2(
    gaussian_logits: torch.Tensor,
    compatibility: torch.Tensor,
    gamma: float = 0.1,
    mode: str = "direct_prior",
    entropy_power: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute a V2 OCR-corrected Gaussian posterior.

    Parameters follow the preregistered V2 protocol. ``direct_prior`` uses
    ``g + gamma * log(C)``. ``entropy_gated_prior`` multiplies the prior by
    normalized entropy of the DOTA posterior. The returned ``ocr_logits`` are
    logits whose softmax is ``p_ocr`` and can therefore be used directly in
    DOTA's existing ``clip_logits + fusion_weight * gaussian_logits`` fusion.

    ``gamma=0`` is an exact identity path (not merely an approximate one),
    which is important for Base/Posterior ablation contracts.
    """
    if mode not in ("direct_prior", "entropy_gated_prior"):
        raise ValueError(f"unsupported V2 posterior mode: {mode}")
    if gamma < 0 or entropy_power < 0:
        raise ValueError("gamma and entropy_power must be non-negative")
    _validate(gaussian_logits, compatibility)

    g = gaussian_logits
    p_dota = torch.softmax(g, dim=-1)
    tiny = torch.finfo(g.dtype).tiny
    log_c = torch.log(compatibility.clamp_min(tiny))
    # Keep entropy in the same shape as a class row with a singleton class
    # dimension. This works for both [K] and [B,K] inputs.
    k = g.shape[-1]
    entropy = -(p_dota * p_dota.clamp_min(tiny).log()).sum(-1, keepdim=True)
    log_k = torch.log(torch.tensor(float(max(k, 2)), device=g.device, dtype=g.dtype)).clamp_min(tiny)
    entropy_norm = (entropy / log_k).clamp(0.0, 1.0)

    if gamma == 0 or bool((compatibility == 1).all()):
        # Exact identity: preserve bitwise equality for the DOTA ablation.
        prediction_gate = torch.ones_like(entropy_norm)
        return {
            "p_dota": p_dota,
            "ocr_logits": g,
            "p_ocr": p_dota,
            "correction": torch.zeros_like(g),
            "prediction_gate": prediction_gate,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "prior": torch.zeros_like(g),
        }

    if mode == "direct_prior":
        prediction_gate = torch.ones_like(entropy_norm)
    else:
        prediction_gate = entropy_norm.pow(entropy_power)
    prior = gamma * log_c
    correction = prediction_gate * prior
    ocr_logits = g + correction
    p_ocr = torch.softmax(ocr_logits, dim=-1)
    return {
        "p_dota": p_dota,
        "ocr_logits": ocr_logits,
        "p_ocr": p_ocr,
        "correction": correction,
        "prediction_gate": prediction_gate,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "prior": prior,
    }


__all__ = ["compute_ocr_posterior_v2"]
