"""V2 Gaussian responsibility interface.

Only the two preregistered responsibility mechanisms are exposed here:
sample-level clean-mass gating and calibrated CLIP allocation. The Base mode
is retained for exact degeneration tests.
"""
from __future__ import annotations

import torch


def compute_clean_mass_v2(p_dota: torch.Tensor, compatibility: torch.Tensor) -> torch.Tensor:
    if p_dota.shape != compatibility.shape:
        raise ValueError("p_dota and compatibility shapes must match")
    if not torch.isfinite(p_dota).all() or not torch.isfinite(compatibility).all():
        raise ValueError("responsibility inputs must be finite")
    tiny = torch.finfo(p_dota.dtype).tiny
    if bool((compatibility == 1).all()):
        return torch.ones_like(p_dota[..., :1])
    return (p_dota * compatibility).sum(-1, keepdim=True).clamp(min=tiny, max=1.0)


def compute_ocr_responsibility_v2(
    p_zs: torch.Tensor,
    compatibility: torch.Tensor,
    clean_mass: torch.Tensor,
    mode: str = "gate_only",
    delta: float = 1.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """Return per-view Gaussian responsibilities for V2 R0/R1/R2.

    ``p_zs`` may be one row and is broadcast to the number of views. The
    compatibility tensor is expected to be one class-level row (the same row
    computed from the sample's mean view) and is broadcast in exactly the same
    way. No target/label is accepted by this API.
    """
    if mode not in ("dota", "gate_only", "calibrated_clip"):
        raise ValueError(f"unsupported V2 responsibility mode: {mode}")
    if delta < 0 or beta < 0:
        raise ValueError("delta and beta must be non-negative")
    if p_zs.ndim == 1:
        p_zs = p_zs.unsqueeze(0)
    if compatibility.ndim == 1:
        compatibility = compatibility.unsqueeze(0)
    if p_zs.ndim != 2 or compatibility.ndim != 2 or p_zs.shape[-1] != compatibility.shape[-1]:
        raise ValueError("responsibility tensors must be [views/classes] and share classes")
    if clean_mass.numel() != 1:
        clean_mass = clean_mass.reshape(-1, 1)
    if clean_mass.shape[0] != 1:
        raise ValueError("clean_mass must be one sample-level scalar")
    views = p_zs.shape[0]
    c = compatibility[:1].expand(views, -1)
    m = clean_mass.to(dtype=p_zs.dtype).clamp(0, 1).pow(delta)
    if mode == "dota":
        return p_zs
    if mode == "gate_only":
        return m * p_zs
    tiny = torch.finfo(p_zs.dtype).tiny
    allocation = p_zs * c.clamp_min(tiny).pow(beta)
    allocation = allocation / allocation.sum(-1, keepdim=True).clamp_min(tiny)
    return m * allocation


__all__ = ["compute_clean_mass_v2", "compute_ocr_responsibility_v2"]
