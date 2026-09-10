"""Immutable protocol and identity helpers for the PaperCore V2 ten-set run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

PROTOCOL_VERSION = "paper-core-v2-ten-prereg-1"
TEN_DATASETS = (
    "fgvc", "caltech101", "stanford_cars", "dtd", "eurosat",
    "oxford_flowers", "food101", "oxford_pets", "sun397", "ucf101",
)
DEV_DATASETS = ("fgvc", "dtd", "eurosat", "oxford_pets", "ucf101")
EVAL_DATASETS = ("caltech101", "stanford_cars", "oxford_flowers", "food101", "sun397")

DIRECT_GAMMAS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.30)
ENTROPY_GAMMAS = (0.025, 0.05, 0.10, 0.20, 0.30)
RESPONSIBILITY_CANDIDATES = (
    {"mode": "gate_only", "delta": 0.5, "beta": 0.0},
    {"mode": "gate_only", "delta": 1.0, "beta": 0.0},
    {"mode": "calibrated_clip", "delta": 0.5, "beta": 0.5},
    {"mode": "calibrated_clip", "delta": 1.0, "beta": 0.5},
    {"mode": "calibrated_clip", "delta": 0.5, "beta": 1.0},
    {"mode": "calibrated_clip", "delta": 1.0, "beta": 1.0},
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def protocol_dict() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "datasets": list(TEN_DATASETS),
        "development": list(DEV_DATASETS),
        "evaluation": list(EVAL_DATASETS),
        "tau_rank": 0.15,
        "posterior": {
            "modes": ["direct_prior", "entropy_gated_prior"],
            "direct_gamma": list(DIRECT_GAMMAS),
            "entropy_gamma": list(ENTROPY_GAMMAS),
            "entropy_power": 1.0,
            "selection": "Dev-5 correct strictly > Base; macro >= Base; net correction > 0",
        },
        "responsibility": {
            "modes": ["dota", "gate_only", "calibrated_clip"],
            "candidates": list(RESPONSIBILITY_CANDIDATES),
            "screen_prefix": 1000,
            "selection": "Dev-5 macro, correct, WRM, late50; R1 preferred within 0.01 pp",
        },
        "base": {
            "policy": "Original DOTA epsilon/sigma/eta/rho from configs/vit YAML",
            "responsibility": "zero-shot CLIP probability",
            "state": "LegacyState exact DOTA FP32",
            "geometry": False,
        },
        "precision": "fp32",
        "global_seed": 1,
        "label_policy": "targets are read only after prediction and update; never enter adaptation",
        "historical_exposure": "V2 held-out only; all datasets were seen in historical research",
    }


def v2_config(
    posterior_mode: str = "direct_prior",
    gamma: float = 0.1,
    entropy_power: float = 1.0,
    responsibility_mode: str = "gate_only",
    delta: float = 1.0,
    beta: float = 0.0,
    tau_rank: float = 0.15,
) -> dict[str, Any]:
    if posterior_mode not in ("direct_prior", "entropy_gated_prior"):
        raise ValueError(posterior_mode)
    if responsibility_mode not in ("dota", "gate_only", "calibrated_clip"):
        raise ValueError(responsibility_mode)
    return {
        "tau_rank": float(tau_rank),
        "posterior_mode": posterior_mode,
        "gamma": float(gamma),
        "entropy_power": float(entropy_power),
        "responsibility_mode": responsibility_mode,
        "delta": float(delta),
        "beta": float(beta),
    }


def identity_payload(
    *, protocol: dict[str, Any], dataset: str, cache_path: Path,
    cache_sha256: str, order_sha256: str, code: dict[str, str], base: dict[str, float],
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "dataset": dataset,
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "order_sha256": order_sha256,
        "code_sha256": code,
        "base": base,
    }


__all__ = [
    "PROTOCOL_VERSION", "TEN_DATASETS", "DEV_DATASETS", "EVAL_DATASETS",
    "DIRECT_GAMMAS", "ENTROPY_GAMMAS", "RESPONSIBILITY_CANDIDATES",
    "canonical", "sha256_bytes", "sha256_file", "digest", "protocol_dict",
    "v2_config", "identity_payload",
]
