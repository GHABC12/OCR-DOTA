#!/usr/bin/env python3
"""Deterministic cache smoke test for the V3 online path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

V3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V3_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V3_ROOT))

from ocr_dota_v3 import OCRDOTAV3  # noqa: E402


def digest_tensors(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def replay(cache: dict, device: str, count: int) -> dict:
    text = cache["text_prototypes"].to(device=device, dtype=torch.float32)
    dim, classes = (int(x) for x in cache["clip_shape"])
    base = {"epsilon": 1e-4, "sigma": 0.002}
    config = {
        "rank": {"tau_rank": 0.15, "prediction_strength": 1.0, "update_power": 1.0},
        "update": {
            "beta": 0.5,
            "residual_strength": 0.25,
            "init_count": 1.0,
            "init_mu": "text",
            "stat_decay": 1.0,
            "covariance_mode": "diag",
            "gaussian_mode": "shared_cov",
        },
    }
    model = OCRDOTAV3(base, config, dim, classes, text, device=device)
    predictions = []
    targets = cache["targets"][:count].cpu().numpy().astype(np.int64, copy=False)
    trajectory = hashlib.sha256()
    compat = hashlib.sha256()
    with torch.no_grad():
        for index in range(count):
            views = cache["features"][index].to(device=device, dtype=torch.float32)
            z = views.mean(0, keepdim=True)
            clip_logits = cache["clip_logits"][index:index + 1].to(device=device, dtype=torch.float32)
            weight = torch.clamp(0.02 * model.C.mean() / views.shape[0], max=0.3)
            final, prediction_parts = model.fused_prediction(z, clip_logits, weight, return_parts=True)
            predictions.append(int(final.argmax(-1).item()))
            update_parts = model.posterior_and_responsibility(views, return_parts=True)
            omega = update_parts["omega"]
            # Both consumers record compatibility derived by the same function.
            compat.update(prediction_parts["rank_compatibility"].cpu().numpy().tobytes())
            compat.update(update_parts["rank_compatibility"].cpu().numpy().tobytes())
            trajectory.update(omega.cpu().numpy().tobytes())
            model.fit_ocr(views, omega)
            model.update()
    predictions_np = np.asarray(predictions, dtype=np.int64)
    return {
        "num_samples": count,
        "correct": int((predictions_np == targets).sum()),
        "prediction_sha256": hashlib.sha256(predictions_np.tobytes()).hexdigest(),
        "trajectory_sha256": trajectory.hexdigest(),
        "compatibility_sha256": compat.hexdigest(),
        "state_sha256": digest_tensors([model.C, model.S, model.mu, model.pi, model.Sigma_diag]),
        "count_min": float(model.C.min().cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    cache = torch.load(args.cache, map_location="cpu")
    count = min(int(args.max_samples), int(cache["targets"].shape[0]))
    first = replay(cache, args.device, count)
    second = replay(cache, args.device, count)
    for key in ("correct", "prediction_sha256", "trajectory_sha256", "compatibility_sha256", "state_sha256"):
        if first[key] != second[key]:
            raise RuntimeError(f"cold replay mismatch: {key}")
    if first["count_min"] <= 0:
        raise RuntimeError("unhealthy state: count_min <= 0")
    print(json.dumps({"status": "ok", "run1": first, "run2": second}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
