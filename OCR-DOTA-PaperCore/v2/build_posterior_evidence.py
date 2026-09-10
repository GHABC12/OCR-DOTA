#!/usr/bin/env python3
"""Build one exact DOTA replay evidence cache per V2 dataset.

The replay is the Base trajectory: zero-shot CLIP probabilities are used for
the update and labels are read only after the update. All posterior candidates
are evaluated later by :mod:`evaluate_posterior_offline` without another
Gaussian replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
REPO = V2_ROOT.parent
sys.path.insert(0, str(REPO / "OCR-DOTA-V3"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V2_ROOT))

import tune_nonimagenet21 as core  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402
from ocr_dota_paper.compatibility import compute_ocr_compatibility  # noqa: E402
from ocr_dota_paper.responsibility_v2 import compute_clean_mass_v2  # noqa: E402
from v2.protocol_v2 import PROTOCOL_VERSION, TEN_DATASETS, sha256_bytes, sha256_file  # noqa: E402


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().contiguous().cpu().numpy()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _state_hash(state: Any) -> str:
    return legacy.state_sha256(state)


def _prediction_hash(ids: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> str:
    return core.prediction_sha(ids, targets, predictions)


def build_evidence(dataset: str, output: Path, device: str = "cuda", max_samples: int | None = None) -> dict[str, Any]:
    if dataset not in TEN_DATASETS:
        raise ValueError(f"V2 only permits the ten datasets; got {dataset}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    cache = core.cache_path(dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT)
    data, meta = legacy.load_cache(cache, device, max_samples)
    count = int(data["features"].shape[0])
    dim, classes = map(int, data["clip_shape"])
    base = core.load_original_base(REPO, dataset)
    out = output / "evidence" / f"{dataset}.npz"
    manifest_path = output / "evidence" / f"{dataset}.json"
    stop = output / "STOP"
    if stop.exists():
        raise RuntimeError(f"STOP marker exists: {stop}")
    started = time.time()

    # Text prototypes are the stable semantic branch. Normalize once, exactly
    # as the PaperCore model does.
    text = data["text_prototypes"].to(device=device, dtype=torch.float32)
    if tuple(text.shape) == (dim, classes):
        text = text.t()
    if tuple(text.shape) != (classes, dim):
        raise ValueError(f"unexpected text prototype shape {tuple(text.shape)}")
    text = torch.nn.functional.normalize(text, dim=-1)
    state = legacy.LegacyState(base, dim, classes, device)
    trajectory = hashlib.sha256(core.canonical(base).encode("utf-8"))
    compat_hash = hashlib.sha256()
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in (
        "sample_id", "target", "clip_logits", "gaussian_logits", "fusion_weight",
        "compatibility", "clean_mass", "base_prediction", "base_margin",
        "rank_displacement",
    )}
    with torch.no_grad():
        for i in range(count):
            if stop.exists():
                raise RuntimeError(f"STOP at {i}/{count}")
            views = data["features"][i].to(dtype=torch.float32)
            clip = data["clip_logits"][i:i + 1].to(dtype=torch.float32)
            p_zs = data["prob_maps"][i].to(dtype=torch.float32)
            z = views.mean(0, keepdim=True)
            gaussian = state.scores(z, use_prior=False)
            stable = torch.nn.functional.normalize(z, dim=-1) @ text.t()
            evidence = compute_ocr_compatibility(stable, gaussian, tau_rank=0.15)
            p_dota = torch.softmax(gaussian, dim=-1)
            clean_mass = compute_clean_mass_v2(p_dota, evidence["compatibility"])
            weight = torch.clamp(float(base["rho"]) * state.count.mean() / len(views), max=float(base["eta"]))
            final = clip + weight * gaussian
            aligned = legacy.align_prob_map(p_zs, len(views))
            if not all(bool(torch.isfinite(t).all()) for t in (gaussian, final, aligned, evidence["compatibility"])):
                raise FloatingPointError(f"non-finite evidence at sample {i}")

            # Online update happens before reading target. This ordering is a
            # hard guard against accidental label leakage into adaptation.
            state.fit(views, aligned)
            state.refresh_inverse()
            target = int(data["targets"][i].item())
            ids = int(data["sample_ids"][i])
            values = {
                "sample_id": np.asarray(ids, dtype=np.int64),
                "target": np.asarray(target, dtype=legacy.compact_dtype(classes)),
                "clip_logits": _array(clip[0]),
                "gaussian_logits": _array(gaussian[0]),
                "fusion_weight": np.asarray(float(weight.item()), dtype=np.float32),
                "compatibility": _array(evidence["compatibility"][0]),
                "clean_mass": np.asarray(float(clean_mass.item()), dtype=np.float32),
                "base_prediction": np.asarray(int(final.argmax(-1).item()), dtype=legacy.compact_dtype(classes)),
                "base_margin": np.asarray(float((final.topk(min(2, classes), -1).values[0, 0] - final.topk(min(2, classes), -1).values[0, -1]).item()), dtype=np.float32),
                "rank_displacement": _array(evidence["rank_displacement"][0]),
            }
            for key, value in values.items():
                arrays[key].append(value)
            trajectory.update(i.to_bytes(8, "little")); trajectory.update(_array(aligned).tobytes(order="C"))
            compat_hash.update(i.to_bytes(8, "little")); compat_hash.update(_array(evidence["compatibility"]).tobytes(order="C"))

    packed: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        packed[key] = np.stack(values, axis=0)
    packed["sample_id"] = packed["sample_id"].astype(np.int64, copy=False)
    packed["target"] = packed["target"].astype(legacy.compact_dtype(classes), copy=False)
    packed["base_prediction"] = packed["base_prediction"].astype(legacy.compact_dtype(classes), copy=False)
    predictions = packed["base_prediction"].reshape(-1)
    targets = packed["target"].reshape(-1)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "cache_path": str(cache),
        "cache_sha256": meta["sha256"],
        "full_count": meta["full_count"],
        "num_samples": count,
        "order_sha256": meta["order_sha256"],
        "full_order_sha256": meta["full_order_sha256"],
        "base": base,
        "tau_rank": 0.15,
        "global_seed": 1,
        "precision": "fp32",
        "base_prediction_sha256": _prediction_hash(packed["sample_id"], targets, predictions),
        "base_trajectory_sha256": trajectory.hexdigest(),
        "base_state_sha256": _state_hash(state),
        "compatibility_sha256": compat_hash.hexdigest(),
        "evidence_sha256": None,
        "created_at": time.time(),
        "elapsed_sec": time.time() - started,
        "label_policy": "target is read after state.fit; target is not used by prediction or update",
        "fields": sorted(packed),
    }
    _atomic_npz(out, **packed)
    metadata["evidence_sha256"] = sha256_file(out)
    _atomic_json(manifest_path, metadata)
    manifests = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in (output / "evidence").glob("*.json")
        if p.name != "posterior_evidence_manifest.json"
    }
    _atomic_json(output / "posterior_evidence_manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "datasets": {key: manifests[key] for key in sorted(manifests)},
    })
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=TEN_DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    result = build_evidence(args.dataset, args.output.resolve(), args.device, args.max_samples)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
