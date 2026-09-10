#!/usr/bin/env python3
"""Efficient paired replay for the final V2 2x2 ablation.

``base_posterior`` shares one Original-DOTA state trajectory and computes both
predictions before one DOTA update. ``responsibility_full`` does the same for
the OCR responsibility trajectory. This gives four arm outputs from two
replays per dataset while preserving the online ordering.
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
from ocr_dota_paper.posterior_v2 import compute_ocr_posterior_v2  # noqa: E402
from ocr_dota_paper.responsibility_v2 import compute_clean_mass_v2, compute_ocr_responsibility_v2  # noqa: E402
from v2.protocol_v2 import TEN_DATASETS  # noqa: E402


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().contiguous().cpu().numpy()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _trace_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(core.canonical(list(value.shape)).encode()); digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _metrics(
    arrays: dict[str, np.ndarray], target: np.ndarray, ids: np.ndarray,
    trajectory_sha: str, state_sha: str, compatibility_sha: str,
    compare_prediction: np.ndarray | None = None,
) -> dict[str, Any]:
    # Keep the compact cache dtypes for the reproducibility hash.  Cast only
    # the working views to int64 for comparisons/metrics.
    hash_pred = arrays["prediction"].reshape(-1)
    hash_target = target.reshape(-1)
    pred = hash_pred.astype(np.int64, copy=False)
    target = hash_target.astype(np.int64, copy=False)
    correct = pred == target
    n = len(target)
    confidence = arrays["confidence"].reshape(-1)
    nll = arrays["nll"].reshape(-1)
    brier = arrays["brier"].reshape(-1)
    ece = 0.0
    for i in range(15):
        lo, hi = i / 15, (i + 1) / 15
        mask = (confidence >= lo) & ((confidence <= hi) if i == 14 else (confidence < hi))
        if mask.any(): ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    mass = arrays["update_mass"].reshape(-1)
    true_mass = arrays["true_class_weight"].reshape(-1)
    total = float(mass.sum()); true = float(true_mass.sum())
    if compare_prediction is None:
        compare_prediction = arrays["base_prediction"].reshape(-1).astype(np.int64, copy=False)
    compared_correct = compare_prediction == target
    wc = int((~compared_correct & correct).sum()); cw = int((compared_correct & ~correct).sum())
    rows = {k: np.asarray(v) for k, v in arrays.items()}
    rows.update(prediction=pred)
    return {
        "num_samples": int(n), "correct": int(correct.sum()),
        "accuracy": 100.0 * float(correct.mean()) if n else 0.0,
        "late50_accuracy": 100.0 * float(correct[n // 2:].mean()) if n else 0.0,
        "nll": float(nll.mean()) if n else 0.0, "brier": float(brier.mean()) if n else 0.0,
        "ece": float(ece), "wrm": 100.0 * (total - true) / max(total, 1e-30),
        "crm": 100.0 * true / max(total, 1e-30), "total_update_mass": total,
        "true_class_responsibility_mass": true, "wrong_class_responsibility_mass": total - true,
        "mean_update_gate": float(arrays["gate"].mean()),
        "prediction_sha256": core.prediction_sha(ids, hash_target, hash_pred),
        "trajectory_sha256": trajectory_sha, "state_sha256": state_sha,
        "compatibility_sha256": compatibility_sha, "analysis_trace_sha256": _trace_hash(rows),
        "health_status": "healthy", "temporal_accuracy": [100.0 * float(x.mean()) for x in np.array_split(correct, 10) if len(x)],
        "wrong_to_correct": wc, "correct_to_wrong": cw, "net_correction_vs_same_state_base": wc - cw,
        "changed_prediction_rate": 100.0 * float(np.mean(pred != compare_prediction)) if n else 0.0,
    }


def _empty_trace(classes: int) -> dict[str, list[np.ndarray]]:
    return {k: [] for k in ("prediction", "base_prediction", "confidence", "nll", "brier",
                            "update_mass", "true_class_weight", "gate")}


def replay_pair(
    data: dict[str, Any], base: dict[str, float], posterior_config: dict[str, Any],
    responsibility_config: dict[str, Any], pair: str, stop: Path, save_dir: Path | None = None,
) -> dict[str, Any]:
    if pair not in ("base_posterior", "responsibility_full"):
        raise ValueError(pair)
    dim, classes = map(int, data["clip_shape"])
    device = str(data["features"].device)
    state = legacy.LegacyState(base, dim, classes, device)
    text = data["text_prototypes"].to(device=device, dtype=torch.float32)
    if tuple(text.shape) == (dim, classes): text = text.t()
    if tuple(text.shape) != (classes, dim): raise ValueError("invalid text prototype shape")
    text = torch.nn.functional.normalize(text, dim=-1)
    count = int(data["features"].shape[0])
    ids = np.asarray(data["sample_ids"], dtype=np.int64)
    target_dtype = legacy.compact_dtype(classes)
    names = ("base", "posterior") if pair == "base_posterior" else ("responsibility", "full")
    traces = {name: _empty_trace(classes) for name in names}
    targets: list[np.ndarray] = []
    trajectory = hashlib.sha256(core.canonical(base).encode("utf-8")); compat_sha = hashlib.sha256()
    started = time.time()
    with torch.no_grad():
        for i in range(count):
            if stop.exists(): raise RuntimeError(f"STOP at {i}/{count}")
            views = data["features"][i].to(dtype=torch.float32)
            clip = data["clip_logits"][i:i + 1].to(dtype=torch.float32)
            p_zs = legacy.align_prob_map(data["prob_maps"][i].to(dtype=torch.float32), len(views))
            z = views.mean(0, keepdim=True)
            gaussian = state.scores(z, use_prior=False)
            stable = torch.nn.functional.normalize(z, dim=-1) @ text.t()
            relation = compute_ocr_compatibility(stable, gaussian, tau_rank=float(posterior_config.get("tau_rank", .15)))
            p_dota = torch.softmax(gaussian, -1)
            clean_mass = compute_clean_mass_v2(p_dota, relation["compatibility"])
            posterior = compute_ocr_posterior_v2(gaussian, relation["compatibility"],
                gamma=float(posterior_config.get("gamma", .1)), mode=posterior_config.get("posterior_mode", "direct_prior"),
                entropy_power=float(posterior_config.get("entropy_power", 1.0)))
            weight = torch.clamp(float(base["rho"]) * state.count.mean() / len(views), max=float(base["eta"]))
            base_final = clip + weight * gaussian
            full_final = clip + weight * posterior["ocr_logits"]
            if pair == "base_posterior":
                update = p_zs
            else:
                update = compute_ocr_responsibility_v2(
                    p_zs, relation["compatibility"], clean_mass,
                    mode=responsibility_config.get("mode", "gate_only"),
                    delta=float(responsibility_config.get("delta", 1.0)),
                    beta=float(responsibility_config.get("beta", 0.0)),
                )
            if not all(bool(torch.isfinite(v).all()) for v in (base_final, full_final, update, relation["compatibility"])):
                raise FloatingPointError(f"nonfinite at sample {i}")
            # Update must be complete before evaluation labels are accessed.
            state.fit(views, update); state.refresh_inverse()
            target = int(data["targets"][i].item()); targets.append(np.asarray(target, dtype=target_dtype))
            total_mass = update.sum(); true_mass = update[:, target].sum()
            gate = clean_mass.pow(float(responsibility_config.get("delta", 1.0))) if pair == "responsibility_full" else torch.ones((1,), device=update.device)
            for name, logits in ((names[0], base_final), (names[1], full_final)):
                prob = torch.softmax(logits, -1); pred = int(logits.argmax(-1).item())
                top = logits.topk(min(2, classes), -1).values
                t = traces[name]
                t["prediction"].append(np.asarray(pred, dtype=target_dtype)); t["base_prediction"].append(np.asarray(int(base_final.argmax()), dtype=target_dtype))
                t["confidence"].append(np.asarray(float(prob.max()), dtype=np.float32)); t["nll"].append(np.asarray(float(-torch.log_softmax(logits, -1)[0, target]), dtype=np.float32))
                t["brier"].append(np.asarray(float(prob.square().sum() - 2 * prob[0, target] + 1), dtype=np.float32)); t["update_mass"].append(np.asarray(float(total_mass), dtype=np.float32)); t["true_class_weight"].append(np.asarray(float(true_mass), dtype=np.float32)); t["gate"].append(np.asarray(float(gate.item()), dtype=np.float32))
            trajectory.update(i.to_bytes(8, "little")); trajectory.update(_array(update).tobytes(order="C")); compat_sha.update(i.to_bytes(8, "little")); compat_sha.update(_array(relation["compatibility"]).tobytes(order="C"))
    target = np.stack(targets).reshape(-1)
    result: dict[str, Any] = {"pair": pair, "num_samples": count, "elapsed_sec": time.time() - started}
    for name in names:
        arrays = {key: np.stack(value) for key, value in traces[name].items()}
        arrays["target"] = target; arrays["sample_id"] = ids
        # For the responsibility/full pair, the meaningful same-state
        # posterior contrast is Full versus Responsibility (not Full versus
        # the DOTA-only arm). Keep that reference in the Full trace.
        if pair == "responsibility_full" and name == "full":
            arrays["base_prediction"] = np.stack(traces["responsibility"]["prediction"])
        # Write trace before hashing so callers can verify both semantic and
        # file identities.
        if save_dir is not None:
            trace_path = save_dir / f"{name}.npz"; _atomic_npz(trace_path, **arrays)
        metric = _metrics(arrays, target, ids, trajectory.hexdigest(), legacy.state_sha256(state), compat_sha.hexdigest(),
                           arrays["base_prediction"] if name in ("posterior", "full") else None)
        if save_dir is not None:
            metric["trace_path"] = str(save_dir / f"{name}.npz")
            metric["trace_file_sha256"] = core.sha256_file(save_dir / f"{name}.npz")
        result[name] = metric
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=TEN_DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", choices=("base_posterior", "responsibility_full"), required=True)
    parser.add_argument("--posterior-mode", choices=("direct_prior", "entropy_gated_prior"), default="direct_prior")
    parser.add_argument("--gamma", type=float, default=.1)
    parser.add_argument("--delta", type=float, default=1.)
    parser.add_argument("--beta", type=float, default=0.)
    parser.add_argument("--responsibility-mode", choices=("gate_only", "calibrated_clip"), default="gate_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    cache = core.cache_path(args.dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT)
    data, _ = legacy.load_cache(cache, args.device, args.max_samples)
    base = core.load_original_base(REPO, args.dataset); core.setup_seed(1)
    config = {"posterior_mode": args.posterior_mode, "gamma": args.gamma, "entropy_power": 1., "tau_rank": .15}
    rconfig = {"mode": args.responsibility_mode, "delta": args.delta, "beta": args.beta}
    result = replay_pair(data, base, config, rconfig, args.pair, args.output / "STOP", args.output / "traces")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.dataset}_{args.pair}.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
