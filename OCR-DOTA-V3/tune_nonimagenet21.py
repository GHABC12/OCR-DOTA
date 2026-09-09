#!/usr/bin/env python3
"""Strict full-stream online tuning for OCR-DOTA V3 on 21 non-ImageNet streams."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml


V3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V3_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V3_ROOT))

from ocr_dota_v3 import OCRDOTAV3  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402


VERSION = "ocr-dota-v3-nonimagenet21-full-online-v2"
DEFAULT_OUTPUT = V3_ROOT / "log" / "nonimagenet21_full_tuning_20260830"
ANCHOR_SOURCE = REPO_ROOT / "BEST_SINGLE_CONFIG_ALL41_AFTER_8_ROUNDS.json"
TEN_CACHE_ROOT = REPO_ROOT / "log/all_dataset_perf/cache"
DOMAIN_CACHE_ROOT = REPO_ROOT / "log/all_dataset_perf/officehome_visda_domainnet_dtd_style_20260803/cache"
EXPECTED_CACHE_MANIFEST = REPO_ROOT / "log/all_dataset_perf/m1v2_m2_two_module_ablation_21streams_20260822/cache_manifest.json"

TEN_DATASETS = (
    "fgvc", "caltech101", "stanford_cars", "dtd", "eurosat",
    "oxford_flowers", "food101", "oxford_pets", "sun397", "ucf101",
)
OFFICE_HOME = (
    "office_home_art", "office_home_clipart", "office_home_product",
    "office_home_real_world",
)
VISDA = ("visda2017_validation",)
DOMAINNET = (
    "domainnet_clipart", "domainnet_infograph", "domainnet_painting",
    "domainnet_quickdraw", "domainnet_real", "domainnet_sketch",
)
ALL_DATASETS = TEN_DATASETS + OFFICE_HOME + VISDA + DOMAINNET
ALIASES = {
    "aircraft": "fgvc", "cars": "stanford_cars", "flower102": "oxford_flowers",
    "flood101": "food101", "food101": "food101", "pets": "oxford_pets",
    "sun387": "sun397", "visda-2017": "visda2017_validation",
}
STAGES = ("initial", "round2", "round3")
UPDATE_RULES = ("clip", "mix08clip", "mix05", "omega")
PRIORITY_DATASETS = (
    "eurosat", "visda2017_validation", "domainnet_infograph",
    "domainnet_painting", "domainnet_quickdraw", "domainnet_real",
    "office_home_product",
)
HISTORICAL_SECONDS = {
    "fgvc": 21.7, "caltech101": 16.0, "stanford_cars": 67.1,
    "dtd": 9.7, "eurosat": 86.5, "oxford_flowers": 16.9,
    "food101": 195.3, "oxford_pets": 20.4, "sun397": 229.0,
    "ucf101": 26.0, "office_home_art": 27.2,
    "office_home_clipart": 47.4, "office_home_product": 49.2,
    "office_home_real_world": 48.8, "visda2017_validation": 603.7,
    "domainnet_clipart": 205.3, "domainnet_infograph": 220.9,
    "domainnet_painting": 308.5, "domainnet_quickdraw": 737.1,
    "domainnet_real": 740.1, "domainnet_sketch": 293.7,
}


class StopRequested(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


sha_value = stable_sha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("short append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class PidLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "PidLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            text = self.path.read_text(encoding="utf-8").strip() if self.path.exists() else ""
            if text.isdigit() and Path(f"/proc/{text}").exists():
                raise RuntimeError(f"live tuning process holds {self.path}: PID {text}")
            self.path.unlink(missing_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(self.fd, str(os.getpid()).encode("ascii"))
        os.fsync(self.fd)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_dataset(value: str) -> str:
    key = value.strip().lower()
    canonical_names = {name.lower(): name for name in ALL_DATASETS}
    if key in canonical_names:
        return canonical_names[key]
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(f"unsupported dataset: {value}")


def parse_datasets(value: str | None) -> tuple[str, ...]:
    if not value:
        return ALL_DATASETS
    result = tuple(normalize_dataset(item) for item in value.split(",") if item.strip())
    if len(set(result)) != len(result):
        raise ValueError("dataset list contains duplicates after alias normalization")
    return result


def check_stop(path: Path, context: str = "") -> None:
    if path.exists():
        suffix = f" ({context})" if context else ""
        raise StopRequested(f"STOP requested{suffix}")


def cache_path(dataset: str, ten_root: Path, domain_root: Path) -> Path:
    root = ten_root if dataset in TEN_DATASETS else domain_root
    path = root / f"{dataset}_vitb16.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_anchor_table(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    table = value.get("datasets", value)
    missing = set(ALL_DATASETS) - set(table)
    if missing:
        raise RuntimeError(f"anchor source lacks datasets: {sorted(missing)}")
    return table


def migrate_anchor(dataset: str, row: Mapping[str, Any]) -> dict[str, Any]:
    base = {key: float(row["base"][key]) for key in ("epsilon", "sigma", "eta", "rho")}
    old = row["ocr"]
    update = {
        "beta": float(old.get("beta", 0.5)),
        "residual_strength": 0.25,
        "stat_decay": float(old.get("stat_decay", 1.0)),
        "init_count": float(old.get("init_count", 1.0)),
        "init_mu": old.get("init_mu", "constant"),
        "prior_eps": float(old.get("prior_eps", 1e-6)),
        "norm_eps": float(old.get("norm_eps", 1e-12)),
        "gaussian_mode": old.get("gaussian_mode", "shared_cov"),
        "covariance_mode": old.get("covariance_mode", "full"),
    }
    config = {
        "base": base,
        "rank": {"tau_rank": 0.15, "prediction_strength": 1.0, "update_power": 1.0, "eps": 1e-12},
        "update": update,
        "update_rule": row.get("update_rule", "mix08clip"),
        "update_views": old.get("update_views", "selected"),
    }
    validate_config(config)
    return config


def load_original_base(repo_root: Path, dataset: str) -> dict[str, float]:
    path = repo_root / "configs/vit" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing original DOTA config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = {key: float(raw[key]) for key in ("epsilon", "sigma", "eta", "rho")}
    for key, value in base.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid original DOTA {dataset} {key}={value}")
    return base


def allocation_by_rule(rule: str, prob_map: torch.Tensor, p_geometry: torch.Tensor) -> torch.Tensor:
    """Choose class allocation before applying the shared sample compatibility gate."""
    prob = legacy.align_prob_map(prob_map, p_geometry.size(0))
    if rule == "clip":
        return prob
    if rule == "mix05":
        return 0.5 * prob + 0.5 * p_geometry
    if rule == "mix08clip":
        return 0.8 * prob + 0.2 * p_geometry
    if rule == "omega":
        return p_geometry
    raise ValueError(f"unsupported update rule: {rule}")


def validate_config(config: Mapping[str, Any]) -> None:
    base, rank, update = config["base"], config["rank"], config["update"]
    for key in ("epsilon", "sigma", "eta", "rho"):
        if not math.isfinite(float(base[key])) or float(base[key]) <= 0:
            raise ValueError(f"base.{key} must be finite and positive")
    if float(rank["tau_rank"]) < 0.03:
        raise ValueError("rank.tau_rank must be >= 0.03 to avoid compatibility-floor saturation")
    for key in ("prediction_strength", "update_power"):
        if not math.isfinite(float(rank[key])) or float(rank[key]) < 0:
            raise ValueError(f"rank.{key} must be finite and non-negative")
    if not 0 <= float(update["beta"]) <= 1:
        raise ValueError("update.beta must be in [0,1]")
    if float(update["residual_strength"]) < 0:
        raise ValueError("update.residual_strength must be non-negative")
    if float(update["init_count"]) <= 0:
        raise ValueError("update.init_count must be positive")
    if not 0 <= float(update["stat_decay"]) <= 1:
        raise ValueError("update.stat_decay must be in [0,1]")
    if update["init_mu"] not in {"text", "constant", "zero"}:
        raise ValueError("unsupported update.init_mu")
    if update["gaussian_mode"] not in {"shared_cov", "class_diag"}:
        raise ValueError("unsupported update.gaussian_mode")
    if update["covariance_mode"] not in {"full", "diag"}:
        raise ValueError("unsupported update.covariance_mode")
    if config["update_rule"] not in UPDATE_RULES:
        raise ValueError("unsupported update_rule")
    if config["update_views"] not in {"selected", "all", "mean", "prediction", "single"}:
        raise ValueError("unsupported update_views")


def candidate_fingerprint(dataset: str, config: Mapping[str, Any]) -> str:
    return stable_sha({"dataset": dataset, "config": config})


fingerprint = candidate_fingerprint


def changed(config: Mapping[str, Any], label: str, **changes: Any) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for path, value in changes.items():
        parts = path.split("__")
        target = result
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    validate_config(result)
    return {"label": label, "config": result}


def unique_candidates(dataset: str, specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for spec in specs:
        config = dict(spec["config"])
        config_fingerprint = fingerprint(dataset, config)
        if config_fingerprint in seen:
            continue
        seen.add(config_fingerprint)
        result.append({
            "candidate_id": spec.get("candidate_id", config_fingerprint[:16]),
            "fingerprint": config_fingerprint,
            "label": spec["label"],
            "config": config,
        })
    return result


def build_initial_candidates(anchor_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the preregistered A0-A7 candidates before dataset fingerprinting."""

    table = (
        ("A0", 0.15, 0.0, 0.0),
        ("A1", 0.15, 0.5, 0.0),
        ("A2", 0.15, 1.0, 0.0),
        ("A3", 0.15, 2.0, 0.0),
        ("A4", 0.075, 0.5, 1.0),
        ("A5", 0.15, 1.0, 1.0),
        ("A6", 0.30, 2.0, 1.0),
        ("A7", 0.15, 1.0, 2.0),
    )
    specs: list[dict[str, Any]] = []
    for candidate_id, tau, prediction_strength, update_power in table:
        spec = changed(
            anchor_config, candidate_id,
            rank__tau_rank=tau,
            rank__prediction_strength=prediction_strength,
            rank__update_power=update_power,
            update__residual_strength=0.25,
        )
        spec["candidate_id"] = candidate_id
        specs.append(spec)
    return specs


def generate_stage(dataset: str, stage: str, incumbent: Mapping[str, Any]) -> list[dict[str, Any]]:
    cfg = incumbent["config"] if "config" in incumbent else incumbent
    specs: list[dict[str, Any]] = []
    if stage == "initial":
        specs = build_initial_candidates(cfg)
    elif stage == "round2":
        beta0 = float(cfg["update"]["beta"])
        for residual in (0.0, 0.125, 0.5):
            specs.append(changed(cfg, f"R2_residual_{residual:g}", update__residual_strength=residual))
        for factor in (0.75, 1.25):
            specs.append(changed(
                cfg, f"R2_beta_x{factor:g}",
                update__beta=min(1.0, max(0.0, beta0 * factor)),
            ))
        count_factor = 0.75 if dataset in {"eurosat", "domainnet_quickdraw"} else 1.25
        specs.append(changed(
            cfg, f"R2_init_count_x{count_factor:g}",
            update__init_count=float(cfg["update"]["init_count"]) * count_factor,
        ))
    elif stage == "round3":
        for axis in ("eta", "epsilon", "sigma"):
            value = float(cfg["base"][axis])
            for factor in (0.90, 1.10):
                specs.append(changed(cfg, f"R3_{axis}_x{factor:g}", **{f"base__{axis}": value * factor}))
    else:
        raise ValueError(f"unsupported stage: {stage}")
    return unique_candidates(dataset, specs)


def build_run_identity(
    *, datasets: Iterable[str], stages: Iterable[str], global_seed: int,
    max_samples: int | None, anchor_source: Path, anchors: Mapping[str, Any],
    cache_records: Mapping[str, Any], code_sha256: Mapping[str, str],
    original_bases: Mapping[str, Any], time_budget_hours: float,
    verification_reserve_hours: float,
) -> dict[str, Any]:
    identity = {
        "version": VERSION, "formal": max_samples is None,
        "datasets": tuple(datasets), "stages": tuple(stages),
        "global_seed": int(global_seed), "max_samples": max_samples,
        "precision": "fp32", "worker_count": 1,
        "time_budget_hours": float(time_budget_hours),
        "verification_reserve_hours": float(verification_reserve_hours),
        "anchor_source": str(anchor_source),
        "anchor_source_sha256": sha256_file(anchor_source),
        "anchors_sha256": stable_sha(anchors), "original_bases": dict(original_bases),
        "code_sha256": dict(code_sha256),
        "caches": dict(cache_records),
    }
    return {"identity": identity, "identity_sha256": stable_sha(identity)}


def validate_resume_identity(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    if old.get("identity_sha256") != new.get("identity_sha256"):
        raise RuntimeError("resume identity mismatch")


def config_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    total = 0.0
    for section in ("base", "rank", "update"):
        keys = set(left[section]) | set(right[section])
        for key in keys:
            a, b = left[section].get(key), right[section].get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                a, b = float(a), float(b)
                if a > 0 and b > 0:
                    total += abs(math.log(a / b))
                else:
                    total += abs(a - b)
            else:
                total += float(a != b)
    total += float(left["update_rule"] != right["update_rule"])
    total += float(left["update_views"] != right["update_views"])
    return total


def select_best(rows: Iterable[Mapping[str, Any]], anchor: Mapping[str, Any]) -> dict[str, Any]:
    eligible = [dict(row) for row in rows if row.get("status") == "ok" and row.get("health_status") == "healthy"]
    if not eligible:
        raise RuntimeError("no healthy successful candidates")
    return min(eligible, key=lambda row: (
        -int(row["correct"]), config_distance(row["config"], anchor), row["candidate_id"],
    ))


def update_features(views: torch.Tensor, prediction_feature: torch.Tensor, mode: str) -> torch.Tensor:
    return views if mode in {"selected", "all"} else prediction_feature


def prediction_sha(sample_ids: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, value in (("sample_id", sample_ids), ("target", targets), ("prediction", predictions)):
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def state_sha(model: OCRDOTAV3) -> str:
    names = ["C", "S", "mu", "pi"]
    names += ["Q", "Sigma", "Lambda"] if model.covariance_mode == "full" else ["Q_diag", "Sigma_diag", "Lambda_diag"]
    digest = hashlib.sha256()
    for name in names:
        value = getattr(model, name).detach().contiguous().cpu().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def health(model: OCRDOTAV3) -> dict[str, Any]:
    tensors = [model.C, model.S, model.mu, model.pi]
    tensors += [model.Q, model.Sigma, model.Lambda] if model.covariance_mode == "full" else [model.Q_diag, model.Sigma_diag, model.Lambda_diag]
    finite = all(bool(torch.isfinite(value).all().item()) for value in tensors)
    stats = model.summary_stats()
    healthy = finite and stats["count_min"] > 0 and abs(float(model.pi.sum().cpu()) - 1.0) < 1e-4
    return {"finite": finite, "pi_sum": float(model.pi.sum().cpu()), **stats, "health_status": "healthy" if healthy else "unhealthy"}


def replay_v3(
    data: Mapping[str, Any], config: Mapping[str, Any], stop: Path,
    stop_interval: int, save_path: Path | None = None,
) -> dict[str, Any]:
    validate_config(config)
    device = str(data["features"].device)
    dim, classes = map(int, data["clip_shape"])
    model = OCRDOTAV3(config["base"], {"rank": config["rank"], "update": config["update"]}, dim, classes, data["text_prototypes"], device=device)
    model.eval()
    count = int(data["features"].shape[0])
    dtype = legacy.compact_dtype(classes)
    targets = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    sample_ids = np.asarray(data["sample_ids"], dtype=np.int64)
    predictions = np.empty(count, dtype=dtype)
    trajectory = hashlib.sha256(canonical(config).encode("utf-8"))
    compatibility_trajectory = hashlib.sha256()
    sums = {name: 0.0 for name in ("compatibility", "gate", "residual", "fusion_weight", "update_mass")}
    floor_hits = cap_hits = 0
    floor_total = 0
    started = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for index in range(count):
            if index % max(1, stop_interval) == 0 and stop.exists():
                raise StopRequested(f"STOP at sample {index}/{count}")
            views = data["features"][index].to(device=device, dtype=torch.float32)
            clip_logits = data["clip_logits"][index:index + 1].to(device=device, dtype=torch.float32)
            prob_map = data["prob_maps"][index].to(device=device, dtype=torch.float32)
            z = views.mean(0, keepdim=True)
            # Compute rank compatibility exactly once per sample.  Both the
            # prediction prior and update suitability consume these same parts.
            parts = model.posterior_and_responsibility(z, return_parts=True)
            pre_cap = float(config["base"]["rho"]) * model.C.mean() / views.size(0)
            weight = torch.clamp(pre_cap, max=float(config["base"]["eta"]))
            final = clip_logits + weight * parts["rank_logits"]
            if not bool(torch.isfinite(final).all().item()):
                raise FloatingPointError(f"non-finite prediction at sample {index}")
            predictions[index] = int(final.argmax(-1).item())
            cap_hits += int(bool((pre_cap >= float(config["base"]["eta"])).item()))

            features = update_features(views, z, config["update_views"])
            gaussian_update = model.gaussian_logits(features)
            residual_update = model._residual_magnitude(features)
            geometry_update = gaussian_update - model.residual_strength * residual_update
            p_geometry = F.softmax(geometry_update, dim=-1)
            allocation = allocation_by_rule(config["update_rule"], prob_map, p_geometry)
            gate = parts["update_gate"]
            weights = gate * allocation
            if not bool(torch.isfinite(weights).all().item()):
                raise FloatingPointError(f"non-finite update at sample {index}")
            trajectory.update(index.to_bytes(8, "little"))
            trajectory.update(weights.detach().contiguous().cpu().numpy().tobytes(order="C"))
            model.fit_ocr(features, weights)
            model.update()

            compatibility = parts["rank_compatibility"]
            compatibility_trajectory.update(index.to_bytes(8, "little"))
            compatibility_trajectory.update(
                compatibility.detach().contiguous().cpu().numpy().tobytes(order="C")
            )
            sums["compatibility"] += float(compatibility.mean().cpu())
            sums["gate"] += float(gate.mean().cpu())
            sums["residual"] += float(parts["residual_magnitude"].mean().cpu())
            sums["fusion_weight"] += float(weight.cpu())
            sums["update_mass"] += float(weights.sum(-1).mean().cpu())
            floor_hits += int((compatibility <= float(config["rank"]["eps"]) * 1.0001).sum().cpu())
            floor_total += compatibility.numel()
    state_health = health(model)
    correct = int((predictions == targets).sum())
    result = {
        "correct": correct, "num_samples": count, "accuracy": 100.0 * correct / count,
        "prediction_sha256": prediction_sha(sample_ids, targets, predictions),
        "trajectory_sha256": trajectory.hexdigest(), "state_sha256": state_sha(model),
        "compatibility_sha256": compatibility_trajectory.hexdigest(),
        "mean_rank_compatibility": sums["compatibility"] / count,
        "mean_update_gate": sums["gate"] / count,
        "mean_residual_magnitude": sums["residual"] / count,
        "mean_fusion_weight": sums["fusion_weight"] / count,
        "mean_update_mass": sums["update_mass"] / count,
        "compatibility_floor_rate": floor_hits / max(1, floor_total),
        "fusion_cap_rate": cap_hits / count, "state_health": state_health,
        "health_status": state_health["health_status"], "elapsed_sec": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0,
    }
    if save_path is not None:
        atomic_npz(save_path, sample_id=sample_ids, target=targets, prediction=predictions)
        result["predictions_path"] = str(save_path)
        result["predictions_file_sha256"] = sha256_file(save_path)
    return result


def replay_dota(
    data: Mapping[str, Any], base: Mapping[str, Any], stop: Path,
    stop_interval: int, save_path: Path | None = None,
) -> dict[str, Any]:
    device = str(data["features"].device)
    dim, classes = map(int, data["clip_shape"])
    state = legacy.LegacyState(base, dim, classes, device)
    count = int(data["features"].shape[0])
    dtype = legacy.compact_dtype(classes)
    targets = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    sample_ids = np.asarray(data["sample_ids"], dtype=np.int64)
    predictions = np.empty(count, dtype=dtype)
    trajectory = hashlib.sha256(canonical(base).encode("utf-8"))
    started = time.time()
    with torch.no_grad():
        for index in range(count):
            if index % max(1, stop_interval) == 0 and stop.exists():
                raise StopRequested(f"STOP at DOTA sample {index}/{count}")
            views = data["features"][index].to(device=device, dtype=torch.float32)
            clip_logits = data["clip_logits"][index:index + 1].to(device=device, dtype=torch.float32)
            prob_map = data["prob_maps"][index].to(device=device, dtype=torch.float32)
            z = views.mean(0, keepdim=True)
            gaussian = state.scores(z, use_prior=False)
            pre_cap = float(base["rho"]) * state.count.mean() / views.size(0)
            weight = torch.clamp(pre_cap, max=float(base["eta"]))
            final = clip_logits + weight * gaussian
            predictions[index] = int(final.argmax(-1).item())
            aligned = legacy.align_prob_map(prob_map, views.size(0))
            trajectory.update(index.to_bytes(8, "little"))
            trajectory.update(aligned.detach().contiguous().cpu().numpy().tobytes(order="C"))
            state.fit(views, aligned)
            state.refresh_inverse()
    correct = int((predictions == targets).sum())
    result = {
        "correct": correct, "num_samples": count, "accuracy": 100.0 * correct / count,
        "prediction_sha256": prediction_sha(sample_ids, targets, predictions),
        "trajectory_sha256": trajectory.hexdigest(), "state_sha256": legacy.state_sha256(state),
        "state_health": state.summary(), "health_status": "healthy", "elapsed_sec": time.time() - started,
    }
    if save_path is not None:
        atomic_npz(save_path, sample_id=sample_ids, target=targets, prediction=predictions)
        result["predictions_path"] = str(save_path)
        result["predictions_file_sha256"] = sha256_file(save_path)
    return result


def assert_reproduced(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    keys = ["correct", "num_samples", "prediction_sha256", "trajectory_sha256", "state_sha256"]
    if "compatibility_sha256" in left or "compatibility_sha256" in right:
        keys.append("compatibility_sha256")
    for key in keys:
        if left[key] != right[key]:
            raise RuntimeError(f"{label} cold replay mismatch for {key}: {left[key]} != {right[key]}")


def compact_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "correct", "num_samples", "accuracy", "prediction_sha256", "trajectory_sha256",
        "state_sha256", "compatibility_sha256", "health_status", "elapsed_sec",
    ) if key in row}


def write_summary(
    out: Path, datasets: tuple[str, ...], original: Mapping[str, Any],
    matched: Mapping[str, Any], winners: Mapping[str, Any], status: str,
) -> None:
    rows: dict[str, Any] = {}
    total_original = total_matched = total_winner = total_samples = 0
    for dataset in datasets:
        if dataset not in original or dataset not in matched or dataset not in winners:
            continue
        fixed, paired, winner = original[dataset], matched[dataset], winners[dataset]
        rows[dataset] = {
            "original_dota_fixed": fixed,
            "matched_base_dota": paired,
            "v3_winner": winner,
            "delta_vs_original_correct": int(winner["correct"]) - int(fixed["correct"]),
            "delta_vs_original_pp": float(winner["accuracy"]) - float(fixed["accuracy"]),
            "delta_vs_matched_correct": int(winner["correct"]) - int(paired["correct"]),
            "delta_vs_matched_pp": float(winner["accuracy"]) - float(paired["accuracy"]),
        }
        total_original += int(fixed["correct"])
        total_matched += int(paired["correct"])
        total_winner += int(winner["correct"])
        total_samples += int(winner["num_samples"])
    aggregate = None if not total_samples else {
        "num_samples": total_samples, "original_dota_correct": total_original,
        "matched_base_dota_correct": total_matched, "v3_correct": total_winner,
        "delta_vs_original_correct": total_winner - total_original,
        "delta_vs_matched_correct": total_winner - total_matched,
        "original_dota_accuracy": 100.0 * total_original / total_samples,
        "matched_base_dota_accuracy": 100.0 * total_matched / total_samples,
        "v3_accuracy": 100.0 * total_winner / total_samples,
        "delta_vs_original_pp": 100.0 * (total_winner - total_original) / total_samples,
        "delta_vs_matched_pp": 100.0 * (total_winner - total_matched) / total_samples,
    }
    payload = {
        "status": status,
        "protocol": "per-stream full-label tuning oracle; not an unbiased generalization estimate",
        "datasets": rows, "aggregate": aggregate,
    }
    atomic_json(out / "summary.json", payload)
    atomic_json(out / "winners.json", {"status": status, "datasets": winners})
    if rows:
        csv_path = out / "summary.csv"
        temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "N", "original_dota", "matched_base_dota", "v3", "delta_original", "delta_matched"))
            for dataset in datasets:
                if dataset not in rows:
                    continue
                row = rows[dataset]
                writer.writerow((dataset, row["v3_winner"]["num_samples"],
                    row["original_dota_fixed"]["accuracy"], row["matched_base_dota"]["accuracy"],
                    row["v3_winner"]["accuracy"], row["delta_vs_original_pp"], row["delta_vs_matched_pp"]))
        os.replace(temporary, csv_path)
        lines = ["OCR-DOTA-V3 非ImageNet 21流全量调参结果", "口径：per-stream full-label tuning oracle", ""]
        for dataset in datasets:
            if dataset in rows:
                row = rows[dataset]
                lines.append(
                    f"{dataset}: Original {row['original_dota_fixed']['accuracy']:.6f}% | "
                    f"Matched {row['matched_base_dota']['accuracy']:.6f}% | "
                    f"V3 {row['v3_winner']['accuracy']:.6f}% | "
                    f"ΔOriginal {row['delta_vs_original_pp']:+.6f}pp | "
                    f"ΔMatched {row['delta_vs_matched_pp']:+.6f}pp"
                )
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", help="comma-separated canonical names or supported aliases")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--global-seed", type=int, default=1)
    parser.add_argument("--stop-check-interval", type=int, default=25)
    parser.add_argument("--stop", help="STOP sentinel path; defaults to OUTPUT/STOP")
    parser.add_argument("--time-budget-hours", type=float, default=24.0)
    parser.add_argument("--verification-reserve-hours", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, help="smoke only; forbidden for formal completion")
    parser.add_argument("--dry-run", action="store_true", help="validate identities and candidate generation only")
    parser.add_argument("--anchor-source", default=str(ANCHOR_SOURCE))
    parser.add_argument("--ten-cache-root", default=str(TEN_CACHE_ROOT))
    parser.add_argument("--domain-cache-root", default=str(DOMAIN_CACHE_ROOT))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo_root).resolve()
    if repo != REPO_ROOT:
        raise RuntimeError(f"this sidecar is bound to parent repository {REPO_ROOT}, got {repo}")
    if args.time_budget_hours <= 0 or args.verification_reserve_hours <= 0:
        raise ValueError("time budgets must be positive")
    if args.verification_reserve_hours >= args.time_budget_hours:
        raise ValueError("verification reserve must be smaller than total budget")
    out = Path(args.output).resolve()
    datasets = parse_datasets(args.datasets)
    stages = tuple(value.strip() for value in args.stages.split(",") if value.strip())
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError(f"stages must be an ordered subset of {STAGES}")
    if tuple(STAGES.index(stage) for stage in stages) != tuple(sorted(STAGES.index(stage) for stage in stages)):
        raise ValueError("stages must preserve canonical order")
    anchor_source = Path(args.anchor_source).resolve()
    ten_root, domain_root = Path(args.ten_cache_root).resolve(), Path(args.domain_cache_root).resolve()
    anchors_raw = load_anchor_table(anchor_source)
    anchors = {dataset: migrate_anchor(dataset, anchors_raw[dataset]) for dataset in datasets}
    original_bases = {dataset: load_original_base(repo, dataset) for dataset in datasets}
    cache_paths = {dataset: cache_path(dataset, ten_root, domain_root) for dataset in datasets}
    expected_manifest = json.loads(EXPECTED_CACHE_MANIFEST.read_text(encoding="utf-8"))
    cache_records = {
        dataset: {
            "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "num_samples": int(expected_manifest[dataset]["num_samples"]),
            "order_sha256": expected_manifest[dataset]["order_sha256"],
        }
        for dataset, path in cache_paths.items()
    }
    for dataset, record in cache_records.items():
        if record["sha256"] != expected_manifest[dataset]["sha256"]:
            raise RuntimeError(f"cache SHA does not match expected manifest: {dataset}")
    identity_record = build_run_identity(
        datasets=datasets, stages=stages, global_seed=args.global_seed,
        max_samples=args.max_samples, anchor_source=anchor_source, anchors=anchors,
        cache_records=cache_records, original_bases=original_bases,
        time_budget_hours=args.time_budget_hours,
        verification_reserve_hours=args.verification_reserve_hours,
        code_sha256={
            "runner": sha256_file(Path(__file__).resolve()),
            "model": sha256_file(V3_ROOT / "ocr_dota_v3/model.py"),
            "rank_compatibility": sha256_file(V3_ROOT / "ocr_dota_v3/rank_compatibility.py"),
            "legacy_runner": sha256_file(REPO_ROOT / "scripts/run_cross_benchmark_ocr_ablation.py"),
            "expected_cache_manifest": sha256_file(EXPECTED_CACHE_MANIFEST),
        },
    )
    identity, identity_sha = identity_record["identity"], identity_record["identity_sha256"]
    if args.dry_run:
        plan = {
            dataset: {stage: len(generate_stage(dataset, stage, {"config": anchors[dataset]})) for stage in stages}
            for dataset in datasets
        }
        print(json.dumps({"identity_sha256": identity_sha, "candidate_counts_from_anchor": plan}, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    manifest_path, jobs_path = out / "manifest.json", out / "jobs.json"
    results_path, state_path = out / "candidate_results.jsonl", out / "state.json"
    stop = Path(args.stop).resolve() if args.stop else out / "STOP"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError("output exists; pass --resume to continue")
        validate_resume_identity(old, identity_record)
        started_at = float(old["started_at"])
        search_deadline = float(old["search_deadline"])
        total_deadline = float(old["total_deadline"])
    else:
        started_at = time.time()
        search_deadline = started_at + 3600.0 * (args.time_budget_hours - args.verification_reserve_hours)
        total_deadline = started_at + 3600.0 * args.time_budget_hours
        atomic_json(manifest_path, {
            "status": "running", "identity": identity, "identity_sha256": identity_sha,
            "started_at": started_at, "search_deadline": search_deadline,
            "total_deadline": total_deadline,
        })
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else {"version": VERSION, "datasets": {}}
    results = load_jsonl(results_path)
    done = {
        (row.get("dataset"), row.get("fingerprint")): row for row in results
        if row.get("run_identity_sha256") == identity_sha and row.get("status") == "ok"
    }
    baselines_path = out / "baselines.json"
    baseline_payload = json.loads(baselines_path.read_text(encoding="utf-8")) if baselines_path.exists() else {}
    original_baselines = baseline_payload.get("original_dota_fixed", {})
    matched_baselines = baseline_payload.get("matched_base_dota", {})
    winners: dict[str, Any] = {}

    def save_jobs() -> None:
        jobs["jobs_sha256"] = stable_sha(jobs.get("datasets", {}))
        atomic_json(jobs_path, jobs)

    def load_data(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
        setup_seed(args.global_seed)
        data, meta = legacy.load_cache(cache_paths[dataset], args.device, args.max_samples)
        expected = identity["caches"][dataset]
        if meta["sha256"] != expected["sha256"]:
            raise RuntimeError(f"cache changed after manifest creation: {dataset}")
        if args.max_samples is None:
            if int(meta["num_samples"]) != int(expected["num_samples"]):
                raise RuntimeError(f"sample count mismatch: {dataset}")
            if meta["order_sha256"] != expected["order_sha256"]:
                raise RuntimeError(f"sample order mismatch: {dataset}")
        return data, meta

    def estimate_seconds(dataset: str) -> float:
        observed = [float(row["elapsed_sec"]) for row in results
                    if row.get("dataset") == dataset and row.get("status") == "ok" and row.get("elapsed_sec")]
        return max(observed[-5:]) if observed else HISTORICAL_SECONDS[dataset] * 1.15

    def stage_order(stage: str, incumbents: Mapping[str, Any], stage_info: Mapping[str, Any]) -> list[str]:
        if stage == "initial":
            return list(datasets)
        eligible = []
        for dataset in datasets:
            initial = stage_info.get(dataset, {}).get("initial")
            if not initial or initial.get("status") != "complete" or int(initial.get("gain_correct", 0)) < 1:
                continue
            if stage == "round3":
                round2 = stage_info.get(dataset, {}).get("round2")
                if not round2 or round2.get("status") != "complete" or int(round2.get("gain_correct", 0)) < 1:
                    continue
            eligible.append(dataset)
        priority_index = {name: index for index, name in enumerate(PRIORITY_DATASETS)}
        return sorted(eligible, key=lambda name: (
            0 if name in priority_index else 1,
            priority_index.get(name, 999),
            -int(stage_info[name]["initial"].get("gain_correct", 0)),
            HISTORICAL_SECONDS[name], name,
        ))

    with PidLock(out / "RUNNING.pid"):
        try:
            stage_info: dict[str, dict[str, Any]] = {
                dataset: jobs.get("datasets", {}).get(dataset, {}).get("stage_results", {})
                for dataset in datasets
            }
            incumbents: dict[str, dict[str, Any]] = {dataset: {"config": anchors[dataset]} for dataset in datasets}
            # Restore already-completed adaptive winners in canonical stage order.
            for stage in stages:
                for dataset in datasets:
                    info = stage_info.get(dataset, {}).get(stage)
                    if info and info.get("status") == "complete":
                        key = (dataset, info["winner_fingerprint"])
                        if key not in done:
                            raise RuntimeError(f"completed stage lacks reusable winner: {dataset}/{stage}")
                        incumbents[dataset] = done[key]

            budget_exhausted = False
            for stage_index, stage in enumerate(stages):
                ordered = stage_order(stage, incumbents, stage_info)
                eligible_set = set(ordered)
                for dataset in datasets:
                    if stage != "initial" and dataset not in eligible_set:
                        jobs["datasets"].setdefault(dataset, {"stages": {}, "stage_results": {}})
                        jobs["datasets"][dataset]["stage_results"].setdefault(stage, {
                            "status": "skipped_no_integer_gain"
                        })
                save_jobs()
                for dataset_index, dataset in enumerate(ordered):
                    existing_info = jobs["datasets"].get(dataset, {}).get("stage_results", {}).get(stage)
                    if existing_info and existing_info.get("status") == "complete":
                        continue
                    check_stop(stop, f"before {dataset}/{stage}")
                    if time.time() + estimate_seconds(dataset) >= search_deadline:
                        budget_exhausted = True
                        jobs["datasets"].setdefault(dataset, {"stages": {}, "stage_results": {}})
                        jobs["datasets"][dataset]["stage_results"][stage] = {"status": "skipped_budget"}
                        save_jobs()
                        break
                    data, cache_meta = load_data(dataset)
                    dataset_dir = out / dataset
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    incumbent = incumbents[dataset]
                    dataset_jobs = jobs["datasets"].setdefault(dataset, {"cache": cache_meta, "stages": {}, "stage_results": {}})
                    dataset_jobs.setdefault("cache", cache_meta)
                    dataset_jobs.setdefault("stages", {})
                    dataset_jobs.setdefault("stage_results", {})
                    stage_block = dataset_jobs["stages"].get(stage)
                    if stage_block is None:
                        candidates = generate_stage(dataset, stage, incumbent)
                        stage_block = {
                            "incumbent_fingerprint": fingerprint(dataset, incumbent["config"]),
                            "candidates": candidates,
                        }
                        dataset_jobs["stages"][stage] = stage_block
                        save_jobs()
                    elif stage_block["incumbent_fingerprint"] != fingerprint(dataset, incumbent["config"]):
                        raise RuntimeError(f"adaptive job identity mismatch: {dataset}/{stage}")
                    candidates = stage_block["candidates"]
                    for candidate_index, candidate in enumerate(candidates):
                        key = (dataset, candidate["fingerprint"])
                        row = done.get(key)
                        if row is None:
                            if time.time() + estimate_seconds(dataset) >= search_deadline:
                                budget_exhausted = True
                                break
                            atomic_json(state_path, {
                                "status": "running", "dataset": dataset, "dataset_index": dataset_index,
                                "stage": stage, "stage_index": stage_index,
                                "candidate_index": candidate_index, "candidate_count": len(candidates),
                                "search_deadline": search_deadline, "updated_at": time.time(),
                            })
                            setup_seed(args.global_seed)
                            started = time.time()
                            try:
                                metrics = replay_v3(data, candidate["config"], stop, args.stop_check_interval)
                                status, error = "ok", None
                            except StopRequested:
                                raise
                            except Exception as exc:
                                status, error = "error", f"{type(exc).__name__}: {exc}"
                                metrics = {"elapsed_sec": time.time() - started, "health_status": "unhealthy"}
                            row = {
                                "status": status, "error": error, "run_identity_sha256": identity_sha,
                                "dataset": dataset, "stage": stage, **candidate,
                                "cache_sha256": cache_meta["sha256"], "order_sha256": cache_meta["order_sha256"],
                                "num_samples_expected": cache_meta["num_samples"],
                                "global_seed": args.global_seed, "precision": "fp32", **metrics,
                            }
                            append_jsonl(results_path, row)
                            results.append(row)
                            if status == "ok":
                                done[key] = row
                        elif (row.get("cache_sha256"), row.get("order_sha256"), row.get("global_seed")) != (cache_meta["sha256"], cache_meta["order_sha256"], args.global_seed):
                            raise RuntimeError(f"candidate resume identity mismatch: {dataset}/{candidate['candidate_id']}")
                    stage_rows = [done[(dataset, candidate["fingerprint"])] for candidate in candidates
                                  if (dataset, candidate["fingerprint"]) in done]
                    if len(stage_rows) != len(candidates):
                        dataset_jobs["stage_results"][stage] = {"status": "partial_budget" if budget_exhausted else "partial"}
                        save_jobs()
                        del data
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        break
                    stage_best = select_best(stage_rows, anchors[dataset])
                    if stage == "initial":
                        incumbent = stage_best
                        a0 = next(row for row in stage_rows if row["candidate_id"] == "A0")
                        gain = int(stage_best["correct"]) - int(a0["correct"])
                        reference = int(a0["correct"])
                    else:
                        previous = incumbent
                        incumbent = stage_best if int(stage_best["correct"]) > int(previous["correct"]) else previous
                        gain = int(incumbent["correct"]) - int(previous["correct"])
                        reference = int(previous["correct"])
                    incumbents[dataset] = incumbent
                    info = {
                        "status": "complete", "reference_correct": reference,
                        "winner_correct": int(incumbent["correct"]), "gain_correct": gain,
                        "winner_fingerprint": incumbent["fingerprint"],
                    }
                    dataset_jobs["stage_results"][stage] = info
                    stage_info.setdefault(dataset, {})[stage] = info
                    save_jobs()
                    del data
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                if budget_exhausted:
                    break

            # Verification and the two DOTA baselines use the protected reserve.
            final_datasets = [dataset for dataset in datasets
                              if stage_info.get(dataset, {}).get("initial", {}).get("status") == "complete"]
            for dataset in final_datasets:
                check_stop(stop, f"before verification {dataset}")
                data, cache_meta = load_data(dataset)
                dataset_dir = out / dataset
                dataset_dir.mkdir(parents=True, exist_ok=True)
                incumbent = incumbents[dataset]
                setup_seed(args.global_seed)
                verify = replay_v3(data, incumbent["config"], stop, args.stop_check_interval,
                                   dataset_dir / "winner_predictions_replay.npz")
                assert_reproduced(incumbent, verify, f"{dataset} winner")
                winner = {
                    **{key: incumbent[key] for key in ("candidate_id", "fingerprint", "stage", "label", "config")},
                    **compact_result(incumbent), "verification": compact_result(verify),
                    "cache_sha256": cache_meta["sha256"], "order_sha256": cache_meta["order_sha256"],
                }
                winners[dataset] = winner

                fixed_identity = stable_sha({"run": identity_sha, "dataset": dataset, "kind": "original", "base": original_bases[dataset]})
                fixed = original_baselines.get(dataset)
                if fixed is None:
                    setup_seed(args.global_seed)
                    fixed_run = replay_dota(data, original_bases[dataset], stop, args.stop_check_interval,
                                            dataset_dir / "original_dota_predictions.npz")
                    fixed = {"identity_sha256": fixed_identity, "base": original_bases[dataset], **compact_result(fixed_run)}
                    original_baselines[dataset] = fixed
                elif fixed.get("identity_sha256") != fixed_identity:
                    raise RuntimeError(f"original DOTA resume identity mismatch: {dataset}")

                matched_identity = stable_sha({"run": identity_sha, "dataset": dataset, "kind": "matched", "base": incumbent["config"]["base"]})
                paired = matched_baselines.get(dataset)
                if paired is None:
                    setup_seed(args.global_seed)
                    paired_run = replay_dota(data, incumbent["config"]["base"], stop, args.stop_check_interval,
                                             dataset_dir / "matched_base_dota_predictions.npz")
                    paired = {"identity_sha256": matched_identity, "base": incumbent["config"]["base"], **compact_result(paired_run)}
                    matched_baselines[dataset] = paired
                elif paired.get("identity_sha256") != matched_identity:
                    raise RuntimeError(f"matched DOTA resume identity mismatch: {dataset}")
                atomic_json(baselines_path, {
                    "status": "running", "original_dota_fixed": original_baselines,
                    "matched_base_dota": matched_baselines,
                })
                write_summary(out, datasets, original_baselines, matched_baselines, winners, "running")
                del data
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            formal_complete = args.max_samples is None and len(winners) == len(datasets)
            final_status = "complete_budget_limited" if formal_complete and budget_exhausted else "complete" if formal_complete else "partial"
            atomic_json(baselines_path, {
                "status": final_status, "original_dota_fixed": original_baselines,
                "matched_base_dota": matched_baselines,
            })
            write_summary(out, datasets, original_baselines, matched_baselines, winners, final_status)
            atomic_json(state_path, {
                "status": final_status, "datasets": len(winners), "budget_exhausted": budget_exhausted,
                "finished_at": time.time(), "total_deadline": total_deadline,
            })
            final_manifest = {
                "status": final_status, "identity": identity, "identity_sha256": identity_sha,
                "started_at": started_at, "search_deadline": search_deadline, "total_deadline": total_deadline,
                "jobs_sha256": sha256_file(jobs_path), "summary_sha256": sha256_file(out / "summary.json"),
            }
            atomic_json(manifest_path, final_manifest)
            if formal_complete and set(datasets) == set(ALL_DATASETS):
                best_payload = {"version": VERSION, "status": final_status, "datasets": winners,
                                "summary_sha256": final_manifest["summary_sha256"]}
                atomic_json(V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21.json", best_payload)
                lines = [f"{dataset}: {winners[dataset]['correct']}/{winners[dataset]['num_samples']} = {winners[dataset]['accuracy']:.6f}%"
                         for dataset in datasets]
                (V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            return 0
        except StopRequested as exc:
            atomic_json(state_path, {"status": "interrupted", "reason": str(exc), "updated_at": time.time()})
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
