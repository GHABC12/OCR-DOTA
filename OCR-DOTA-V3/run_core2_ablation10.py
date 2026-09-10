#!/usr/bin/env python3
"""OCR-DOTA V3 two-core ablation for the ten non-ImageNet streams.

This is an isolated sidecar.  It imports the repository's existing cache and
model runners but never changes model.py, rank_compatibility.py or defaults.
Labels are consumed only after a complete online replay for post-hoc metrics.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

V3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V3_ROOT.parent
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V3_ROOT))

from ocr_dota_v3 import OCRDOTAV3  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402
import tune_nonimagenet21 as tune  # noqa: E402


VERSION = "ocr-dota-v3-core2-ablation10-v1"
DATASETS = (
    "fgvc", "caltech101", "stanford_cars", "dtd", "eurosat",
    "oxford_flowers", "food101", "oxford_pets", "sun397", "ucf101",
)
DISPLAY = {
    "fgvc": "Aircraft", "caltech101": "Caltech101", "stanford_cars": "Cars",
    "dtd": "DTD", "eurosat": "EuroSAT", "oxford_flowers": "Flower102",
    "food101": "Food101", "oxford_pets": "Pets", "sun397": "SUN397",
    "ucf101": "UCF101",
}
ALIASES = {"aircraft": "fgvc", "cars": "stanford_cars", "flower102": "oxford_flowers",
           "flood101": "food101", "pets": "oxford_pets", "sun387": "sun397"}
DEFAULT_RANK_FREE_SOURCE = REPO_ROOT / "V3-P0-ablation/results/nonimagenet21_p0_20260906/results.jsonl"
POSTERIOR_STRENGTHS = (0.0, 0.0075, 0.015, 0.0375, 0.075, 0.15, 0.30, 0.60, 1.0)
POSTERIOR_IDS = tuple(f"P{i}" for i in range(len(POSTERIOR_STRENGTHS)))
UPDATE_TAUS = (0.075, 0.15, 0.30, 0.60)
UPDATE_POWERS = (0.25, 0.50, 1.0, 2.0)
UPDATE_RULES = ("clip", "mix08clip", "mix05", "omega")


class StopRequested(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical(dict(row)) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(path.read_text(encoding="utf-8").splitlines()) - 1:
                raise
    return result


class PidLock:
    def __init__(self, path: Path): self.path, self.fd = path, None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            text = self.path.read_text().strip() if self.path.exists() else ""
            if text.isdigit() and Path(f"/proc/{text}").exists():
                raise RuntimeError(f"live process holds {self.path}: {text}")
            self.path.unlink(missing_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(self.fd, str(os.getpid()).encode()); os.fsync(self.fd)
        return self
    def __exit__(self, *_):
        if self.fd is not None: os.close(self.fd)
        self.path.unlink(missing_ok=True)


def setup_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def normalize_dataset(value: str) -> str:
    key = value.strip().lower()
    if key in {x.lower() for x in DATASETS}: return next(x for x in DATASETS if x.lower() == key)
    if key in ALIASES: return ALIASES[key]
    raise ValueError(f"unsupported dataset: {value}")


def parse_datasets(value: str | None) -> tuple[str, ...]:
    if not value: return DATASETS
    out = tuple(normalize_dataset(x) for x in value.split(",") if x.strip())
    if len(set(out)) != len(out): raise ValueError("duplicate dataset")
    return out


def check_stop(path: Path, context: str = "") -> None:
    if path.exists(): raise StopRequested(f"STOP requested {context}")


def cache_path(repo: Path, dataset: str) -> Path:
    root = repo / "log/all_dataset_perf/cache"
    path = root / f"{dataset}_vitb16.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_anchor_table(anchor_path: Path) -> tuple[dict[str, Any], str]:
    """Return dataset records and a source-kind label.

    The P0 JSONL is the authoritative Rank-free Base source.  The historical
    BEST JSON is accepted only as an explicit fallback for missing P0 rows.
    """
    if anchor_path.suffix.lower() == ".jsonl":
        rows = load_jsonl(anchor_path)
        table = {r["dataset"]: r for r in rows if r.get("variant") == "rank_free" and r.get("status", "ok") == "ok"}
        return table, "rank_free_p0_jsonl"
    raw = json.loads(anchor_path.read_text(encoding="utf-8"))
    return raw.get("datasets", raw), "legacy_best_json"


def resolve_anchor(anchor_path: Path, datasets: Iterable[str], fallback_path: Path | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    table, source_kind = _read_anchor_table(anchor_path)
    fallback_table: dict[str, Any] = {}
    fallback_kind = ""
    if fallback_path is not None and fallback_path.exists():
        fallback_table, fallback_kind = _read_anchor_table(fallback_path)
    result = {}
    expected = {}
    sources = {}
    for dataset in datasets:
        if dataset in table:
            entry = table[dataset]; sources[dataset] = source_kind
        elif dataset in fallback_table:
            entry = fallback_table[dataset]; sources[dataset] = fallback_kind
        else:
            raise KeyError(f"anchor lacks {dataset} in primary and fallback")
        config = copy.deepcopy(entry.get("config", entry))
        if "rank" not in config: config["rank"] = {"tau_rank": 0.15, "prediction_strength": 0.0, "update_power": 0.0, "eps": 1e-12}
        config["rank"].setdefault("eps", 1e-12)
        config["rank"]["tau_rank"] = 0.15
        config["update_rule"] = config.get("update_rule", "mix08clip")
        config["update_views"] = config.get("update_views", "selected")
        tune.validate_config(config)
        result[dataset] = config
        expected[dataset] = dict(entry)
    return result, expected, sources


def candidate_fingerprint(dataset: str, config: Mapping[str, Any], phase: str) -> str:
    return stable_sha({"dataset": dataset, "phase": phase, "config": config})


def make_config(anchor: Mapping[str, Any], *, prediction_strength: float | None = None,
                tau: float | None = None, power: float | None = None,
                update_rule: str | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(anchor))
    cfg["rank"] = dict(cfg["rank"])
    if prediction_strength is not None: cfg["rank"]["prediction_strength"] = float(prediction_strength)
    if tau is not None: cfg["rank"]["tau_rank"] = float(tau)
    if power is not None: cfg["rank"]["update_power"] = float(power)
    if update_rule is not None: cfg["update_rule"] = update_rule
    tune.validate_config(cfg)
    return cfg


def model_state_sha(model: OCRDOTAV3) -> str:
    names = ["C", "S", "mu", "pi"]
    names += ["Q", "Sigma", "Lambda"] if model.covariance_mode == "full" else ["Q_diag", "Sigma_diag", "Lambda_diag"]
    h = hashlib.sha256()
    for name in names:
        value = getattr(model, name).detach().contiguous().cpu().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(canonical(list(value.shape)).encode()); h.update(value.tobytes())
    return h.hexdigest()


def pred_sha(ids: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, value in (("sample_id", ids), ("target", labels), ("prediction", preds)):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(canonical(list(value.shape)).encode()); h.update(value.tobytes())
    return h.hexdigest()


def _accuracy(pred: np.ndarray, target: np.ndarray, start: int = 0) -> tuple[int, float]:
    correct = int((pred[start:] == target[start:]).sum()); total = int(len(pred) - start)
    return correct, 100.0 * correct / max(1, total)


def _quartile_rows(base_pred: np.ndarray, pred: np.ndarray, target: np.ndarray, compat: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(compat, kind="stable")
    buckets = np.array_split(order, 4)
    rows = []
    for qi, idx in enumerate(buckets, 1):
        b = base_pred[idx] == target[idx]; c = pred[idx] == target[idx]
        rows.append({"quartile": f"Q{qi}", "num_samples": int(len(idx)), "base_correct": int(b.sum()),
                     "posterior_correct": int(c.sum()), "base_accuracy": float(100*b.mean()) if len(idx) else 0.0,
                     "posterior_accuracy": float(100*c.mean()) if len(idx) else 0.0,
                     "delta_pp": float(100*(c.mean()-b.mean())) if len(idx) else 0.0,
                     "corrected": int((~b & c).sum()), "regressed": int((b & ~c).sum())})
    return rows


def _diagnostics(base_pred: np.ndarray, pred: np.ndarray, target: np.ndarray, compat: np.ndarray) -> dict[str, Any]:
    base_ok = base_pred == target; candidate_ok = pred == target
    corrected = int((~base_ok & candidate_ok).sum()); regressed = int((base_ok & ~candidate_ok).sum())
    assert int(candidate_ok.sum()) - int(base_ok.sum()) == corrected - regressed
    b = corrected; c = regressed
    if b + c == 0: stat, p = 0.0, 1.0
    else:
        stat = (abs(b-c)-1.0) ** 2 / float(b+c)
        p = math.erfc(math.sqrt(max(0.0, stat) / 2.0))
    return {"corrected": corrected, "regressed": regressed, "net_correction": corrected-regressed,
            "unchanged_correct": int((base_ok & candidate_ok).sum()), "unchanged_wrong": int((~base_ok & ~candidate_ok).sum()),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_stat": stat, "mcnemar_p": p,
            "compat_mean": float(compat.mean()), "compat_p10": float(np.percentile(compat, 10)),
            "compat_p50": float(np.percentile(compat, 50)), "compat_p90": float(np.percentile(compat, 90)),
            "quartiles": _quartile_rows(base_pred, pred, target, compat)}


def _write_npz(path: Path, ids: np.ndarray, target: np.ndarray, predictions: Mapping[str, np.ndarray], **diag: np.ndarray) -> str:
    arrays = {"sample_id": ids, "target": target}
    arrays.update({f"prediction_{k}": v for k, v in predictions.items()}); arrays.update(diag)
    atomic_npz(path, **arrays); return sha256_file(path)


def replay_posterior_grid(data: Mapping[str, Any], anchor: Mapping[str, Any], stop: Path,
                          stop_interval: int, strengths: tuple[float, ...], save_dir: Path | None = None) -> dict[str, Any]:
    """One online state trajectory with multiple posterior heads.

    update_power=0 makes gate exactly one, and allocation depends only on CLIP
    and geometry probabilities.  The assert below is also exercised by tests.
    """
    cfg0 = make_config(anchor, prediction_strength=0.0, power=0.0, tau=0.15)
    device = str(data["features"].device); dim, classes = map(int, data["clip_shape"])
    model = OCRDOTAV3(cfg0["base"], {"rank": cfg0["rank"], "update": cfg0["update"]}, dim, classes, data["text_prototypes"], device=device)
    model.eval(); count = int(data["features"].shape[0]); dtype = legacy.compact_dtype(classes)
    target = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False); ids = np.asarray(data["sample_ids"], dtype=np.int64)
    predictions = {f"P{i}": np.empty(count, dtype=dtype) for i in range(len(strengths))}
    compat_sample = np.empty(count, dtype=np.float32); stable_top = np.empty(count, dtype=np.int32); dynamic_top = np.empty(count, dtype=np.int32)
    trajectory = hashlib.sha256(canonical(cfg0).encode()); comp_sha = hashlib.sha256(); gate_sum = 0.0
    with torch.no_grad():
        for index in range(count):
            if index % max(1, stop_interval) == 0: check_stop(stop, f"posterior sample {index}")
            views = data["features"][index].to(device=device, dtype=torch.float32); z = views.mean(0, keepdim=True)
            clip_logits = data["clip_logits"][index:index+1].to(device=device, dtype=torch.float32)
            prob_map = data["prob_maps"][index].to(device=device, dtype=torch.float32)
            parts = model.posterior_and_responsibility(z, return_parts=True)
            pre_cap = float(cfg0["base"]["rho"]) * model.C.mean() / views.size(0)
            weight = torch.clamp(pre_cap, max=float(cfg0["base"]["eta"]))
            for pi, strength in enumerate(strengths):
                rank_cfg = dict(cfg0["rank"]); rank_cfg["prediction_strength"] = float(strength)
                rank_logits, _ = tune_rank_prior(parts["geometry_logits"], parts["rank_compatibility"], strength, rank_cfg["eps"])
                final = clip_logits + weight * rank_logits
                predictions[f"P{pi}"][index] = int(final.argmax(-1).item())
            c = parts["rank_compatibility"].mean(-1).item(); compat_sample[index] = c
            stable_top[index] = int(parts["stable_rank"].argmax(-1).item()); dynamic_top[index] = int(parts["dynamic_rank"].argmax(-1).item())
            comp_sha.update(index.to_bytes(8, "little")); comp_sha.update(parts["rank_compatibility"].detach().cpu().numpy().tobytes())
            features = views
            p_geometry = parts["p_geometry"]
            allocation = tune.allocation_by_rule(cfg0["update_rule"], prob_map, p_geometry)
            gate = parts["update_gate"]
            assert torch.allclose(gate, torch.ones_like(gate), atol=0.0, rtol=0.0), "power=0 must produce gate=1"
            weights = allocation
            trajectory.update(index.to_bytes(8, "little")); trajectory.update(weights.detach().cpu().numpy().tobytes())
            model.fit_ocr(features, weights); model.update(); gate_sum += float(gate.mean())
    result = {"num_samples": count, "target": target, "sample_id": ids, "predictions": predictions,
              "compat_sample": compat_sample, "stable_top1": stable_top, "dynamic_top1": dynamic_top,
              "trajectory_sha256": trajectory.hexdigest(), "compatibility_sha256": comp_sha.hexdigest(),
              "state_sha256": model_state_sha(model), "mean_update_gate": gate_sum / max(1, count),
              "health": tune.health(model), "model": model}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        result["npz_sha256"] = _write_npz(save_dir / "posterior_predictions.npz", ids, target, predictions,
                                           compat_sample=compat_sample, stable_top1=stable_top, dynamic_top1=dynamic_top)
    return result


def tune_rank_prior(geometry_logits: torch.Tensor, compatibility: torch.Tensor, strength: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the formula delegated to the canonical V3 implementation.
    from ocr_dota_v3.rank_compatibility import prediction_rank_prior
    return prediction_rank_prior(geometry_logits, compatibility, prediction_strength=float(strength), eps=float(eps))


def independent_replay_state_sha(data: Mapping[str, Any], anchor: Mapping[str, Any], strength: float, stop: Path, interval: int) -> str:
    cfg = make_config(anchor, prediction_strength=strength, power=0.0, tau=0.15)
    # Replaying through the shared path with one head is a strict independent
    # check for the posterior/no-state-coupling contract.
    return replay_posterior_grid(data, anchor, stop, interval, (strength,))["state_sha256"]


def replay_update(data: Mapping[str, Any], config: Mapping[str, Any], stop: Path, stop_interval: int,
                  save_path: Path | None = None) -> dict[str, Any]:
    tune.validate_config(config); device = str(data["features"].device); dim, classes = map(int, data["clip_shape"])
    model = OCRDOTAV3(config["base"], {"rank": config["rank"], "update": config["update"]}, dim, classes, data["text_prototypes"], device=device); model.eval()
    count = int(data["features"].shape[0]); dtype = legacy.compact_dtype(classes)
    target = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False); ids = np.asarray(data["sample_ids"], dtype=np.int64); pred = np.empty(count, dtype=dtype)
    gates = np.empty(count, dtype=np.float32); masses = np.empty(count, dtype=np.float32); trajectory = hashlib.sha256(canonical(config).encode())
    with torch.no_grad():
        for index in range(count):
            if index % max(1, stop_interval) == 0: check_stop(stop, f"update sample {index}")
            views = data["features"][index].to(device=device, dtype=torch.float32); z = views.mean(0, keepdim=True); clip_logits = data["clip_logits"][index:index+1].to(device=device, dtype=torch.float32); prob_map = data["prob_maps"][index].to(device=device, dtype=torch.float32)
            parts = model.posterior_and_responsibility(z, return_parts=True); pre_cap = float(config["base"]["rho"]) * model.C.mean() / views.size(0); weight = torch.clamp(pre_cap, max=float(config["base"]["eta"]))
            pred[index] = int((clip_logits + weight * parts["rank_logits"]).argmax(-1).item())
            allocation = tune.allocation_by_rule(config["update_rule"], prob_map, parts["p_geometry"]); weights = parts["update_gate"] * allocation
            gates[index] = float(parts["update_gate"].mean()); masses[index] = float(weights.sum(-1).mean()); trajectory.update(index.to_bytes(8, "little")); trajectory.update(weights.detach().cpu().numpy().tobytes())
            model.fit_ocr(views, weights); model.update()
    first_n = count // 2; all_correct, all_acc = _accuracy(pred, target); last_correct, last_acc = _accuracy(pred, target, first_n)
    result = {"num_samples": count, "correct": all_correct, "accuracy": all_acc, "last50_correct": last_correct, "last50_accuracy": last_acc, "first50_accuracy": _accuracy(pred, target, 0)[1] if first_n == count else 100.0 * int((pred[:first_n] == target[:first_n]).sum()) / max(1, first_n), "prediction_sha256": pred_sha(ids, target, pred), "trajectory_sha256": trajectory.hexdigest(), "state_sha256": model_state_sha(model), "health": tune.health(model), "mean_gate": float(gates.mean()), "median_gate": float(np.median(gates)), "gate_p10": float(np.percentile(gates, 10)), "gate_p25": float(np.percentile(gates, 25)), "gate_p75": float(np.percentile(gates, 75)), "gate_p90": float(np.percentile(gates, 90)), "mean_effective_update_mass": float(masses.mean()), "predictions": pred, "target": target, "sample_id": ids}
    if save_path is not None:
        atomic_npz(save_path, sample_id=ids, target=target, prediction=pred); result["predictions_file_sha256"] = sha256_file(save_path)
    return result


def build_identity(repo: Path, anchor_path: Path, anchors: Mapping[str, Any], caches: Mapping[str, Any], seed: int, anchor_sources: Mapping[str, str] | None = None) -> dict[str, Any]:
    files = {"runner": sha256_file(Path(__file__).resolve()), "model": sha256_file(V3_ROOT / "ocr_dota_v3/model.py"), "rank_compatibility": sha256_file(V3_ROOT / "ocr_dota_v3/rank_compatibility.py"), "anchor": sha256_file(anchor_path)}
    identity = {"version": VERSION, "datasets": list(anchors), "global_seed": seed, "precision": "fp32", "anchor_source": str(anchor_path), "anchor_source_kind": "rank_free_p0_jsonl" if anchor_path.suffix.lower() == ".jsonl" else "legacy_best_json", "anchor_source_sha256": files["anchor"], "anchor_sources_by_dataset": dict(anchor_sources or {}), "anchors_sha256": stable_sha(anchors), "caches": caches, "code_sha256": files}
    return {"identity": identity, "identity_sha256": stable_sha(identity)}


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def summarize_posterior(out: Path, rows: list[Mapping[str, Any]], datasets: tuple[str, ...]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("phase") == "posterior" and r.get("status") == "ok"]
    global_rows = []
    for cand in POSTERIOR_IDS:
        rr = [r for r in usable if r["candidate_id"] == cand]
        if len(rr) != len(datasets): continue
        macro = float(np.mean([r["accuracy"] for r in rr])); micro_n = sum(int(r["num_samples"]) for r in rr); micro_c = sum(int(r["correct"]) for r in rr)
        global_rows.append({"candidate": cand, "strength": rr[0]["config"]["rank"]["prediction_strength"], "macro_accuracy": macro, "micro_accuracy": 100*micro_c/micro_n, "positive_datasets": sum(int(r["delta_correct"]) > 0 for r in rr), "worst_delta": min(float(r["delta_pp"]) for r in rr), "correct": micro_c, "num_samples": micro_n})
    best = max(global_rows, key=lambda x: (x["macro_accuracy"], x["positive_datasets"], x["micro_accuracy"], x["worst_delta"])) if global_rows else None
    atomic_json(out / "posterior_best_global.json", {"selection": "macro then positive datasets then micro then worst delta", "best": best, "rows": global_rows})
    oracle_rows = []
    for dataset in datasets:
        rr = [r for r in usable if r["dataset"] == dataset]; oracle = max(rr, key=lambda r: (int(r["correct"]), -float(r["config"]["rank"]["prediction_strength"]))) if rr else None
        if oracle: oracle_rows.append({"dataset": dataset, "display_name": DISPLAY[dataset], "candidate": oracle["candidate_id"], "strength": oracle["config"]["rank"]["prediction_strength"], "delta_correct": oracle["delta_correct"], "delta_pp": oracle["delta_pp"]})
    fields = ["dataset", "display_name", "candidate", "strength", "delta_correct", "delta_pp"]; write_csv(out / "posterior_per_dataset_oracle.csv", oracle_rows, fields)
    loo_rows = []
    for held in datasets:
        train = [r for r in global_rows if True]
        scores = []
        for cand in POSTERIOR_IDS:
            rr = [r for r in usable if r["candidate_id"] == cand and r["dataset"] != held]
            if len(rr) != len(datasets)-1: continue
            scores.append((float(np.mean([r["accuracy"] for r in rr])), sum(int(r["delta_correct"]) > 0 for r in rr), float(np.mean([r["delta_pp"] for r in rr])), cand))
        if scores:
            chosen = max(scores, key=lambda x: (x[0], x[1], x[2]))[3]; row = next(r for r in usable if r["dataset"] == held and r["candidate_id"] == chosen)
            loo_rows.append({"heldout_dataset": held, "selected_candidate_from_other9": chosen, "selected_strength": row["config"]["rank"]["prediction_strength"], "base_acc": row["base_accuracy"], "candidate_acc": row["accuracy"], "delta_pp": row["delta_pp"], "delta_correct": row["delta_correct"], "corrected": row.get("corrected"), "regressed": row.get("regressed")})
    write_csv(out / "posterior_loocv.csv", loo_rows, ["heldout_dataset", "selected_candidate_from_other9", "selected_strength", "base_acc", "candidate_acc", "delta_pp", "delta_correct", "corrected", "regressed"])
    return {"global_rows": global_rows, "best": best, "oracle_rows": oracle_rows, "loo_rows": loo_rows}


def _update_candidate_id(tau: float, power: float, rule: str = "") -> str:
    base = f"t{tau:g}_p{power:g}"
    return f"U2_{base}_{rule}" if rule else f"U1_{base}"


def summarize_update(out: Path, rows: list[Mapping[str, Any]], datasets: tuple[str, ...]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("phase") in {"update_u1", "update_u2"} and r.get("status") == "ok"]
    candidates = sorted({str(r["candidate_id"]) for r in usable})
    global_rows = []
    for cand in candidates:
        rr = [r for r in usable if r["candidate_id"] == cand]
        if len(rr) != len(datasets): continue
        n = sum(int(r["num_samples"]) for r in rr); c = sum(int(r["correct"]) for r in rr)
        global_rows.append({"candidate": cand, "config": rr[0]["config"], "macro_accuracy": float(np.mean([r["accuracy"] for r in rr])), "macro_last50_accuracy": float(np.mean([r["last50_accuracy"] for r in rr])), "micro_accuracy": 100*c/n, "positive_datasets": sum(int(r["delta_correct"]) > 0 for r in rr), "positive_last50_datasets": sum(float(r["delta_last50_pp"]) > 0 for r in rr), "correct": c, "num_samples": n})
    best = max(global_rows, key=lambda r: (r["macro_accuracy"], r["macro_last50_accuracy"], r["positive_datasets"], r["micro_accuracy"])) if global_rows else None
    atomic_json(out / "update_best_global.json", {"selection": "macro then last50 macro then positive datasets then micro", "best": best, "rows": global_rows})
    oracle_rows = []
    for dataset in datasets:
        rr = [r for r in usable if r["dataset"] == dataset]
        if not rr: continue
        r = max(rr, key=lambda x: (int(x["correct"]), float(x["last50_accuracy"]), x["candidate_id"]))
        oracle_rows.append({"dataset": dataset, "display_name": DISPLAY[dataset], "candidate": r["candidate_id"], "delta_correct": r["delta_correct"], "delta_pp": r["delta_pp"], "delta_last50_pp": r["delta_last50_pp"]})
    write_csv(out / "update_per_dataset_oracle.csv", oracle_rows, ["dataset", "display_name", "candidate", "delta_correct", "delta_pp", "delta_last50_pp"])
    loo_rows = []
    for held in datasets:
        scores = []
        for cand in candidates:
            rr = [r for r in usable if r["candidate_id"] == cand and r["dataset"] != held]
            if len(rr) != len(datasets)-1: continue
            scores.append((float(np.mean([r["accuracy"] for r in rr])), float(np.mean([r["last50_accuracy"] for r in rr])), sum(int(r["delta_correct"]) > 0 for r in rr), cand))
        if scores:
            cand = max(scores, key=lambda x: (x[0], x[1], x[2]))[3]
            r = next(x for x in usable if x["dataset"] == held and x["candidate_id"] == cand)
            loo_rows.append({"heldout_dataset": held, "selected_candidate_from_other9": cand, "candidate_acc": r["accuracy"], "base_acc": r["base_accuracy"], "delta_pp": r["delta_pp"], "delta_correct": r["delta_correct"], "candidate_last50_acc": r["last50_accuracy"], "base_last50_acc": r["base_last50_accuracy"], "delta_last50_pp": r["delta_last50_pp"]})
    write_csv(out / "update_loocv.csv", loo_rows, ["heldout_dataset", "selected_candidate_from_other9", "candidate_acc", "base_acc", "delta_pp", "delta_correct", "candidate_last50_acc", "base_last50_acc", "delta_last50_pp"])
    update_rows = []
    for r in usable:
        update_rows.append({k: r.get(k) for k in ("dataset", "candidate_id", "base_accuracy", "accuracy", "delta_pp", "base_last50_accuracy", "last50_accuracy", "delta_last50_pp", "delta_correct", "delta_last50_correct", "mean_gate", "median_gate", "gate_p10", "gate_p25", "gate_p75", "gate_p90", "mean_effective_update_mass", "prediction_sha256", "state_sha256", "health_status")})
    write_csv(out / "UPDATE_RESPONSIBILITY_ANALYSIS.csv", update_rows, list(update_rows[0]) if update_rows else ["dataset", "candidate_id"])
    return {"global_rows": global_rows, "best": best, "oracle_rows": oracle_rows, "loo_rows": loo_rows}


def write_final_ablation(out: Path, datasets: tuple[str, ...], base_rows: list[Mapping[str, Any]], posterior: Mapping[str, Any], update: Mapping[str, Any], full_rows: list[Mapping[str, Any]]) -> None:
    pbest = posterior.get("best") or {}; ubest = update.get("best") or {}
    rows = []
    all_rows = load_jsonl(out / "candidate_results.jsonl")
    for dataset in datasets:
        b = next((r for r in base_rows if r["dataset"] == dataset), None)
        p = next((r for r in all_rows if r.get("phase") == "posterior" and r.get("dataset") == dataset and r.get("candidate_id") == pbest.get("candidate")), None)
        u = next((r for r in all_rows if r.get("phase") in {"update_u1", "update_u2"} and r.get("dataset") == dataset and r.get("candidate_id") == ubest.get("candidate")), None)
        f = next((r for r in full_rows if r.get("dataset") == dataset), None)
        if b:
            rows.append({"dataset": dataset, "display_name": DISPLAY[dataset], "base_correct": b["correct"], "base_total": b["num_samples"], "base_accuracy": b["accuracy"], "posterior_correct": p["correct"] if p else "", "posterior_accuracy": p["accuracy"] if p else "", "delta_p_correct": p["delta_correct"] if p else "", "delta_p_pp": p["delta_pp"] if p else "", "update_correct": u["correct"] if u else "", "update_accuracy": u["accuracy"] if u else "", "delta_u_correct": u["delta_correct"] if u else "", "delta_u_pp": u["delta_pp"] if u else "", "full_correct": f["correct"] if f else "", "full_accuracy": f["accuracy"] if f else "", "delta_full_correct": f["delta_correct"] if f else "", "delta_full_pp": f["delta_pp"] if f else ""})
    fields = ["dataset", "display_name", "base_correct", "base_total", "base_accuracy", "posterior_correct", "posterior_accuracy", "delta_p_correct", "delta_p_pp", "update_correct", "update_accuracy", "delta_u_correct", "delta_u_pp", "full_correct", "full_accuracy", "delta_full_correct", "delta_full_pp"]
    write_csv(out / "FINAL_ABLATION_10.csv", rows, fields)
    lines = ["# FINAL_ABLATION_10", "", "| Dataset | Rank-free Base | + Posterior | ΔP | + Responsibility | ΔU | Full V3 | ΔFull |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def fmt(x): return "" if x == "" else f"{float(x):.6f}%"
        lines.append(f"| {r['display_name']} | {fmt(r['base_accuracy'])} | {fmt(r['posterior_accuracy'])} | {r['delta_p_pp']} | {fmt(r['update_accuracy'])} | {r['delta_u_pp']} | {fmt(r['full_accuracy'])} | {r['delta_full_pp']} |")
    (out / "FINAL_ABLATION_10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(out: Path, identity: Mapping[str, Any], base_rows: list[Mapping[str, Any]], posterior: Mapping[str, Any], update: Mapping[str, Any] | None = None, full_rows: list[Mapping[str, Any]] | None = None) -> None:
    lines = ["# OCR-DOTA V3 Core2 Ablation (10 datasets)", "", "本报告由隔离 runner 自动生成；标签仅用于完整 replay 后的评测与 paired diagnostic。", "", "## 实验身份", "", f"- version: {VERSION}", f"- identity_sha256: {identity['identity_sha256']}", f"- precision: fp32", "- selection: global / per-dataset oracle / LOOCV are reported separately", "", "## Rank-free Base", "", "| Dataset | Correct | N | Accuracy | Last50 accuracy |", "|---|---:|---:|---:|---:|"]
    for r in base_rows: lines.append(f"| {r['display_name']} | {r['correct']} | {r['num_samples']} | {r['accuracy']:.6f}% | {r.get('last50_accuracy', 0):.6f}% |")
    if posterior.get("best"):
        b = posterior["best"]; lines += ["", "## Posterior global best", "", f"- candidate: `{b['candidate']}`", f"- prediction_strength: `{b['strength']}`", f"- macro accuracy: `{b['macro_accuracy']:.6f}%`", f"- micro accuracy: `{b['micro_accuracy']:.6f}%`", f"- positive datasets: `{b['positive_datasets']}/{len(base_rows)}`"]
    if update and update.get("best"):
        b = update["best"]; lines += ["", "## Update responsibility global best", "", f"- candidate: `{b['candidate']}`", f"- macro accuracy: `{b['macro_accuracy']:.6f}%`", f"- last50 macro accuracy: `{b['macro_last50_accuracy']:.6f}%`", f"- micro accuracy: `{b['micro_accuracy']:.6f}%`", f"- positive datasets: `{b['positive_datasets']}/{len(base_rows)}`", f"- positive last50 datasets: `{b['positive_last50_datasets']}/{len(base_rows)}`"]
    if full_rows:
        n = sum(int(r["num_samples"]) for r in full_rows); c = sum(int(r["correct"]) for r in full_rows)
        lines += ["", "## Full V3 confirmation", "", f"- datasets completed: {len(full_rows)}/{len(base_rows)}", f"- micro accuracy: {100*c/max(1,n):.6f}%"]
    lines += ["", "## 说明", "", "Per-dataset oracle 仅用于机制诊断；LOOCV 在选择 held-out 数据集参数时不使用 held-out 结果。Update/Full 阶段由同一 sidecar 的后续 phase 写入相应表。"]
    (out / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve(); out = Path(args.output).resolve(); datasets = parse_datasets(args.datasets); anchor_path = Path(args.anchor_source).resolve()
    fallback_path = repo / "OCR-DOTA-V3/BEST_SINGLE_CONFIG_NONIMAGENET21.json"
    anchors, anchor_expected, anchor_sources = resolve_anchor(anchor_path, datasets, fallback_path)
    caches = {}
    for dataset in datasets:
        path = cache_path(repo, dataset); data_raw = torch.load(path, map_location="cpu", weights_only=False); n = int(data_raw["features"].shape[0]); ids = data_raw.get("sample_ids", torch.arange(n)); ids = ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids); caches[dataset] = {"path": str(path), "sha256": sha256_file(path), "num_samples": n, "order_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
        expected = anchor_expected.get(dataset, {})
        if expected.get("cache_sha256") and str(expected["cache_sha256"]) != caches[dataset]["sha256"]:
            raise RuntimeError(f"anchor/cache SHA mismatch for {dataset}")
        if expected.get("order_sha256") and str(expected["order_sha256"]) != caches[dataset]["order_sha256"]:
            raise RuntimeError(f"anchor/order SHA mismatch for {dataset}")
        if expected.get("num_samples") and int(expected["num_samples"]) != n:
            raise RuntimeError(f"anchor sample count mismatch for {dataset}")
    identity = build_identity(repo, anchor_path, anchors, caches, args.global_seed, anchor_sources)
    if args.dry_run:
        print(json.dumps({"identity_sha256": identity["identity_sha256"], "datasets": list(datasets), "posterior_candidates": list(POSTERIOR_IDS), "update_candidates": len(UPDATE_TAUS)*len(UPDATE_POWERS)}, indent=2)); return 0
    out.mkdir(parents=True, exist_ok=True); manifest = out / "manifest.json"
    if manifest.exists() and not args.resume: raise RuntimeError(f"output exists: {out}; use --resume")
    if manifest.exists():
        old = json.loads(manifest.read_text());
        if old.get("identity_sha256") != identity["identity_sha256"]: raise RuntimeError("resume identity mismatch")
    else:
        atomic_json(manifest, {"status": "running", **identity, "started_at": time.time(), "anchor_source": str(anchor_path), "anchor_source_kind": "rank_free_p0_jsonl" if anchor_path.suffix.lower() == ".jsonl" else "legacy_best_json"})
    results_path = out / "candidate_results.jsonl"; rows = load_jsonl(results_path)
    done = {(r.get("dataset"), r.get("candidate_fingerprint")): r for r in rows if r.get("status") == "ok" and r.get("identity_sha256") == identity["identity_sha256"]}
    stop = Path(args.stop).resolve() if args.stop else out / "STOP"; base_rows = []
    with PidLock(out / "RUNNING.pid"):
        for dataset in datasets:
            check_stop(stop, dataset); setup_seed(args.global_seed); path = Path(caches[dataset]["path"]); data, meta = legacy.load_cache(path, args.device, args.max_samples)
            if args.max_samples is None and meta["order_sha256"] != caches[dataset]["order_sha256"]: raise RuntimeError(f"order mismatch {dataset}")
            anchor = anchors[dataset]; phase_dir = out / dataset; phase_dir.mkdir(parents=True, exist_ok=True)
            base_cfg = make_config(anchor, prediction_strength=0.0, power=0.0, tau=0.15)
            cached_rows = {(r.get("phase"), r.get("candidate_fingerprint")): r for r in rows if r.get("dataset") == dataset and r.get("status") == "ok"}
            required_p = [("base", candidate_fingerprint(dataset, base_cfg, "base"))] + [("posterior", candidate_fingerprint(dataset, make_config(anchor, prediction_strength=s, power=0.0, tau=0.15), "posterior")) for s in POSTERIOR_STRENGTHS]
            if all(key in cached_rows for key in required_p) and (phase_dir / "posterior_paired.npz").exists():
                base_rows.append(cached_rows[("base", required_p[0][1])])
                del data
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
            grid = replay_posterior_grid(data, base_cfg, stop, args.stop_interval, POSTERIOR_STRENGTHS, phase_dir)
            base_pred = grid["predictions"]["P0"]; target = grid["target"]; first = len(target)//2; bc, ba = _accuracy(base_pred, target); blc, bla = _accuracy(base_pred, target, first)
            base_row = {"phase": "base", "status": "ok", "identity_sha256": identity["identity_sha256"], "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": "P0", "candidate_fingerprint": candidate_fingerprint(dataset, base_cfg, "base"), "config": base_cfg, "num_samples": len(target), "correct": bc, "accuracy": ba, "last50_correct": blc, "last50_accuracy": bla, "prediction_sha256": pred_sha(grid["sample_id"], target, base_pred), "state_sha256": grid["state_sha256"], "trajectory_sha256": grid["trajectory_sha256"], "compatibility_sha256": grid["compatibility_sha256"], "health_status": grid["health"]["health_status"]}
            if args.max_samples is None:
                expected_entry = anchor_expected.get(dataset, {})
                expected_correct = expected_entry.get("correct")
                expected_sha = expected_entry.get("prediction_sha256")
                if expected_correct is not None and int(expected_correct) != bc:
                    raise RuntimeError(f"Rank-free Base reproduction mismatch for {dataset}: {bc} != {expected_correct}")
                if expected_sha and str(expected_sha) != base_row["prediction_sha256"]:
                    raise RuntimeError(f"Rank-free Base prediction SHA mismatch for {dataset}")
            append_jsonl(results_path, base_row); base_rows.append(base_row)
            # Keep P0 in the posterior table as the preregistered no-prior
            # control.  It is the same cold-start replay as Base, but a
            # separate phase row is required for global/LOOCV bookkeeping.
            p0_fp = candidate_fingerprint(dataset, base_cfg, "posterior")
            p0_diag = {"corrected": 0, "regressed": 0, "net_correction": 0,
                       "unchanged_correct": bc, "unchanged_wrong": len(target) - bc,
                       "mcnemar_b": 0, "mcnemar_c": 0, "mcnemar_stat": 0.0,
                       "mcnemar_p": 1.0, "compat_mean": float(grid["compat_sample"].mean()),
                       "compat_p10": float(np.percentile(grid["compat_sample"], 10)),
                       "compat_p50": float(np.percentile(grid["compat_sample"], 50)),
                       "compat_p90": float(np.percentile(grid["compat_sample"], 90)),
                       "quartiles": _quartile_rows(base_pred, base_pred, target, grid["compat_sample"])}
            p0_row = {"phase": "posterior", "status": "ok", "identity_sha256": identity["identity_sha256"],
                      "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": "P0",
                      "candidate_fingerprint": p0_fp, "config": base_cfg, "num_samples": len(target),
                      "correct": bc, "accuracy": ba, "base_correct": bc, "base_accuracy": ba,
                      "last50_correct": blc, "last50_accuracy": bla, "delta_correct": 0, "delta_pp": 0.0,
                      "prediction_sha256": base_row["prediction_sha256"], "state_sha256": grid["state_sha256"],
                      "trajectory_sha256": grid["trajectory_sha256"], "compatibility_sha256": grid["compatibility_sha256"],
                      "health_status": grid["health"]["health_status"], **p0_diag}
            append_jsonl(results_path, p0_row); rows.append(p0_row)
            # Independent probes enforce that posterior heads do not alter state.
            probe_shas = [independent_replay_state_sha(data, base_cfg, s, stop, args.stop_interval) for s in (0.0, 0.30, 1.0)]
            if len(set(probe_shas)) != 1 or probe_shas[0] != grid["state_sha256"]: raise RuntimeError(f"posterior shared trajectory contract failed: {dataset}")
            for i, strength in enumerate(POSTERIOR_STRENGTHS):
                cid = f"P{i}"; cfg = make_config(anchor, prediction_strength=strength, power=0.0, tau=0.15); fp = candidate_fingerprint(dataset, cfg, "posterior")
                if cid == "P0": continue
                pred = grid["predictions"][cid]; cc, ca = _accuracy(pred, target); lc, la = _accuracy(pred, target, first); d = _diagnostics(base_pred, pred, target, grid["compat_sample"])
                row = {"phase": "posterior", "status": "ok", "identity_sha256": identity["identity_sha256"], "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": cid, "candidate_fingerprint": fp, "config": cfg, "num_samples": len(target), "correct": cc, "accuracy": ca, "base_correct": bc, "base_accuracy": ba, "last50_correct": lc, "last50_accuracy": la, "delta_correct": cc-bc, "delta_pp": ca-ba, "prediction_sha256": pred_sha(grid["sample_id"], target, pred), "state_sha256": grid["state_sha256"], "trajectory_sha256": grid["trajectory_sha256"], "compatibility_sha256": grid["compatibility_sha256"], "health_status": grid["health"]["health_status"], **d}
                append_jsonl(results_path, row); rows.append(row)
            atomic_npz(phase_dir / "posterior_paired.npz", sample_id=grid["sample_id"], target=target, base_prediction=base_pred, compat_sample=grid["compat_sample"], stable_top1=grid["stable_top1"], dynamic_top1=grid["dynamic_top1"], **{f"prediction_{k}": v for k,v in grid["predictions"].items()})
            del data
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Posterior P1 global selection is performed before U1.  It is only a
        # selection over completed streams, never an online label-dependent
        # choice.
        all_rows = load_jsonl(results_path)
        posterior = summarize_posterior(out, all_rows, datasets)
        pbest = posterior.get("best") or {}

        # U1: 16 independent cold-start replays.  Prediction is explicitly
        # disabled; each tau/power changes the state trajectory.
        for dataset in datasets:
            check_stop(stop, f"U1 {dataset}"); setup_seed(args.global_seed)
            path = Path(caches[dataset]["path"]); data, meta = legacy.load_cache(path, args.device, args.max_samples)
            anchor = anchors[dataset]; b = next(r for r in base_rows if r["dataset"] == dataset)
            for tau in UPDATE_TAUS:
                for power in UPDATE_POWERS:
                    cid = _update_candidate_id(tau, power); cfg = make_config(anchor, prediction_strength=0.0, tau=tau, power=power); fp = candidate_fingerprint(dataset, cfg, "update_u1")
                    if (dataset, fp) in done: continue
                    metric = replay_update(data, cfg, stop, args.stop_interval, out / dataset / f"{cid}.npz")
                    dlast = float(metric["last50_accuracy"] - b["last50_accuracy"])
                    row = {"phase": "update_u1", "status": "ok", "identity_sha256": identity["identity_sha256"], "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": cid, "candidate_fingerprint": fp, "config": cfg, "num_samples": metric["num_samples"], "correct": metric["correct"], "accuracy": metric["accuracy"], "base_correct": b["correct"], "base_accuracy": b["accuracy"], "last50_correct": metric["last50_correct"], "last50_accuracy": metric["last50_accuracy"], "base_last50_correct": b["last50_correct"], "base_last50_accuracy": b["last50_accuracy"], "delta_correct": metric["correct"]-b["correct"], "delta_pp": metric["accuracy"]-b["accuracy"], "delta_last50_correct": metric["last50_correct"]-b["last50_correct"], "delta_last50_pp": dlast, **{k: metric[k] for k in ("prediction_sha256", "trajectory_sha256", "state_sha256", "mean_gate", "median_gate", "gate_p10", "gate_p25", "gate_p75", "gate_p90", "mean_effective_update_mass")}, "health_status": metric["health"]["health_status"]}
                    append_jsonl(results_path, row); rows.append(row); done[(dataset, fp)] = row
            del data
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        all_rows = load_jsonl(results_path)
        u1_rows = [r for r in all_rows if r.get("phase") == "update_u1" and r.get("status") == "ok"]
        u1_global = []
        for cid in sorted({r["candidate_id"] for r in u1_rows}):
            rr = [r for r in u1_rows if r["candidate_id"] == cid]
            if len(rr) == len(datasets):
                u1_global.append((float(np.mean([r["accuracy"] for r in rr])), float(np.mean([r["last50_accuracy"] for r in rr])), cid))
        top_u1 = [x[2] for x in sorted(u1_global, key=lambda x: (x[0], x[1]), reverse=True)[:2]]
        top_configs = {}
        for cid in top_u1:
            rr = next(r for r in u1_rows if r["candidate_id"] == cid); top_configs[cid] = rr["config"]

        # U2: allocation rule search is deliberately restricted to the two
        # U1 gate settings with the best global macro score.
        for dataset in datasets:
            check_stop(stop, f"U2 {dataset}"); setup_seed(args.global_seed)
            path = Path(caches[dataset]["path"]); data, meta = legacy.load_cache(path, args.device, args.max_samples); anchor = anchors[dataset]; b = next(r for r in base_rows if r["dataset"] == dataset)
            for gate_cid in top_u1:
                base_gate = top_configs[gate_cid]; tau = float(base_gate["rank"]["tau_rank"]); power = float(base_gate["rank"]["update_power"])
                for rule in UPDATE_RULES:
                    cid = _update_candidate_id(tau, power, rule); cfg = make_config(anchor, prediction_strength=0.0, tau=tau, power=power, update_rule=rule); fp = candidate_fingerprint(dataset, cfg, "update_u2")
                    if (dataset, fp) in done: continue
                    metric = replay_update(data, cfg, stop, args.stop_interval, out / dataset / f"{cid}.npz")
                    row = {"phase": "update_u2", "status": "ok", "identity_sha256": identity["identity_sha256"], "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": cid, "candidate_fingerprint": fp, "config": cfg, "num_samples": metric["num_samples"], "correct": metric["correct"], "accuracy": metric["accuracy"], "base_correct": b["correct"], "base_accuracy": b["accuracy"], "last50_correct": metric["last50_correct"], "last50_accuracy": metric["last50_accuracy"], "base_last50_correct": b["last50_correct"], "base_last50_accuracy": b["last50_accuracy"], "delta_correct": metric["correct"]-b["correct"], "delta_pp": metric["accuracy"]-b["accuracy"], "delta_last50_correct": metric["last50_correct"]-b["last50_correct"], "delta_last50_pp": metric["last50_accuracy"]-b["last50_accuracy"], **{k: metric[k] for k in ("prediction_sha256", "trajectory_sha256", "state_sha256", "mean_gate", "median_gate", "gate_p10", "gate_p25", "gate_p75", "gate_p90", "mean_effective_update_mass")}, "health_status": metric["health"]["health_status"]}
                    append_jsonl(results_path, row); rows.append(row); done[(dataset, fp)] = row
            del data
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        all_rows = load_jsonl(results_path); update = summarize_update(out, all_rows, datasets)
        ubest = update.get("best") or {}
        full_rows: list[dict[str, Any]] = []
        # Full confirmation is conditional on both independent global modules
        # having positive macro evidence.  It uses exactly one combined config.
        base_macro = float(np.mean([r["accuracy"] for r in base_rows])) if base_rows else 0.0
        ppositive = bool(pbest and float(pbest.get("macro_accuracy", 0.0)) > base_macro)
        upositive = bool(ubest and float(ubest.get("macro_accuracy", 0.0)) > base_macro)
        if ppositive and upositive:
            pstrength = float(pbest["strength"]); ucfg = ubest["config"]; utau = float(ucfg["rank"]["tau_rank"]); upower = float(ucfg["rank"]["update_power"]); full_strength = pstrength / 0.15 * utau
            for dataset in datasets:
                check_stop(stop, f"Full {dataset}"); setup_seed(args.global_seed); path = Path(caches[dataset]["path"]); data, meta = legacy.load_cache(path, args.device, args.max_samples); anchor = anchors[dataset]; b = next(r for r in base_rows if r["dataset"] == dataset); cfg = make_config(anchor, prediction_strength=full_strength, tau=utau, power=upower, update_rule=ucfg["update_rule"]); metric = replay_update(data, cfg, stop, args.stop_interval, out / dataset / "full_v3.npz"); row = {"phase": "full", "status": "ok", "identity_sha256": identity["identity_sha256"], "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": "FULL", "candidate_fingerprint": candidate_fingerprint(dataset, cfg, "full"), "config": cfg, "num_samples": metric["num_samples"], "correct": metric["correct"], "accuracy": metric["accuracy"], "base_correct": b["correct"], "base_accuracy": b["accuracy"], "last50_correct": metric["last50_correct"], "last50_accuracy": metric["last50_accuracy"], "delta_correct": metric["correct"]-b["correct"], "delta_pp": metric["accuracy"]-b["accuracy"], "delta_last50_pp": metric["last50_accuracy"]-b["last50_accuracy"], "health_status": metric["health"]["health"]["health_status"] if "health" in metric and isinstance(metric["health"], dict) and "health" in metric["health"] else metric["health"]["health_status"], **{k: metric[k] for k in ("prediction_sha256", "trajectory_sha256", "state_sha256", "mean_gate", "median_gate", "gate_p10", "gate_p25", "gate_p75", "gate_p90", "mean_effective_update_mass")}}; append_jsonl(results_path, row); full_rows.append(row); del data; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # A second replay is the required cold-start verification.
            for row in full_rows:
                dataset = row["dataset"]; data, _ = legacy.load_cache(Path(caches[dataset]["path"]), args.device, args.max_samples); cfg = row["config"]; verify = replay_update(data, cfg, stop, args.stop_interval); 
                if verify["prediction_sha256"] != row["prediction_sha256"] or verify["state_sha256"] != row["state_sha256"]: raise RuntimeError(f"Full cold replay mismatch: {dataset}")
                del data

        all_rows = load_jsonl(results_path); update = summarize_update(out, all_rows, datasets); write_final_ablation(out, datasets, base_rows, posterior, update, full_rows); write_csv(out / "base_results.csv", base_rows, ["dataset", "display_name", "num_samples", "correct", "accuracy", "last50_correct", "last50_accuracy", "prediction_sha256", "state_sha256"]); write_final_report(out, identity, base_rows, posterior, update, full_rows)
        atomic_json(manifest, {"status": "complete", **identity, "finished_at": time.time(), "results_sha256": sha256_file(results_path), "posterior_best": posterior.get("best"), "update_best": update.get("best"), "full_completed": len(full_rows) == len(datasets)})
    print(json.dumps({"status": "complete", "output": str(out), "identity_sha256": identity["identity_sha256"], "base_datasets": len(base_rows), "posterior_candidates": len(POSTERIOR_STRENGTHS), "update_u1_candidates": 16, "update_u2_gate_count": len(top_u1), "full_completed": len(full_rows) == len(datasets)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--repo-root", default=str(REPO_ROOT)); p.add_argument("--output", default=str(V3_ROOT / "log/core2_ablation10_20260910")); p.add_argument("--anchor-source", default=str(DEFAULT_RANK_FREE_SOURCE)); p.add_argument("--device", default="cuda"); p.add_argument("--datasets"); p.add_argument("--global-seed", type=int, default=1); p.add_argument("--stop-interval", type=int, default=25); p.add_argument("--stop"); p.add_argument("--max-samples", type=int); p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true"); return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
