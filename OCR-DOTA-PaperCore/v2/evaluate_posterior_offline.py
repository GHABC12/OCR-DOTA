#!/usr/bin/env python3
"""Evaluate all preregistered V2 posterior candidates from evidence caches.

No model/state replay occurs in this script. Labels are used only for the
offline evaluation metrics and never enter an adaptation computation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from protocol_v2 import (
    DEV_DATASETS,
    DIRECT_GAMMAS,
    ENTROPY_GAMMAS,
    EVAL_DATASETS,
    PROTOCOL_VERSION,
    canonical,
    digest,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=-1, keepdims=True)


def _prediction_sha(sample_id: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> str:
    """Match tune_nonimagenet21.prediction_sha byte-for-byte."""
    import hashlib
    digest = hashlib.sha256()
    for name, value in (("sample_id", sample_id), ("target", target), ("prediction", prediction)):
        value = np.ascontiguousarray(value)
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(canonical(list(value.shape)).encode()); digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    value = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (confidence >= lo) & ((confidence <= hi) if i == bins - 1 else (confidence < hi))
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _macro_accuracy(target: np.ndarray, pred: np.ndarray) -> float:
    labels = np.unique(target)
    if len(labels) == 0:
        return 0.0
    return 100.0 * float(np.mean([np.mean(pred[target == label] == label) for label in labels]))


def _auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    # Average ranks handle tied clean masses without depending on a sorting
    # implementation's tie order.
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    u = ranked[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def candidate_configs() -> list[dict[str, Any]]:
    result = []
    for gamma in DIRECT_GAMMAS:
        result.append({"mode": "direct_prior", "gamma": float(gamma), "entropy_power": 1.0})
    for gamma in ENTROPY_GAMMAS:
        result.append({"mode": "entropy_gated_prior", "gamma": float(gamma), "entropy_power": 1.0})
    return result


def load_evidence(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "sample_id", "target", "clip_logits", "gaussian_logits", "fusion_weight",
            "compatibility", "clean_mass", "base_prediction", "base_margin", "rank_displacement",
        }
        missing = required - set(loaded.files)
        if missing:
            raise ValueError(f"evidence {path} missing {sorted(missing)}")
        return {key: loaded[key].copy() for key in required}


def evaluate_candidate(evidence: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clip = evidence["clip_logits"].astype(np.float64, copy=False)
    gaussian = evidence["gaussian_logits"].astype(np.float64, copy=False)
    compatibility = evidence["compatibility"].astype(np.float64, copy=False)
    fusion = evidence["fusion_weight"].reshape(-1, 1).astype(np.float64, copy=False)
    target = evidence["target"].reshape(-1).astype(np.int64, copy=False)
    ids = evidence["sample_id"].reshape(-1).astype(np.int64, copy=False)
    base_prediction = evidence["base_prediction"].reshape(-1).astype(np.int64, copy=False)
    base_logits = clip + fusion * gaussian
    p_dota = _softmax(gaussian)
    k = gaussian.shape[-1]
    entropy = -(p_dota * np.log(np.clip(p_dota, np.finfo(np.float64).tiny, None))).sum(-1, keepdims=True)
    entropy_norm = entropy / math.log(max(k, 2))
    prior = float(config["gamma"]) * np.log(np.clip(compatibility, np.finfo(np.float64).tiny, None))
    if config["gamma"] == 0 or np.all(compatibility == 1):
        gate = np.ones((len(target), 1), dtype=np.float64)
        correction = np.zeros_like(gaussian)
        ocr_logits = gaussian
    elif config["mode"] == "direct_prior":
        gate = np.ones((len(target), 1), dtype=np.float64)
        correction = prior
        ocr_logits = gaussian + correction
    elif config["mode"] == "entropy_gated_prior":
        gate = entropy_norm ** float(config.get("entropy_power", 1.0))
        correction = gate * prior
        ocr_logits = gaussian + correction
    else:
        raise ValueError(config["mode"])
    final_logits = clip + fusion * ocr_logits
    p_final = _softmax(final_logits)
    pred = final_logits.argmax(axis=-1).astype(np.int64)
    correct = pred == target
    base_correct = base_prediction == target
    wc = int((~base_correct & correct).sum())
    cw = int((base_correct & ~correct).sum())
    confidence = p_final.max(axis=-1)
    nll = -np.log(np.clip(p_final[np.arange(len(target)), target], np.finfo(np.float64).tiny, None))
    onehot_brier = (p_final * p_final).sum(axis=-1) - 2.0 * p_final[np.arange(len(target)), target] + 1.0
    metric = {
        "num_samples": int(len(target)),
        "correct": int(correct.sum()),
        "accuracy": 100.0 * float(correct.mean()) if len(correct) else 0.0,
        "macro_accuracy": _macro_accuracy(target, pred),
        "nll": float(nll.mean()) if len(nll) else 0.0,
        "brier": float(onehot_brier.mean()) if len(onehot_brier) else 0.0,
        "ece": _ece(confidence, correct),
        "wrong_to_correct": wc,
        "correct_to_wrong": cw,
        "net_correction": wc - cw,
        "changed_prediction_rate": 100.0 * float(np.mean(pred != base_prediction)) if len(pred) else 0.0,
        # Replay uses compact class-label dtypes (uint8/uint16/uint32) for
        # prediction SHA. Preserve that dtype even though metrics use int64.
        "prediction_sha256": _prediction_sha(ids, evidence["target"].reshape(-1), pred.astype(evidence["target"].dtype, copy=False)),
        "mean_effective_correction_abs": float(np.abs(fusion * correction).mean()),
        "max_effective_correction_abs": float(np.abs(fusion * correction).max()) if correction.size else 0.0,
        "prediction_gate_mean": float(gate.mean()),
        "mean_clean_mass": float(evidence["clean_mass"].mean()),
    }
    base_top2 = np.argpartition(base_logits, -2, axis=-1)[:, -2:]
    base_top2_scores = np.take_along_axis(base_logits, base_top2, axis=-1)
    base_top2_order = np.argsort(base_top2_scores, axis=-1)[:, ::-1]
    base_top2 = np.take_along_axis(base_top2, base_top2_order, axis=-1)
    effective = fusion * correction
    top1 = base_top2[:, 0]
    top2 = base_top2[:, 1]
    ocr_shift = effective[np.arange(len(target)), top2] - effective[np.arange(len(target)), top1]
    needed_shift = evidence["base_margin"].reshape(-1).astype(np.float64)
    diagnostics = {
        "needed_shift_mean": float(needed_shift.mean()) if len(needed_shift) else 0.0,
        "ocr_shift_mean": float(ocr_shift.mean()) if len(ocr_shift) else 0.0,
        "ocr_shift_exceeds_needed_rate": 100.0 * float(np.mean(ocr_shift > needed_shift)) if len(ocr_shift) else 0.0,
        "effective_correction_mean_abs": float(np.abs(effective).mean()),
        "effective_correction_max_abs": float(np.abs(effective).max()) if effective.size else 0.0,
    }
    margin_rows = []
    bins = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, float("inf")))
    for lo, hi in bins:
        mask = (needed_shift >= lo) & (needed_shift < hi)
        margin_rows.append({
            "margin_lo": lo, "margin_hi": None if math.isinf(hi) else hi,
            "num_samples": int(mask.sum()),
            "changed_predictions": int((mask & (pred != base_prediction)).sum()),
            "wrong_to_correct": int((mask & ~base_correct & correct).sum()),
            "correct_to_wrong": int((mask & base_correct & ~correct).sum()),
            "net": int((mask & ~base_correct & correct).sum() - (mask & base_correct & ~correct).sum()),
        })
    comp_top1 = compatibility[np.arange(len(target)), base_prediction]
    second = base_top2[:, 1]
    comp_diag = {
        "base_correct": int(base_correct.sum()),
        "base_wrong": int((~base_correct).sum()),
        "mean_clean_mass_correct": float(evidence["clean_mass"].reshape(-1)[base_correct].mean()) if base_correct.any() else None,
        "mean_clean_mass_wrong": float(evidence["clean_mass"].reshape(-1)[~base_correct].mean()) if (~base_correct).any() else None,
        "median_clean_mass_correct": float(np.median(evidence["clean_mass"].reshape(-1)[base_correct])) if base_correct.any() else None,
        "median_clean_mass_wrong": float(np.median(evidence["clean_mass"].reshape(-1)[~base_correct])) if (~base_correct).any() else None,
        "mean_c_base_top1": float(comp_top1.mean()) if len(comp_top1) else 0.0,
        "mean_c_true_class": float(compatibility[np.arange(len(target)), target].mean()) if len(target) else 0.0,
        "mean_c_second_class": float(compatibility[np.arange(len(target)), second].mean()) if len(target) else 0.0,
        "clean_mass_auroc_base_correct": _auroc(evidence["clean_mass"].reshape(-1), base_correct),
    }
    # IDs are returned to make it straightforward for a caller to persist an
    # optional prediction trace without rerunning the model.
    diagnostics["sample_id_sha256"] = digest(ids.tolist())
    return metric, diagnostics, {"margin_rows": margin_rows, "compatibility": comp_diag}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def evaluate_directory(evidence_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows, margin_rows, comp_rows = [], [], []
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in (DEV_DATASETS + EVAL_DATASETS):
        path = evidence_root / f"{dataset}.npz"
        if not path.exists():
            continue
        evidence = load_evidence(path)
        base_metric, _, _ = evaluate_candidate(evidence, {"mode": "direct_prior", "gamma": 0.0, "entropy_power": 1.0})
        candidates = []
        for index, config in enumerate(candidate_configs()):
            metric, diag, extras = evaluate_candidate(evidence, config)
            row = {"dataset": dataset, "candidate_id": index, **config, **metric}
            rows.append(row); candidates.append(row)
            for margin in extras["margin_rows"]:
                margin_rows.append({"dataset": dataset, "candidate_id": index, **config, **margin})
            comp_rows.append({"dataset": dataset, "candidate_id": index, **config, **diag, **extras["compatibility"]})
        by_dataset[dataset] = {"base": base_metric, "candidates": candidates}
    # Aggregate only over the fixed Dev-5 for selection; Eval-5 rows are still
    # exported but cannot affect the frozen choice.
    base_dev = [by_dataset[d]["base"] for d in DEV_DATASETS if d in by_dataset]
    base_agg = {
        "correct": int(sum(r["correct"] for r in base_dev)),
        "num_samples": int(sum(r["num_samples"] for r in base_dev)),
        "macro_accuracy": float(np.mean([r["macro_accuracy"] for r in base_dev])) if base_dev else None,
    }
    agg_candidates = []
    for config in candidate_configs():
        crows = [r for r in rows if r["mode"] == config["mode"] and r["gamma"] == config["gamma"] and r["dataset"] in DEV_DATASETS]
        if not crows:
            continue
        aggregate = {
            "correct": int(sum(r["correct"] for r in crows)),
            "num_samples": int(sum(r["num_samples"] for r in crows)),
            "micro_accuracy": 100.0 * sum(r["correct"] for r in crows) / max(1, sum(r["num_samples"] for r in crows)),
            "macro_accuracy": float(np.mean([r["macro_accuracy"] for r in crows])),
            "net_correction": int(sum(r["net_correction"] for r in crows)),
        }
        agg_candidates.append({"config": config, "aggregate": aggregate,
                              "acceptable": bool(aggregate["correct"] > base_agg["correct"] and aggregate["macro_accuracy"] >= base_agg["macro_accuracy"] and aggregate["net_correction"] > 0)})
    acceptable = [r for r in agg_candidates if r["acceptable"]]
    acceptable.sort(key=lambda r: (-r["aggregate"]["macro_accuracy"], -r["aggregate"]["correct"], r["config"]["gamma"], 0 if r["config"]["mode"] == "direct_prior" else 1))
    selected = acceptable[0] if acceptable else None
    result = {"protocol_version": PROTOCOL_VERSION, "base_dev": base_agg,
              "candidates": agg_candidates, "selected": selected,
              "status": "selected" if selected else "research_negative",
              "datasets": sorted(by_dataset)}
    (output / "posterior_search.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output / "posterior_search.csv", rows)
    _write_csv(output / "posterior_margin_analysis.csv", margin_rows)
    (output / "compatibility_analysis.json").write_text(json.dumps(comp_rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_directory(args.evidence_root.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
