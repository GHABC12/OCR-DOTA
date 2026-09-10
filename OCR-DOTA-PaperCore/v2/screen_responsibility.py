#!/usr/bin/env python3
"""Quick/full responsibility screening for the fixed V2 protocol.

The wrapper delegates the sequential FP32 state update to the already
validated PaperCore replay, while constraining its configuration to V2's R0,
R1 and R2 formulas. It never passes targets into the replay path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
REPO = V2_ROOT.parent
sys.path.insert(0, str(REPO / "OCR-DOTA-V3"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V2_ROOT))

from run_paper_ablation import replay_paper  # noqa: E402
import tune_nonimagenet21 as core  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402
from v2.protocol_v2 import DEV_DATASETS, PROTOCOL_VERSION, RESPONSIBILITY_CANDIDATES, TEN_DATASETS  # noqa: E402


def config_for(candidate: dict[str, Any]) -> dict[str, Any]:
    """Map a V2 responsibility candidate to the validated replay API."""
    return {
        "tau_rank": 0.15,
        "alpha": 0.0,
        "lambda_max": 0.0,
        "prediction_power": 1.0,
        "posterior_mode": "gated_log_prior",
        "delta": float(candidate.get("delta", 0.0)),
        "beta_resp": float(candidate.get("beta", 0.0)),
        "responsibility_mode": candidate["mode"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"correct": 0, "num_samples": 0, "micro_accuracy": None, "macro_accuracy": None, "wrm": None, "late50_macro": None}
    return {
        "correct": int(sum(r["correct"] for r in rows)),
        "num_samples": int(sum(r["num_samples"] for r in rows)),
        "micro_accuracy": 100.0 * sum(r["correct"] for r in rows) / sum(r["num_samples"] for r in rows),
        "macro_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "wrm": float(np.mean([r["wrm"] for r in rows])),
        "late50_macro": float(np.mean([r["late50_accuracy"] for r in rows])),
        "total_update_mass": float(sum(r["total_update_mass"] for r in rows)),
    }


def screen(
    datasets: tuple[str, ...], output: Path, max_samples: int | None = 1000,
    device: str = "cuda", candidates: tuple[dict[str, Any], ...] = RESPONSIBILITY_CANDIDATES,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    stop = output / "STOP"
    if stop.exists():
        raise RuntimeError(f"STOP marker exists: {stop}")
    all_rows: list[dict[str, Any]] = []
    base_rows: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        cache = core.cache_path(dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT)
        data, meta = legacy.load_cache(cache, device, max_samples)
        base = core.load_original_base(REPO, dataset)
        core.setup_seed(1)
        base_metrics = replay_paper(data, base, config_for({"mode": "dota", "delta": 0, "beta": 0}), "base", stop, save_path=output / "runs" / dataset / "base.npz")
        base_rows[dataset] = base_metrics
        for index, candidate in enumerate(candidates):
            if stop.exists():
                raise RuntimeError(f"STOP at dataset {dataset}, candidate {index}")
            core.setup_seed(1)
            metrics = replay_paper(data, base, config_for(candidate), "responsibility", stop, save_path=output / "runs" / dataset / f"candidate_{index:02d}.npz")
            row = {"dataset": dataset, "candidate_id": index, **candidate, **metrics,
                   "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"],
                   "screen_prefix": int(metrics["num_samples"])}
            all_rows.append(row)
    aggregates = []
    for index, candidate in enumerate(candidates):
        rows = [r for r in all_rows if r["candidate_id"] == index and r["dataset"] in DEV_DATASETS]
        base = [base_rows[d] for d in datasets if d in DEV_DATASETS and d in base_rows]
        a = _aggregate(rows); b = _aggregate(base)
        aggregates.append({"candidate_id": index, **candidate, "aggregate": a,
                           "base_aggregate": b,
                           "not_obviously_failed": bool(a["correct"] >= b["correct"] and (a["wrm"] is None or b["wrm"] is None or a["wrm"] < b["wrm"]))})
    result = {
        "status": "complete", "protocol_version": PROTOCOL_VERSION,
        "stage": "quick_screen" if max_samples is not None and max_samples <= 1000 else "full_dev",
        "datasets": list(datasets), "prefix": max_samples,
        "base": {d: base_rows[d] for d in base_rows},
        "candidates": aggregates,
        "rows": all_rows,
        "label_policy": "targets are used only by replay metrics after update; no target is passed to model",
    }
    filename = "responsibility_quick_screen.json" if max_samples is not None and max_samples <= 1000 else "responsibility_full_dev.json"
    (output / filename).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DEV_DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    unknown = set(datasets) - set(TEN_DATASETS)
    if unknown:
        raise ValueError(f"unknown V2 dataset(s): {sorted(unknown)}")
    print(json.dumps(screen(datasets, args.output.resolve(), args.max_samples, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
