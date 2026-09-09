#!/usr/bin/env python3
"""Real online 2^3 ablation of the three historical risks on OCR-DOTA-V3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V3 = REPO / "OCR-DOTA-V3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V3))
sys.path.insert(0, str(HERE))

import tune_nonimagenet21 as core  # noqa: E402
from risk_model import OCRDOTAV3RiskAblation, RiskStrengths, RiskSwitches  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402


VERSION = "ocr-dota-v3-old-three-risk-factorial-online-v1"
DEFAULT_DATASETS = core.TEN_DATASETS
DEFAULT_WINNERS = V3 / "BEST_SINGLE_CONFIG_NONIMAGENET21_AFTER_CONTINUATION.json"
DEFAULT_OUTPUT = HERE / "results" / "classic10_full_factorial_20260901"
MASKS = tuple(itertools.product((False, True), repeat=3))
NAMES = ("residual", "semantic", "order")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    core.atomic_json(path, value)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    core.append_jsonl(path, value)


def parse_datasets(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_DATASETS
    values = tuple(core.normalize_dataset(item) for item in raw.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("datasets must be a non-empty unique list")
    return values


def mask_record(mask: tuple[bool, bool, bool]) -> dict[str, Any]:
    switches = RiskSwitches(*mask)
    return {
        "arm": switches.key,
        "effective_risks": dict(zip(NAMES, mask)),
        "mask": [int(value) for value in mask],
    }


def prediction_sha(sample_ids: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> str:
    return core.prediction_sha(sample_ids, targets, predictions)


def replay(
    data: Mapping[str, Any],
    config: Mapping[str, Any],
    switches: RiskSwitches,
    strengths: RiskStrengths,
    stop: Path,
    stop_interval: int,
) -> dict[str, Any]:
    core.validate_config(config)
    device = str(data["features"].device)
    dim, classes = map(int, data["clip_shape"])
    model = OCRDOTAV3RiskAblation(
        config["base"],
        {"rank": config["rank"], "update": config["update"]},
        dim,
        classes,
        data["text_prototypes"],
        device=device,
        switches=switches,
        strengths=strengths,
    )
    model.eval()
    count = int(data["features"].shape[0])
    dtype = legacy.compact_dtype(classes)
    targets = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    sample_ids = np.asarray(data["sample_ids"], dtype=np.int64)
    predictions = np.empty(count, dtype=dtype)
    trajectory = hashlib.sha256(canonical({"config": config, "arm": switches.key, "strengths": vars(strengths)}).encode())
    compatibility_trajectory = hashlib.sha256()
    risk_sums = {name: 0.0 for name in NAMES}
    update_mass = gate_sum = compatibility_sum = 0.0
    started = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for index in range(count):
            if index % max(1, stop_interval) == 0 and stop.exists():
                raise core.StopRequested(f"STOP at {index}/{count}")
            views = data["features"][index].to(device=device, dtype=torch.float32)
            clip_logits = data["clip_logits"][index:index + 1].to(device=device, dtype=torch.float32)
            prob_map = data["prob_maps"][index].to(device=device, dtype=torch.float32)
            z = views.mean(0, keepdim=True)
            parts = model.posterior_and_responsibility(z, return_parts=True)
            pre_cap = float(config["base"]["rho"]) * model.C.mean() / views.size(0)
            weight = torch.clamp(pre_cap, max=float(config["base"]["eta"]))
            final = clip_logits + weight * parts["rank_logits"]
            if not bool(torch.isfinite(final).all().item()):
                raise FloatingPointError(f"non-finite prediction at sample {index}")
            predictions[index] = int(final.argmax(-1).item())

            features = core.update_features(views, z, config["update_views"])
            geometry_update = model.risk_adjusted_geometry(features)
            p_geometry = F.softmax(geometry_update, dim=-1)
            allocation = core.allocation_by_rule(config["update_rule"], prob_map, p_geometry)
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
            compatibility_trajectory.update(compatibility.detach().contiguous().cpu().numpy().tobytes(order="C"))
            compatibility_sum += float(compatibility.mean().cpu())
            gate_sum += float(gate.mean().cpu())
            update_mass += float(weights.sum(-1).mean().cpu())
            for name in NAMES:
                risk_sums[name] += float(parts[{"residual": "residual_magnitude", "semantic": "semantic_leakage", "order": "order_violation"}[name]].mean().cpu())

    state_health = core.health(model)
    correct = int((predictions == targets).sum())
    return {
        "correct": correct,
        "num_samples": count,
        "accuracy": 100.0 * correct / count,
        "prediction_sha256": prediction_sha(sample_ids, targets, predictions),
        "trajectory_sha256": trajectory.hexdigest(),
        "state_sha256": core.state_sha(model),
        "compatibility_sha256": compatibility_trajectory.hexdigest(),
        "mean_rank_compatibility": compatibility_sum / count,
        "mean_update_gate": gate_sum / count,
        "mean_update_mass": update_mass / count,
        "mean_risks": {name: value / count for name, value in risk_sums.items()},
        "health_status": state_health["health_status"],
        "state_health": state_health,
        "elapsed_sec": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0,
    }


def shapley(values: Mapping[tuple[int, int, int], float]) -> dict[str, float]:
    result: dict[str, float] = {}
    players = range(3)
    factorial = math.factorial
    for player in players:
        others = [item for item in players if item != player]
        total = 0.0
        for size in range(3):
            for subset in itertools.combinations(others, size):
                off = [0, 0, 0]
                for item in subset:
                    off[item] = 1
                on = list(off)
                on[player] = 1
                weight = factorial(size) * factorial(2 - size) / factorial(3)
                total += weight * (values[tuple(on)] - values[tuple(off)])
        result[NAMES[player]] = total
    return result


def shares(phi: Mapping[str, float]) -> dict[str, dict[str, float | None]]:
    signed_total = sum(phi.values())
    magnitude_total = sum(abs(value) for value in phi.values())
    positive_total = sum(max(0.0, value) for value in phi.values())
    return {
        name: {
            "signed_percent": None if abs(signed_total) < 1e-12 else 100.0 * value / signed_total,
            "magnitude_percent": None if magnitude_total == 0 else 100.0 * abs(value) / magnitude_total,
            "positive_percent": None if positive_total == 0 else 100.0 * max(0.0, value) / positive_total,
        }
        for name, value in phi.items()
    }


def build_summary(datasets: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [dict(row) for row in rows if row.get("status") == "ok"]
    table = {(row["dataset"], tuple(row["mask"])): row for row in ok}
    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        if any((dataset, tuple(int(x) for x in mask)) not in table for mask in MASKS):
            continue
        count_values = {tuple(int(x) for x in mask): float(table[(dataset, tuple(int(x) for x in mask))]["correct"]) for mask in MASKS}
        pp_values = {mask: 100.0 * value / table[(dataset, mask)]["num_samples"] for mask, value in count_values.items()}
        phi_count = shapley(count_values)
        phi_pp = shapley(pp_values)
        per_dataset[dataset] = {
            "num_samples": table[(dataset, (0, 0, 0))]["num_samples"],
            "arms": {table[(dataset, tuple(int(x) for x in mask))]["arm"]: {
                "correct": table[(dataset, tuple(int(x) for x in mask))]["correct"],
                "accuracy": table[(dataset, tuple(int(x) for x in mask))]["accuracy"],
                "prediction_sha256": table[(dataset, tuple(int(x) for x in mask))]["prediction_sha256"],
            } for mask in MASKS},
            "delta_all_vs_none_correct": count_values[(1, 1, 1)] - count_values[(0, 0, 0)],
            "shapley_correct": phi_count,
            "shapley_pp": phi_pp,
            "shares": shares(phi_count),
        }
    aggregate: dict[str, Any] | None = None
    if len(per_dataset) == len(tuple(datasets)):
        micro_values = {
            tuple(int(x) for x in mask): sum(float(table[(dataset, tuple(int(x) for x in mask))]["correct"]) for dataset in datasets)
            for mask in MASKS
        }
        macro_values = {
            mask: sum(100.0 * table[(dataset, mask)]["correct"] / table[(dataset, mask)]["num_samples"] for dataset in datasets) / len(tuple(datasets))
            for mask in micro_values
        }
        micro_phi = shapley(micro_values)
        macro_phi = shapley(macro_values)
        aggregate = {
            "num_samples": sum(per_dataset[dataset]["num_samples"] for dataset in datasets),
            "micro_arm_correct": {RiskSwitches(*map(bool, mask)).key: value for mask, value in micro_values.items()},
            "macro_arm_accuracy": {RiskSwitches(*map(bool, mask)).key: value for mask, value in macro_values.items()},
            "micro_shapley_correct": micro_phi,
            "micro_shares": shares(micro_phi),
            "macro_shapley_pp": macro_phi,
            "macro_shares": shares(macro_phi),
            "delta_all_vs_none_correct": micro_values[(1, 1, 1)] - micro_values[(0, 0, 0)],
            "delta_all_vs_none_macro_pp": macro_values[(1, 1, 1)] - macro_values[(0, 0, 0)],
        }
    return {
        "status": "complete" if aggregate is not None else "running",
        "protocol": "full-stream true-online 2^3 factorial diagnostic; no offline trajectory scoring",
        "per_dataset": per_dataset,
        "aggregate": aggregate,
    }


def write_reports(out: Path, summary: Mapping[str, Any]) -> None:
    atomic_json(out / "summary.json", summary)
    rows = summary["per_dataset"]
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "N", "none", "residual_shapley", "semantic_shapley", "order_shapley", "all", "delta_all_none"])
        for dataset, row in rows.items():
            writer.writerow([dataset, row["num_samples"], row["arms"]["D0E0O0"]["correct"], row["shapley_correct"]["residual"], row["shapley_correct"]["semantic"], row["shapley_correct"]["order"], row["arms"]["D1E1O1"]["correct"], row["delta_all_vs_none_correct"]])
    lines = ["OCR-DOTA-V3 三风险真实在线全因子消融", "口径：全量流诊断；不是独立测试泛化结果", ""]
    if summary.get("aggregate"):
        agg = summary["aggregate"]
        lines += [f"总样本数: {agg['num_samples']}", f"全开相对全关: {agg['delta_all_vs_none_correct']:+.0f} 个正确样本", "", "Micro Shapley贡献："]
        for name in NAMES:
            share = agg["micro_shares"][name]
            magnitude = "N/A" if share["magnitude_percent"] is None else f"{share['magnitude_percent']:.3f}%"
            positive = "N/A" if share["positive_percent"] is None else f"{share['positive_percent']:.3f}%"
            lines.append(f"- {name}: {agg['micro_shapley_correct'][name]:+.3f} samples; magnitude={magnitude}; positive={positive}")
    lines += ["", "逐数据集："]
    for dataset, row in rows.items():
        lines.append(f"- {dataset}: none={row['arms']['D0E0O0']['correct']}, all={row['arms']['D1E1O1']['correct']}, Δ={row['delta_all_vs_none_correct']:+.0f}; phi(D/E/O)={row['shapley_correct']['residual']:+.3f}/{row['shapley_correct']['semantic']:+.3f}/{row['shapley_correct']['order']:+.3f}")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--datasets")
    value.add_argument("--device", default="cuda")
    value.add_argument("--global-seed", type=int, default=1)
    value.add_argument("--stop-check-interval", type=int, default=25)
    value.add_argument("--max-samples", type=int, help="smoke only")
    value.add_argument("--residual-strength", type=float, default=0.25)
    value.add_argument("--semantic-strength", type=float, default=0.05)
    value.add_argument("--order-strength", type=float, default=0.05)
    value.add_argument("--winner-source", default=str(DEFAULT_WINNERS))
    value.add_argument("--resume", action="store_true")
    value.add_argument("--skip-verification", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    datasets = parse_datasets(args.datasets)
    out = Path(args.output).resolve()
    stop = out / "STOP"
    strengths = RiskStrengths(args.residual_strength, args.semantic_strength, args.order_strength)
    strengths.validate()
    source = Path(args.winner_source).resolve()
    source_rows = json.loads(source.read_text(encoding="utf-8"))["datasets"]
    missing = set(datasets) - set(source_rows)
    if missing:
        raise RuntimeError(f"winner source misses {sorted(missing)}")
    cache_paths = {dataset: core.cache_path(dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT) for dataset in datasets}
    cache_manifest = json.loads(core.EXPECTED_CACHE_MANIFEST.read_text(encoding="utf-8"))
    identity = {
        "version": VERSION,
        "datasets": datasets,
        "global_seed": args.global_seed,
        "max_samples": args.max_samples,
        "precision": "fp32",
        "risk_strengths": vars(strengths),
        "winner_source": str(source),
        "winner_source_sha256": core.sha256_file(source),
        "code_sha256": {
            "runner": core.sha256_file(Path(__file__).resolve()),
            "risk_model": core.sha256_file(HERE / "risk_model.py"),
            "v3_model": core.sha256_file(V3 / "ocr_dota_v3/model.py"),
            "rank_compatibility": core.sha256_file(V3 / "ocr_dota_v3/rank_compatibility.py"),
            "v3_tuner": core.sha256_file(V3 / "tune_nonimagenet21.py"),
            "legacy_runner": core.sha256_file(REPO / "scripts/run_cross_benchmark_ocr_ablation.py"),
        },
        "caches": {dataset: {
            "path": str(cache_paths[dataset]),
            "sha256": core.sha256_file(cache_paths[dataset]),
            "num_samples": int(cache_manifest[dataset]["num_samples"]),
            "order_sha256": cache_manifest[dataset]["order_sha256"],
        } for dataset in datasets},
        "arms": [mask_record(mask) for mask in MASKS],
    }
    identity_sha = stable_sha(identity)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or old.get("identity_sha256") != identity_sha:
            raise RuntimeError("resume identity mismatch or --resume missing")
    else:
        atomic_json(manifest_path, {"status": "running", "identity": identity, "identity_sha256": identity_sha, "started_at": time.time()})
    results_path = out / "results.jsonl"
    verification_path = out / "verification.jsonl"
    results = core.load_jsonl(results_path)
    done = {(row.get("dataset"), row.get("arm")): row for row in results if row.get("status") == "ok" and row.get("identity_sha256") == identity_sha}
    verified_rows = core.load_jsonl(verification_path)
    verified = {(row.get("dataset"), row.get("arm")): row for row in verified_rows if row.get("status") == "reproduced" and row.get("identity_sha256") == identity_sha}

    with core.PidLock(out / "RUNNING.pid"):
        try:
            for dataset_index, dataset in enumerate(datasets):
                if stop.exists():
                    raise core.StopRequested("STOP before dataset")
                core.setup_seed(args.global_seed)
                data, meta = legacy.load_cache(cache_paths[dataset], args.device, args.max_samples)
                expected = identity["caches"][dataset]
                if meta["sha256"] != expected["sha256"]:
                    raise RuntimeError(f"cache SHA mismatch: {dataset}")
                if args.max_samples is None and (meta["num_samples"] != expected["num_samples"] or meta["order_sha256"] != expected["order_sha256"]):
                    raise RuntimeError(f"cache order/size mismatch: {dataset}")
                config = source_rows[dataset]["config"]
                for arm_index, mask in enumerate(MASKS):
                    arm_info = mask_record(mask)
                    key = (dataset, arm_info["arm"])
                    if key not in done:
                        atomic_json(out / "state.json", {"status": "running", "dataset": dataset, "dataset_index": dataset_index, "arm": arm_info["arm"], "arm_index": arm_index, "updated_at": time.time()})
                        core.setup_seed(args.global_seed)
                        started = time.time()
                        try:
                            metrics = replay(data, config, RiskSwitches(*mask), strengths, stop, args.stop_check_interval)
                            status, error = "ok", None
                        except core.StopRequested:
                            raise
                        except Exception as exc:
                            status, error = "error", f"{type(exc).__name__}: {exc}"
                            metrics = {"elapsed_sec": time.time() - started, "health_status": "unhealthy"}
                        row = {"status": status, "error": error, "identity_sha256": identity_sha, "dataset": dataset, "dataset_index": dataset_index, **arm_info, "strengths": vars(strengths), "config": config, "config_sha256": stable_sha(config), "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"], **metrics}
                        append_jsonl(results_path, row)
                        results.append(row)
                        if status != "ok":
                            raise RuntimeError(f"failed arm {dataset}/{arm_info['arm']}: {error}")
                        done[key] = row
                    if not args.skip_verification and key not in verified:
                        core.setup_seed(args.global_seed)
                        replayed = replay(data, config, RiskSwitches(*mask), strengths, stop, args.stop_check_interval)
                        reference = done[key]
                        keys = ("correct", "num_samples", "prediction_sha256", "trajectory_sha256", "state_sha256", "compatibility_sha256")
                        mismatches = {name: [reference[name], replayed[name]] for name in keys if reference[name] != replayed[name]}
                        record = {"status": "reproduced" if not mismatches else "mismatch", "identity_sha256": identity_sha, "dataset": dataset, **arm_info, "mismatches": mismatches, "reference": {name: reference[name] for name in keys}, "replay": {name: replayed[name] for name in keys}, "elapsed_sec": replayed["elapsed_sec"]}
                        append_jsonl(verification_path, record)
                        if mismatches:
                            raise RuntimeError(f"verification mismatch {dataset}/{arm_info['arm']}: {mismatches}")
                        verified[key] = record
                    write_reports(out, build_summary(datasets, results))
                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            summary = build_summary(datasets, results)
            write_reports(out, summary)
            final_status = "complete_smoke" if args.max_samples is not None else "complete"
            atomic_json(out / "state.json", {"status": final_status, "datasets": len(datasets), "arms": len(MASKS), "verified": len(verified), "finished_at": time.time()})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"status": final_status, "finished_at": time.time(), "summary_sha256": core.sha256_file(out / "summary.json")})
            atomic_json(manifest_path, manifest)
        except core.StopRequested as exc:
            atomic_json(out / "state.json", {"status": "stopped", "reason": str(exc), "updated_at": time.time()})
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
