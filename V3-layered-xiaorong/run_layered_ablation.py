#!/usr/bin/env python3
"""True-online layered and factorial ablation for OCR-DOTA-V3."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V3 = REPO / "OCR-DOTA-V3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(V3))

import tune_nonimagenet21 as core  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402


VERSION = "ocr-dota-v3-layered-factorial-online-v1"
DEFAULT_WINNERS = V3 / "BEST_SINGLE_CONFIG_NONIMAGENET21_AFTER_CONTINUATION.json"
DEFAULT_OUTPUT = HERE / "results" / "classic10_layered_20260902"
DEFAULT_DATASETS = core.TEN_DATASETS
FACTORS = ("residual", "prediction_rank", "update_rank")
MASKS = tuple(itertools.product((0, 1), repeat=3))
BASELINE_VARIANTS = ("original_dota", "tuned_base_dota")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def parse_datasets(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_DATASETS
    result = tuple(core.normalize_dataset(item) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("datasets must be non-empty and unique")
    return result


def arm_name(mask: tuple[int, int, int]) -> str:
    return f"G{mask[0]}P{mask[1]}U{mask[2]}"


def factorial_config(full: Mapping[str, Any], mask: tuple[int, int, int]) -> dict[str, Any]:
    result = copy.deepcopy(dict(full))
    result["update"]["residual_strength"] = float(full["update"]["residual_strength"]) if mask[0] else 0.0
    result["rank"]["prediction_strength"] = float(full["rank"]["prediction_strength"]) if mask[1] else 0.0
    result["rank"]["update_power"] = float(full["rank"]["update_power"]) if mask[2] else 0.0
    core.validate_config(result)
    return result


def compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "correct", "num_samples", "accuracy", "prediction_sha256",
        "trajectory_sha256", "state_sha256", "compatibility_sha256",
        "health_status", "elapsed_sec", "peak_cuda_bytes",
        "mean_rank_compatibility", "mean_update_gate", "mean_residual_magnitude",
        "mean_update_mass", "state_health",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def shapley(values: Mapping[tuple[int, int, int], float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for player, name in enumerate(FACTORS):
        others = [index for index in range(3) if index != player]
        total = 0.0
        for size in range(3):
            for subset in itertools.combinations(others, size):
                before = [0, 0, 0]
                for item in subset:
                    before[item] = 1
                after = list(before)
                after[player] = 1
                weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
                total += weight * (values[tuple(after)] - values[tuple(before)])
        result[name] = total
    return result


def contribution_shares(values: Mapping[str, float]) -> dict[str, dict[str, float | None]]:
    signed = sum(values.values())
    magnitude = sum(abs(value) for value in values.values())
    positive = sum(max(0.0, value) for value in values.values())
    return {
        name: {
            "signed_percent": None if abs(signed) < 1e-12 else 100.0 * value / signed,
            "magnitude_percent": None if magnitude == 0 else 100.0 * abs(value) / magnitude,
            "positive_percent": None if positive == 0 else 100.0 * max(0.0, value) / positive,
        }
        for name, value in values.items()
    }


def assert_reproduced(reference: Mapping[str, Any], replay: Mapping[str, Any], label: str) -> dict[str, Any]:
    keys = ["correct", "num_samples", "prediction_sha256", "trajectory_sha256", "state_sha256"]
    if "compatibility_sha256" in reference:
        keys.append("compatibility_sha256")
    mismatches = {key: [reference.get(key), replay.get(key)] for key in keys if reference.get(key) != replay.get(key)}
    if mismatches:
        raise RuntimeError(f"cold replay mismatch {label}: {mismatches}")
    return {"keys": keys, "reference": {key: reference[key] for key in keys}, "replay": {key: replay[key] for key in keys}}


def build_summary(datasets: tuple[str, ...], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [dict(row) for row in rows if row.get("status") == "ok"]
    table = {(row["dataset"], row["variant"]): row for row in successful}
    per_dataset: dict[str, Any] = {}
    required = set(BASELINE_VARIANTS) | {arm_name(mask) for mask in MASKS}
    for dataset in datasets:
        if any((dataset, variant) not in table for variant in required):
            continue
        original = table[(dataset, "original_dota")]
        tuned = table[(dataset, "tuned_base_dota")]
        arms = {mask: table[(dataset, arm_name(mask))] for mask in MASKS}
        path = {
            "base_tuning": float(tuned["correct"] - original["correct"]),
            "enhanced_core": float(arms[(0, 0, 0)]["correct"] - tuned["correct"]),
            "residual": float(arms[(1, 0, 0)]["correct"] - arms[(0, 0, 0)]["correct"]),
            "prediction_rank": float(arms[(1, 1, 0)]["correct"] - arms[(1, 0, 0)]["correct"]),
            "update_rank": float(arms[(1, 1, 1)]["correct"] - arms[(1, 1, 0)]["correct"]),
        }
        factor_values = {mask: float(row["correct"]) for mask, row in arms.items()}
        factor_phi = shapley(factor_values)
        per_dataset[dataset] = {
            "num_samples": int(original["num_samples"]),
            "stages": {
                "original_dota": {"correct": original["correct"], "accuracy": original["accuracy"]},
                "tuned_base_dota": {"correct": tuned["correct"], "accuracy": tuned["accuracy"]},
                "enhanced_core": {"correct": arms[(0, 0, 0)]["correct"], "accuracy": arms[(0, 0, 0)]["accuracy"]},
                "plus_residual": {"correct": arms[(1, 0, 0)]["correct"], "accuracy": arms[(1, 0, 0)]["accuracy"]},
                "plus_prediction_rank": {"correct": arms[(1, 1, 0)]["correct"], "accuracy": arms[(1, 1, 0)]["accuracy"]},
                "full_v3": {"correct": arms[(1, 1, 1)]["correct"], "accuracy": arms[(1, 1, 1)]["accuracy"]},
            },
            "sequential_contribution_correct": path,
            "sequential_shares": contribution_shares(path),
            "factorial_shapley_correct": factor_phi,
            "factorial_shares": contribution_shares(factor_phi),
            "factorial_arms": {arm_name(mask): {"correct": row["correct"], "accuracy": row["accuracy"]} for mask, row in arms.items()},
            "delta_full_vs_original_correct": float(arms[(1, 1, 1)]["correct"] - original["correct"]),
        }
    aggregate = None
    if len(per_dataset) == len(datasets):
        total_n = sum(row["num_samples"] for row in per_dataset.values())
        stage_names = ("original_dota", "tuned_base_dota", "enhanced_core", "plus_residual", "plus_prediction_rank", "full_v3")
        micro_stage = {stage: sum(row["stages"][stage]["correct"] for row in per_dataset.values()) for stage in stage_names}
        macro_stage = {stage: sum(row["stages"][stage]["accuracy"] for row in per_dataset.values()) / len(datasets) for stage in stage_names}
        sequential_micro = {
            "base_tuning": micro_stage["tuned_base_dota"] - micro_stage["original_dota"],
            "enhanced_core": micro_stage["enhanced_core"] - micro_stage["tuned_base_dota"],
            "residual": micro_stage["plus_residual"] - micro_stage["enhanced_core"],
            "prediction_rank": micro_stage["plus_prediction_rank"] - micro_stage["plus_residual"],
            "update_rank": micro_stage["full_v3"] - micro_stage["plus_prediction_rank"],
        }
        sequential_macro = {
            "base_tuning": macro_stage["tuned_base_dota"] - macro_stage["original_dota"],
            "enhanced_core": macro_stage["enhanced_core"] - macro_stage["tuned_base_dota"],
            "residual": macro_stage["plus_residual"] - macro_stage["enhanced_core"],
            "prediction_rank": macro_stage["plus_prediction_rank"] - macro_stage["plus_residual"],
            "update_rank": macro_stage["full_v3"] - macro_stage["plus_prediction_rank"],
        }
        factorial_micro_values = {
            mask: sum(per_dataset[dataset]["factorial_arms"][arm_name(mask)]["correct"] for dataset in datasets)
            for mask in MASKS
        }
        factorial_macro_values = {
            mask: sum(per_dataset[dataset]["factorial_arms"][arm_name(mask)]["accuracy"] for dataset in datasets) / len(datasets)
            for mask in MASKS
        }
        micro_phi = shapley(factorial_micro_values)
        macro_phi = shapley(factorial_macro_values)
        aggregate = {
            "num_samples": total_n,
            "micro_stage_correct": micro_stage,
            "micro_stage_accuracy": {stage: 100.0 * correct / total_n for stage, correct in micro_stage.items()},
            "macro_stage_accuracy": macro_stage,
            "sequential_micro_correct": sequential_micro,
            "sequential_micro_shares": contribution_shares(sequential_micro),
            "sequential_macro_pp": sequential_macro,
            "sequential_macro_shares": contribution_shares(sequential_macro),
            "factorial_micro_shapley_correct": micro_phi,
            "factorial_micro_shares": contribution_shares(micro_phi),
            "factorial_macro_shapley_pp": macro_phi,
            "factorial_macro_shares": contribution_shares(macro_phi),
            "delta_full_vs_original_correct": micro_stage["full_v3"] - micro_stage["original_dota"],
            "delta_full_vs_original_micro_pp": 100.0 * (micro_stage["full_v3"] - micro_stage["original_dota"]) / total_n,
            "delta_full_vs_original_macro_pp": macro_stage["full_v3"] - macro_stage["original_dota"],
        }
    return {
        "status": "complete" if aggregate is not None else "running",
        "protocol": "full-stream true-online layered diagnostic using frozen per-dataset V3 winners",
        "per_dataset": per_dataset,
        "aggregate": aggregate,
    }


def write_reports(out: Path, summary: Mapping[str, Any]) -> None:
    core.atomic_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "N", "original_dota", "tuned_base_dota", "enhanced_core", "plus_residual", "plus_prediction_rank", "full_v3"))
        for dataset, row in summary["per_dataset"].items():
            writer.writerow((dataset, row["num_samples"], *(row["stages"][stage]["accuracy"] for stage in ("original_dota", "tuned_base_dota", "enhanced_core", "plus_residual", "plus_prediction_rank", "full_v3"))))
    lines = ["OCR-DOTA-V3 分层真实在线消融", "口径：冻结冠军的全量流机制诊断，不是无偏泛化评估", ""]
    aggregate = summary.get("aggregate")
    if aggregate:
        lines += [f"总样本: {aggregate['num_samples']}", f"Full V3 vs Original DOTA: {aggregate['delta_full_vs_original_correct']:+.0f} samples / {aggregate['delta_full_vs_original_micro_pp']:+.6f}pp micro / {aggregate['delta_full_vs_original_macro_pp']:+.6f}pp macro", "", "Micro逐层贡献："]
        for name, value in aggregate["sequential_micro_correct"].items():
            share = aggregate["sequential_micro_shares"][name]
            positive = "N/A" if share["positive_percent"] is None else f"{share['positive_percent']:.3f}%"
            lines.append(f"- {name}: {value:+.3f} samples; positive_share={positive}")
        lines += ["", "V3内部G/P/U Shapley："]
        for name, value in aggregate["factorial_micro_shapley_correct"].items():
            share = aggregate["factorial_micro_shares"][name]
            magnitude = "N/A" if share["magnitude_percent"] is None else f"{share['magnitude_percent']:.3f}%"
            lines.append(f"- {name}: {value:+.3f} samples; magnitude_share={magnitude}")
    lines += ["", "逐数据集："]
    for dataset, row in summary["per_dataset"].items():
        stages = row["stages"]
        lines.append(f"- {dataset}: {stages['original_dota']['correct']} -> {stages['tuned_base_dota']['correct']} -> {stages['enhanced_core']['correct']} -> {stages['plus_residual']['correct']} -> {stages['plus_prediction_rank']['correct']} -> {stages['full_v3']['correct']} (Full-Original {row['delta_full_vs_original_correct']:+.0f})")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default=str(DEFAULT_OUTPUT))
    value.add_argument("--datasets")
    value.add_argument("--device", default="cuda")
    value.add_argument("--global-seed", type=int, default=1)
    value.add_argument("--stop-check-interval", type=int, default=25)
    value.add_argument("--max-samples", type=int, help="smoke only")
    value.add_argument("--winner-source", default=str(DEFAULT_WINNERS))
    value.add_argument("--resume", action="store_true")
    value.add_argument("--skip-verification", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    datasets = parse_datasets(args.datasets)
    out = Path(args.output).resolve()
    stop = out / "STOP"
    winner_source = Path(args.winner_source).resolve()
    winners = json.loads(winner_source.read_text(encoding="utf-8"))["datasets"]
    missing = set(datasets) - set(winners)
    if missing:
        raise RuntimeError(f"winner source misses {sorted(missing)}")
    original_bases = {dataset: core.load_original_base(REPO, dataset) for dataset in datasets}
    cache_paths = {dataset: core.cache_path(dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT) for dataset in datasets}
    cache_manifest = json.loads(core.EXPECTED_CACHE_MANIFEST.read_text(encoding="utf-8"))
    identity = {
        "version": VERSION,
        "datasets": datasets,
        "global_seed": args.global_seed,
        "max_samples": args.max_samples,
        "precision": "fp32",
        "winner_source": str(winner_source),
        "winner_source_sha256": core.sha256_file(winner_source),
        "code_sha256": {
            "runner": core.sha256_file(Path(__file__).resolve()),
            "v3_model": core.sha256_file(V3 / "ocr_dota_v3/model.py"),
            "rank_compatibility": core.sha256_file(V3 / "ocr_dota_v3/rank_compatibility.py"),
            "v3_tuner": core.sha256_file(V3 / "tune_nonimagenet21.py"),
            "legacy_runner": core.sha256_file(REPO / "scripts/run_cross_benchmark_ocr_ablation.py"),
        },
        "original_bases": original_bases,
        "caches": {dataset: {"path": str(cache_paths[dataset]), "sha256": core.sha256_file(cache_paths[dataset]), "num_samples": int(cache_manifest[dataset]["num_samples"]), "order_sha256": cache_manifest[dataset]["order_sha256"]} for dataset in datasets},
        "variants": [*BASELINE_VARIANTS, *(arm_name(mask) for mask in MASKS)],
    }
    identity_sha = stable_sha(identity)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or previous.get("identity_sha256") != identity_sha:
            raise RuntimeError("resume identity mismatch or --resume missing")
    else:
        core.atomic_json(manifest_path, {"status": "running", "identity": identity, "identity_sha256": identity_sha, "started_at": time.time()})
    results_path = out / "results.jsonl"
    verification_path = out / "verification.jsonl"
    results = core.load_jsonl(results_path)
    done = {(row.get("dataset"), row.get("variant")): row for row in results if row.get("status") == "ok" and row.get("identity_sha256") == identity_sha}
    verification_rows = core.load_jsonl(verification_path)
    verified = {(row.get("dataset"), row.get("variant")): row for row in verification_rows if row.get("status") == "reproduced" and row.get("identity_sha256") == identity_sha}
    invariants_path = out / "invariants.json"
    invariants = json.loads(invariants_path.read_text(encoding="utf-8")) if invariants_path.exists() else {"datasets": {}}

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
                    raise RuntimeError(f"cache order or size mismatch: {dataset}")
                full_config = winners[dataset]["config"]
                variants: list[tuple[str, str, Any]] = [
                    ("original_dota", "dota", original_bases[dataset]),
                    ("tuned_base_dota", "dota", full_config["base"]),
                ] + [(arm_name(mask), "v3", factorial_config(full_config, mask)) for mask in MASKS]
                for variant_index, (variant, kind, config) in enumerate(variants):
                    key = (dataset, variant)
                    if key not in done:
                        core.atomic_json(out / "state.json", {"status": "running", "dataset": dataset, "dataset_index": dataset_index, "variant": variant, "variant_index": variant_index, "updated_at": time.time()})
                        core.setup_seed(args.global_seed)
                        started = time.time()
                        try:
                            metrics = core.replay_dota(data, config, stop, args.stop_check_interval) if kind == "dota" else core.replay_v3(data, config, stop, args.stop_check_interval)
                            status, error = "ok", None
                        except core.StopRequested:
                            raise
                        except Exception as exc:
                            status, error = "error", f"{type(exc).__name__}: {exc}"
                            metrics = {"elapsed_sec": time.time() - started, "health_status": "unhealthy"}
                        row = {"status": status, "error": error, "identity_sha256": identity_sha, "dataset": dataset, "dataset_index": dataset_index, "variant": variant, "kind": kind, "config": config, "config_sha256": stable_sha(config), "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"], **compact(metrics)}
                        if kind == "v3" and variant.endswith("U0") and abs(float(row["mean_update_gate"]) - 1.0) > 1e-7:
                            raise RuntimeError(f"update gate is not exact identity for {dataset}/{variant}: {row['mean_update_gate']}")
                        core.append_jsonl(results_path, row)
                        results.append(row)
                        if status != "ok":
                            raise RuntimeError(f"failed {dataset}/{variant}: {error}")
                        done[key] = row
                    if not args.skip_verification and key not in verified:
                        core.setup_seed(args.global_seed)
                        replayed = core.replay_dota(data, config, stop, args.stop_check_interval) if kind == "dota" else core.replay_v3(data, config, stop, args.stop_check_interval)
                        audit = assert_reproduced(done[key], replayed, f"{dataset}/{variant}")
                        record = {"status": "reproduced", "identity_sha256": identity_sha, "dataset": dataset, "variant": variant, **audit, "elapsed_sec": replayed["elapsed_sec"]}
                        core.append_jsonl(verification_path, record)
                        verified[key] = record
                    write_reports(out, build_summary(datasets, results))
                prediction_only_checks = []
                for geometry in (0, 1):
                    for update in (0, 1):
                        left = done[(dataset, arm_name((geometry, 0, update)))]
                        right = done[(dataset, arm_name((geometry, 1, update)))]
                        # core.replay_v3 seeds trajectory_sha256 with the full
                        # canonical config, so P0/P1 hashes differ even when
                        # their update weights are identical.  State and
                        # compatibility hashes are config-prefix independent
                        # and are the valid cross-arm invariants here.
                        fields = ("state_sha256", "compatibility_sha256")
                        mismatches = {field: [left[field], right[field]] for field in fields if left[field] != right[field]}
                        if mismatches:
                            raise RuntimeError(f"prediction-rank polluted update state for {dataset}/G{geometry}U{update}: {mismatches}")
                        prediction_only_checks.append({"geometry": geometry, "update_rank": update, "fields": fields, "status": "identical"})
                invariants["datasets"][dataset] = {
                    "prediction_rank_does_not_change_update": prediction_only_checks,
                    "update_gate_off_is_identity": {
                        arm_name(mask): done[(dataset, arm_name(mask))]["mean_update_gate"]
                        for mask in MASKS if mask[2] == 0
                    },
                    "status": "passed",
                }
                core.atomic_json(invariants_path, invariants)
                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            summary = build_summary(datasets, results)
            write_reports(out, summary)
            status = "complete_smoke" if args.max_samples is not None else "complete"
            core.atomic_json(out / "state.json", {"status": status, "datasets": len(datasets), "variants": 10, "verified": len(verified), "finished_at": time.time()})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"status": status, "finished_at": time.time(), "summary_sha256": core.sha256_file(out / "summary.json")})
            core.atomic_json(manifest_path, manifest)
        except core.StopRequested as exc:
            core.atomic_json(out / "state.json", {"status": "stopped", "reason": str(exc), "updated_at": time.time()})
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
