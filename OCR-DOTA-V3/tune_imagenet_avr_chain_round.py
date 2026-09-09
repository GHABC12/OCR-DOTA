#!/usr/bin/env python3
"""One sealed fine-search generation in the AVR continuation chain.

This is an independent continuation runner.  It consumes a sealed parent run,
never mutates it, excludes every configuration already evaluated by the
parent, and writes a new self-contained result tree.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


V3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V3_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V3_ROOT))

import tune_nonimagenet21 as core  # noqa: E402


VERSION = "ocr-dota-v3-imagenet-avr-chain-round-v1"
DATASETS = ("A", "R", "V")
STAGES = ("rank_refine", "update_refine", "base_refine")
DEFAULT_OUTPUT = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_imagenet_avr_chain_manual"
DEFAULT_PARENT = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_imagenet_avr_fixed_order_6h_20260901"
CACHE_ROOT = REPO_ROOT / "log/all_dataset_perf/cache"

REGISTRY: dict[str, dict[str, Any]] = {
    "A": {
        "cache": "A_vitb16.pt", "cache_sha256": "505ff03f191f38fb12adc2c09bd305023b90968d370e6294a892ec83fc49e702",
        "num_samples": 7500, "num_classes": 200, "sample_order_seed": 152,
        "sample_order_sha256": "b265ef0e9aeac95702d27d21a0aef53588c137d6b93200ddbd5b97bceed082ac",
        "anchor": "order", "original_config": "configs/vit/imagenet_a.yaml",
    },
    "R": {
        "cache": "R_vitb16.pt", "cache_sha256": "68f048d3934161b479f4b6c6be9bca79ecd343666f793e8ce58e3465bcda82d0",
        "num_samples": 30000, "num_classes": 200, "sample_order_seed": None,
        "sample_order_sha256": "7146f11166efda28ca9ca43d607a7b85de395dda7fb78283479068313b757972",
        "anchor": "order", "original_config": "configs/vit/imagenet_r.yaml",
    },
    "V": {
        "cache": "V_vitb16.pt", "cache_sha256": "af4acd16315cd02212e0c48f06402dce697e9109422dccd86feb93c13f7d471a",
        "num_samples": 10000, "num_classes": 1000, "sample_order_seed": 20,
        "sample_order_sha256": "5c72e762340104ee441aca8a53022d3206ff8cc838b559e4721e75d666c496d2",
        "anchor": "latest_config_order_identity", "original_config": "configs/vit/imagenet_v.yaml",
    },
}

HISTORICAL_SECONDS = {"A": 75.0, "R": 300.0, "V": 240.0}
REQUIRED_KEYS = {"features", "clip_logits", "prob_maps", "targets", "text_prototypes", "clip_shape"}


def parse_datasets(value: str | None) -> tuple[str, ...]:
    if not value:
        return DATASETS
    result = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if len(result) != len(set(result)) or set(result) - set(DATASETS):
        raise ValueError(f"datasets must be a unique subset of {DATASETS}")
    return result


def parse_stages(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or set(result) - set(STAGES):
        raise ValueError(f"stages must be a subset of {STAGES}")
    indices = tuple(STAGES.index(item) for item in result)
    if indices != tuple(sorted(indices)):
        raise ValueError("stages must preserve canonical order")
    return result


def permutation_for(count: int, seed: int | None) -> torch.Tensor:
    if seed is None:
        return torch.arange(count, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(count, generator=generator)


def int64_sha(value: torch.Tensor | np.ndarray) -> str:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def validate_raw_cache(dataset: str, raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
    spec = REGISTRY[dataset]
    missing = REQUIRED_KEYS - set(raw)
    if missing:
        raise RuntimeError(f"{dataset} cache lacks keys: {sorted(missing)}")
    count, classes = int(spec["num_samples"]), int(spec["num_classes"])
    if int(raw["features"].shape[0]) != count or tuple(raw["features"].shape[1:]) != (6, 512):
        raise RuntimeError(f"{dataset} feature shape mismatch: {tuple(raw['features'].shape)}")
    expected_shapes = {
        "clip_logits": (count, classes), "prob_maps": (count, 6, classes), "targets": (count,),
        "text_prototypes": (classes, 512),
    }
    for key, shape in expected_shapes.items():
        if tuple(raw[key].shape) != shape:
            raise RuntimeError(f"{dataset} {key} shape mismatch: {tuple(raw[key].shape)} != {shape}")
    if tuple(int(x) for x in raw["clip_shape"]) != (512, classes):
        raise RuntimeError(f"{dataset} clip_shape mismatch")
    ids = raw.get("sample_ids", torch.arange(count, dtype=torch.long))
    if not torch.is_tensor(ids) or tuple(ids.shape) != (count,):
        raise RuntimeError(f"{dataset} sample_ids must be a length-{count} tensor")
    order = permutation_for(count, spec["sample_order_seed"])
    permutation_sha = int64_sha(order)
    if permutation_sha != spec["sample_order_sha256"]:
        raise RuntimeError(f"{dataset} fixed permutation SHA mismatch")
    ordered_ids_sha = int64_sha(ids.index_select(0, order))
    if dataset == "V":
        meta = raw.get("meta") or {}
        if meta.get("synset_aware_classnames") is not True:
            raise RuntimeError("V cache is not the synset-aware cache")
    return {
        "path": str(path), "sha256": core.sha256_file(path), "bytes": path.stat().st_size,
        "num_samples": count, "num_classes": classes, "sample_order_seed": spec["sample_order_seed"],
        "permutation_sha256": permutation_sha, "ordered_sample_ids_sha256": ordered_ids_sha,
        "legacy_meta": not bool(raw.get("meta")),
    }


def inspect_cache(dataset: str, cache_root: Path) -> dict[str, Any]:
    path = cache_root / REGISTRY[dataset]["cache"]
    if not path.exists() or core.sha256_file(path) != REGISTRY[dataset]["cache_sha256"]:
        raise RuntimeError(f"{dataset} cache file/SHA mismatch: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return validate_raw_cache(dataset, raw, path)
    finally:
        del raw


def load_fixed_cache(
    dataset: str, cache_root: Path, device: str, max_samples: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = cache_root / REGISTRY[dataset]["cache"]
    if core.sha256_file(path) != REGISTRY[dataset]["cache_sha256"]:
        raise RuntimeError(f"{dataset} cache SHA mismatch")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    meta = validate_raw_cache(dataset, raw, path)
    full_count = int(REGISTRY[dataset]["num_samples"])
    full_order = permutation_for(full_count, REGISTRY[dataset]["sample_order_seed"])
    used = full_count if max_samples is None else min(full_count, int(max_samples))
    index = full_order[:used]
    ids = raw.get("sample_ids", torch.arange(full_count, dtype=torch.long)).index_select(0, index)
    data = dict(raw)
    for key in ("features", "clip_logits", "prob_maps", "targets"):
        data[key] = raw[key].index_select(0, index).to(device=device)
    data["text_prototypes"] = raw["text_prototypes"].to(device=device, dtype=torch.float32)
    data["sample_ids"] = ids.detach().cpu().numpy().astype(np.int64, copy=False)
    data["sample_order_seed"] = REGISTRY[dataset]["sample_order_seed"]
    meta.update(
        full_count=full_count, num_samples=used,
        order_sha256=int64_sha(ids), full_permutation_sha256=int64_sha(full_order),
    )
    return data, meta


def unwrap_config(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("config", row)
    return {
        "base": copy.deepcopy(value["base"]), "ocr": copy.deepcopy(value["ocr"]),
        "update_rule": value.get("update_rule", row.get("update_rule", "mix08clip")),
    }


def load_parent(parent: Path, datasets: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[tuple[str, str]]]:
    required = ("manifest.json", "winners.json", "summary.json", "candidate_results.jsonl")
    if any(not (parent / name).is_file() for name in required):
        raise RuntimeError(f"parent run is incomplete: {parent}")
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "complete_budget_limited"}:
        raise RuntimeError(f"parent run is not sealed: {manifest.get('status')}")
    winners_payload = json.loads((parent / "winners.json").read_text(encoding="utf-8"))
    if winners_payload.get("status") not in {"complete", "complete_budget_limited"}:
        raise RuntimeError("parent winners are not sealed")
    table = winners_payload["datasets"]
    anchors: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for dataset in datasets:
        row = table[dataset]
        if row.get("sample_order_seed") != REGISTRY[dataset]["sample_order_seed"]:
            raise RuntimeError(f"{dataset} parent seed mismatch")
        if row.get("sample_order_sha256") != REGISTRY[dataset]["sample_order_sha256"]:
            raise RuntimeError(f"{dataset} parent order SHA mismatch")
        if row.get("cache_sha256") != REGISTRY[dataset]["cache_sha256"]:
            raise RuntimeError(f"{dataset} parent cache SHA mismatch")
        if any(row.get("verification", {}).get(key) != row.get(key) for key in ("correct", "prediction_sha256", "state_sha256", "trajectory_sha256")):
            raise RuntimeError(f"{dataset} parent winner is not exactly reproduced")
        anchors[dataset] = copy.deepcopy(row["config"])
        provenance[dataset] = {
            "parent": str(parent), "parent_candidate_id": row["candidate_id"],
            "parent_fingerprint": row["fingerprint"], "parent_correct": int(row["correct"]),
            "sample_order_seed": REGISTRY[dataset]["sample_order_seed"],
            "sample_order_sha256": REGISTRY[dataset]["sample_order_sha256"],
        }
    historical: set[tuple[str, str]] = set()
    cursor, seen = parent, set()
    while cursor not in seen:
        seen.add(cursor)
        result_file = cursor / "candidate_results.jsonl"
        winner_file = cursor / "winners.json"
        if result_file.is_file():
            historical.update((str(row["dataset"]), str(row["fingerprint"])) for row in core.load_jsonl(result_file) if row.get("status") == "ok")
        if winner_file.is_file():
            rows = json.loads(winner_file.read_text(encoding="utf-8")).get("datasets", {})
            historical.update((d, str(row["fingerprint"])) for d, row in rows.items())
        manifest_file = cursor / "manifest.json"
        if not manifest_file.is_file(): break
        prior = json.loads(manifest_file.read_text(encoding="utf-8")).get("identity", {}).get("parent_run")
        if not prior: break
        cursor = Path(prior).resolve()
    return anchors, provenance, table, historical


def load_original_bases(repo_root: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    bases, hashes = {}, {}
    for dataset, spec in REGISTRY.items():
        path = repo_root / spec["original_config"]
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = {key: float(raw[key]) for key in ("epsilon", "sigma", "eta", "rho")}
        if any(not math.isfinite(value) or value <= 0 for value in base.values()):
            raise ValueError(f"invalid original DOTA config for {dataset}")
        bases[dataset], hashes[dataset] = base, core.sha256_file(path)
    return bases, hashes


def spec(config: Mapping[str, Any], label: str, candidate_id: str | None = None, **changes: Any) -> dict[str, Any]:
    result = core.changed(config, label, **changes)
    if candidate_id is not None:
        result["candidate_id"] = candidate_id
    return result


def generate_stage(dataset: str, stage: str, incumbent: Mapping[str, Any], round_index: int = 3) -> list[dict[str, Any]]:
    cfg = incumbent.get("config", incumbent)
    specs: list[dict[str, Any]] = []
    scale = .5 ** max(0, round_index - 3)
    if stage == "rank_refine":
        rank_spaces = []
        for axis in ("prediction_strength", "tau_rank"):
            center = float(cfg["rank"][axis]); delta = .05 * scale
            rank_spaces.append((axis, (center * (1 - delta), center * (1 + delta))))
        power = float(cfg["rank"]["update_power"])
        values = (power * (1 - .10 * scale), power * (1 + .10 * scale)) if power > 0 else (.125 * scale, .25 * scale)
        rank_spaces.append(("update_power", values))
        for axis, values in rank_spaces:
            specs.extend(spec(cfg, f"R_{axis}_{value:g}", **{f"rank__{axis}": value}) for value in values)
    elif stage == "update_refine":
        residual = float(cfg["update"]["residual_strength"]); step = .03125 * scale
        residual_values = (max(0., residual - step), min(1., residual + step))
        specs.extend(spec(cfg, f"U_residual_{value:g}", update__residual_strength=value) for value in residual_values)
        beta = float(cfg["update"]["beta"])
        count = float(cfg["update"]["init_count"])
        factors = (1 - .05 * scale, 1 + .05 * scale)
        specs.extend(spec(cfg, f"U_beta_x{factor:g}", update__beta=min(1., max(0., beta * factor))) for factor in factors)
        specs.extend(spec(cfg, f"U_count_x{factor:g}", update__init_count=count * factor) for factor in factors)
    elif stage == "base_refine":
        for axis in ("eta", "rho"):
            delta = .025 * scale
            factors = (1 - delta, 1 + delta)
            for factor in factors:
                specs.append(spec(cfg, f"B_{axis}_x{factor:g}", **{f"base__{axis}": float(cfg["base"][axis]) * factor}))
    else:
        raise ValueError(stage)
    return core.unique_candidates(dataset, specs)


def build_identity(
    *, datasets: tuple[str, ...], stages: tuple[str, ...], global_seed: int,
    max_samples: int | None, time_budget_hours: float, verification_reserve_hours: float,
    cache_records: Mapping[str, Any], anchors: Mapping[str, Any], provenance: Mapping[str, Any],
    original_bases: Mapping[str, Any], source_shas: Mapping[str, str], code_shas: Mapping[str, str],
    parent_run: str, historical_fingerprint_sha256: str, round_index: int,
) -> dict[str, Any]:
    identity = {
        "version": VERSION, "protocol": "fixed-order per-dataset full-stream tuning oracle",
        "formal": max_samples is None, "datasets": datasets, "stages": stages,
        "global_seed": int(global_seed), "max_samples": max_samples,
        "precision": "fp32", "worker_count": 1,
        "time_budget_hours": float(time_budget_hours),
        "verification_reserve_hours": float(verification_reserve_hours),
        "caches": dict(cache_records), "anchors_sha256": core.stable_sha(anchors),
        "anchor_provenance": dict(provenance), "original_bases": dict(original_bases),
        "source_shas": dict(source_shas), "code_shas": dict(code_shas),
        "parent_run": parent_run, "historical_fingerprint_sha256": historical_fingerprint_sha256,
        "round_index": int(round_index),
    }
    return {"identity": identity, "identity_sha256": core.stable_sha(identity)}


def write_summary(
    out: Path, datasets: tuple[str, ...], original: Mapping[str, Any],
    matched: Mapping[str, Any], winners: Mapping[str, Any], status: str,
) -> None:
    rows, totals = {}, {"n": 0, "original": 0, "matched": 0, "v3": 0}
    for dataset in datasets:
        if dataset not in original or dataset not in matched or dataset not in winners:
            continue
        fixed, paired, winner = original[dataset], matched[dataset], winners[dataset]
        rows[dataset] = {
            "sample_order_seed": REGISTRY[dataset]["sample_order_seed"],
            "sample_order_sha256": REGISTRY[dataset]["sample_order_sha256"],
            "selection_scope": "fixed_order_tuning_oracle" if dataset in {"A", "V"} else "canonical_order_tuning_oracle",
            "original_dota_fixed": fixed, "matched_base_dota": paired, "v3_winner": winner,
            "delta_vs_original_correct": winner["correct"] - fixed["correct"],
            "delta_vs_original_pp": winner["accuracy"] - fixed["accuracy"],
            "delta_vs_matched_correct": winner["correct"] - paired["correct"],
            "delta_vs_matched_pp": winner["accuracy"] - paired["accuracy"],
        }
        totals["n"] += winner["num_samples"]
        totals["original"] += fixed["correct"]
        totals["matched"] += paired["correct"]
        totals["v3"] += winner["correct"]
    micro = None
    macro = None
    if rows:
        n = totals["n"]
        micro = {
            "num_samples": n, "original_correct": totals["original"], "matched_correct": totals["matched"], "v3_correct": totals["v3"],
            "original_accuracy": 100 * totals["original"] / n, "matched_accuracy": 100 * totals["matched"] / n,
            "v3_accuracy": 100 * totals["v3"] / n, "delta_vs_original_pp": 100 * (totals["v3"] - totals["original"]) / n,
            "delta_vs_matched_pp": 100 * (totals["v3"] - totals["matched"]) / n,
        }
        macro = {
            "streams": len(rows),
            "original_accuracy": sum(row["original_dota_fixed"]["accuracy"] for row in rows.values()) / len(rows),
            "matched_accuracy": sum(row["matched_base_dota"]["accuracy"] for row in rows.values()) / len(rows),
            "v3_accuracy": sum(row["v3_winner"]["accuracy"] for row in rows.values()) / len(rows),
        }
        macro["delta_vs_original_pp"] = macro["v3_accuracy"] - macro["original_accuracy"]
        macro["delta_vs_matched_pp"] = macro["v3_accuracy"] - macro["matched_accuracy"]
    payload = {"status": status, "protocol": "A seed152 / V seed20 fixed-order oracle; R canonical-order oracle", "datasets": rows, "micro": micro, "macro": macro}
    core.atomic_json(out / "summary.json", payload)
    core.atomic_json(out / "winners.json", {"status": status, "datasets": winners})
    if rows:
        temporary = out / f".summary.csv.tmp.{os.getpid()}"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "N", "seed", "original_dota", "matched_base_dota", "v3", "delta_original", "delta_matched"))
            for dataset in datasets:
                if dataset in rows:
                    row = rows[dataset]
                    writer.writerow((dataset, row["v3_winner"]["num_samples"], row["sample_order_seed"],
                                     row["original_dota_fixed"]["accuracy"], row["matched_base_dota"]["accuracy"],
                                     row["v3_winner"]["accuracy"], row["delta_vs_original_pp"], row["delta_vs_matched_pp"]))
        os.replace(temporary, out / "summary.csv")
        lines = ["OCR-DOTA-V3 ImageNet-A/R/V fixed-order continuation tuning oracle", ""]
        for dataset in datasets:
            if dataset in rows:
                row = rows[dataset]
                lines.append(f"{dataset}: Original {row['original_dota_fixed']['accuracy']:.6f}% | Matched {row['matched_base_dota']['accuracy']:.6f}% | V3 {row['v3_winner']['accuracy']:.6f}% | ΔOriginal {row['delta_vs_original_pp']:+.6f}pp | ΔMatched {row['delta_vs_matched_pp']:+.6f}pp")
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-root", default=str(CACHE_ROOT))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--datasets")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--global-seed", type=int, default=1)
    parser.add_argument("--stop-check-interval", type=int, default=25)
    parser.add_argument("--stop")
    parser.add_argument("--time-budget-hours", type=float, default=8.0)
    parser.add_argument("--verification-reserve-hours", type=float, default=1.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, help="smoke only; formal winner output is disabled")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo_root).resolve()
    if repo != REPO_ROOT:
        raise RuntimeError(f"runner is bound to {REPO_ROOT}, got {repo}")
    if args.global_seed != 1:
        raise ValueError("formal AVR protocol fixes global_seed=1")
    if args.time_budget_hours <= 0 or args.verification_reserve_hours <= 0 or args.verification_reserve_hours >= args.time_budget_hours:
        raise ValueError("invalid time budget/reserve")
    datasets, stages = parse_datasets(args.datasets), parse_stages(args.stages)
    if args.round_index < 3:
        raise ValueError("round-index must be >= 3")
    out, cache_root = Path(args.output).resolve(), Path(args.cache_root).resolve()
    parent = Path(args.parent).resolve()
    anchors_all, provenance_all, parent_winners_all, historical_all = load_parent(parent, datasets)
    anchors = {dataset: anchors_all[dataset] for dataset in datasets}
    provenance = {dataset: provenance_all[dataset] for dataset in datasets}
    original_all, original_hashes_all = load_original_bases(repo)
    original_bases = {dataset: original_all[dataset] for dataset in datasets}
    cache_records = {dataset: inspect_cache(dataset, cache_root) for dataset in datasets}
    identity_record = build_identity(
        datasets=datasets, stages=stages, global_seed=args.global_seed,
        max_samples=args.max_samples, time_budget_hours=args.time_budget_hours,
        verification_reserve_hours=args.verification_reserve_hours,
        cache_records=cache_records, anchors=anchors, provenance=provenance,
        original_bases=original_bases,
        source_shas={"parent_manifest": core.sha256_file(parent / "manifest.json"),
                     "parent_winners": core.sha256_file(parent / "winners.json"),
                     "parent_summary": core.sha256_file(parent / "summary.json"),
                     "parent_results": core.sha256_file(parent / "candidate_results.jsonl"),
                     **{f"original_{dataset}": original_hashes_all[dataset] for dataset in datasets}},
        code_shas={"runner": core.sha256_file(Path(__file__).resolve()),
                   "core_runner": core.sha256_file(V3_ROOT / "tune_nonimagenet21.py"),
                   "model": core.sha256_file(V3_ROOT / "ocr_dota_v3/model.py"),
                   "rank": core.sha256_file(V3_ROOT / "ocr_dota_v3/rank_compatibility.py")},
        parent_run=str(parent),
        historical_fingerprint_sha256=core.stable_sha(sorted([list(item) for item in historical_all])),
        round_index=args.round_index,
    )
    identity, identity_sha = identity_record["identity"], identity_record["identity_sha256"]
    if args.dry_run:
        plan = {dataset: {stage: len([c for c in generate_stage(dataset, stage, {"config": anchors[dataset]}, args.round_index)
                                     if (dataset, c["fingerprint"]) not in historical_all]) for stage in stages} for dataset in datasets}
        print(json.dumps({"identity_sha256": identity_sha, "candidate_counts_from_anchor": plan,
                          "historical_fingerprints": len(historical_all), "caches": cache_records}, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    manifest_path, jobs_path = out / "manifest.json", out / "jobs.json"
    results_path, state_path = out / "candidate_results.jsonl", out / "state.json"
    baselines_path, stop = out / "baselines.json", Path(args.stop).resolve() if args.stop else out / "STOP"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError("output exists; pass --resume")
        core.validate_resume_identity(old, identity_record)
        started_at = float(old["started_at"])
        search_deadline_wall = float(old["search_deadline_wall"])
        total_deadline_wall = float(old["total_deadline_wall"])
    else:
        started_at = time.time()
        search_deadline_wall = started_at + 3600 * (args.time_budget_hours - args.verification_reserve_hours)
        total_deadline_wall = started_at + 3600 * args.time_budget_hours
        core.atomic_json(manifest_path, {"status": "running", **identity_record, "started_at": started_at,
                                        "search_deadline_wall": search_deadline_wall, "total_deadline_wall": total_deadline_wall})
    search_deadline_mono = time.monotonic() + max(0., search_deadline_wall - time.time())
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else {"version": VERSION, "datasets": {}}
    results = core.load_jsonl(results_path)
    done = {(row.get("dataset"), row.get("fingerprint")): row for row in results
            if row.get("run_identity_sha256") == identity_sha and row.get("status") == "ok"}
    baseline_payload = json.loads(baselines_path.read_text(encoding="utf-8")) if baselines_path.exists() else {}
    original_results = baseline_payload.get("original_dota_fixed", {})
    matched_results = baseline_payload.get("matched_base_dota", {})
    winners: dict[str, Any] = {}

    def save_jobs() -> None:
        jobs["jobs_sha256"] = core.stable_sha(jobs.get("datasets", {}))
        core.atomic_json(jobs_path, jobs)

    def load_data(dataset: str):
        core.setup_seed(args.global_seed)
        data, meta = load_fixed_cache(dataset, cache_root, args.device, args.max_samples)
        expected = identity["caches"][dataset]
        for key in ("sha256", "sample_order_seed", "permutation_sha256", "ordered_sample_ids_sha256"):
            if meta[key] != expected[key]:
                raise RuntimeError(f"{dataset} cache/order identity changed: {key}")
        if args.max_samples is None and meta["num_samples"] != expected["num_samples"]:
            raise RuntimeError(f"{dataset} sample count mismatch")
        return data, meta

    def estimate(dataset: str) -> float:
        observed = [float(row["elapsed_sec"]) for row in results if row.get("dataset") == dataset and row.get("status") == "ok"]
        if args.max_samples is not None:
            return max(observed[-3:] or [15.]) * 1.2
        return max(observed[-3:]) if observed else HISTORICAL_SECONDS[dataset] * 1.2

    with core.PidLock(out / "RUNNING.pid"):
        try:
            stage_info = {dataset: jobs.get("datasets", {}).get(dataset, {}).get("stage_results", {}) for dataset in datasets}
            incumbents: dict[str, dict[str, Any]] = {dataset: copy.deepcopy(parent_winners_all[dataset]) for dataset in datasets}
            # Full parent winners are valid formal controls, but their full-N correct
            # counts cannot be compared with a smoke prefix.  Materialize a sealed
            # prefix control once so smoke/resume exercises the real selection path.
            if args.max_samples is not None:
                controls_path = out / "smoke_parent_controls.json"
                controls_payload = json.loads(controls_path.read_text(encoding="utf-8")) if controls_path.exists() else {}
                if controls_payload and controls_payload.get("run_identity_sha256") != identity_sha:
                    raise RuntimeError("smoke parent control identity mismatch")
                controls = controls_payload.get("datasets", {})
                for dataset in datasets:
                    if dataset not in controls:
                        data, _ = load_data(dataset)
                        core.setup_seed(args.global_seed)
                        metric = core.replay_v3(data, anchors[dataset], stop, args.stop_check_interval)
                        controls[dataset] = {**copy.deepcopy(parent_winners_all[dataset]),
                                             "stage": "parent_control", **core.compact_result(metric)}
                        core.atomic_json(controls_path, {"run_identity_sha256": identity_sha, "datasets": controls})
                        del data
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    incumbents[dataset] = controls[dataset]
            for stage in stages:
                for dataset in datasets:
                    info = stage_info[dataset].get(stage)
                    if info and info.get("status") == "complete":
                        key = (dataset, info["winner_fingerprint"])
                        parent_row = parent_winners_all[dataset]
                        if key in done:
                            incumbents[dataset] = done[key]
                        elif info["winner_fingerprint"] == parent_row["fingerprint"]:
                            pass  # keep the full parent or the materialized smoke-prefix control
                        else:
                            raise RuntimeError(f"completed stage lacks winner: {dataset}/{stage}")
            budget_exhausted = False
            for stage_index, stage in enumerate(stages):
                eligible: list[str] = []
                for dataset in datasets:
                    eligible.append(dataset)
                save_jobs()
                for dataset_index, dataset in enumerate(eligible):
                    existing = jobs.get("datasets", {}).get(dataset, {}).get("stage_results", {}).get(stage)
                    if existing and existing.get("status") == "complete":
                        continue
                    core.check_stop(stop, f"before {dataset}/{stage}")
                    if time.monotonic() + estimate(dataset) >= search_deadline_mono:
                        budget_exhausted = True
                        break
                    data, meta = load_data(dataset)
                    block = jobs["datasets"].setdefault(dataset, {"cache": meta, "stages": {}, "stage_results": {}})
                    incumbent = incumbents[dataset]
                    stage_block = block["stages"].get(stage)
                    if stage_block is None:
                        generated = generate_stage(dataset, stage, incumbent, args.round_index)
                        candidates = [candidate for candidate in generated if (dataset, candidate["fingerprint"]) not in historical_all]
                        excluded = [candidate["fingerprint"] for candidate in generated if (dataset, candidate["fingerprint"]) in historical_all]
                        stage_block = {"incumbent_fingerprint": core.fingerprint(dataset, incumbent["config"]),
                                       "candidates": candidates, "excluded_historical_fingerprints": excluded}
                        block["stages"][stage] = stage_block
                        save_jobs()
                    elif stage_block["incumbent_fingerprint"] != core.fingerprint(dataset, incumbent["config"]):
                        raise RuntimeError(f"adaptive job identity mismatch: {dataset}/{stage}")
                    candidates = stage_block["candidates"]
                    for candidate_index, candidate in enumerate(candidates):
                        key = (dataset, candidate["fingerprint"])
                        if key not in done:
                            if time.monotonic() + estimate(dataset) >= search_deadline_mono:
                                budget_exhausted = True
                                break
                            core.atomic_json(state_path, {"status": "running", "dataset": dataset, "stage": stage,
                                                           "dataset_index": dataset_index, "stage_index": stage_index,
                                                           "candidate_index": candidate_index, "candidate_count": len(candidates),
                                                           "updated_at": time.time(), "search_deadline_wall": search_deadline_wall})
                            core.setup_seed(args.global_seed)
                            started = time.time()
                            try:
                                metrics = core.replay_v3(data, candidate["config"], stop, args.stop_check_interval)
                                status, error = "ok", None
                            except core.StopRequested:
                                raise
                            except Exception as exc:
                                status, error = "error", f"{type(exc).__name__}: {exc}"
                                metrics = {"elapsed_sec": time.time() - started, "health_status": "unhealthy"}
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            row = {"status": status, "error": error, "run_identity_sha256": identity_sha,
                                   "dataset": dataset, "stage": stage, **candidate,
                                   "cache_sha256": meta["sha256"], "sample_order_seed": meta["sample_order_seed"],
                                   "full_permutation_sha256": meta["full_permutation_sha256"], "order_sha256": meta["order_sha256"],
                                   "num_samples_expected": meta["num_samples"], "global_seed": args.global_seed,
                                   "precision": "fp32", **metrics}
                            core.append_jsonl(results_path, row)
                            results.append(row)
                            if status == "ok":
                                done[key] = row
                        else:
                            row = done[key]
                            if (row.get("cache_sha256"), row.get("sample_order_seed"), row.get("full_permutation_sha256"), row.get("global_seed")) != (
                                meta["sha256"], meta["sample_order_seed"], meta["full_permutation_sha256"], args.global_seed):
                                raise RuntimeError(f"resume data identity mismatch: {dataset}/{candidate['candidate_id']}")
                    stage_rows = [done[(dataset, candidate["fingerprint"])] for candidate in candidates if (dataset, candidate["fingerprint"]) in done]
                    if len(stage_rows) != len(candidates):
                        block["stage_results"][stage] = {"status": "partial_budget" if budget_exhausted else "partial"}
                        save_jobs()
                        del data
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        break
                    previous, reference = incumbent, int(incumbent["correct"])
                    best = core.select_best(stage_rows, anchors[dataset]) if stage_rows else previous
                    incumbent = best if int(best["correct"]) > int(previous["correct"]) else previous
                    incumbents[dataset] = incumbent
                    info = {"status": "complete", "reference_correct": reference, "winner_correct": int(incumbent["correct"]),
                            "gain_correct": int(incumbent["correct"]) - reference, "winner_fingerprint": incumbent["fingerprint"]}
                    block["stage_results"][stage] = info
                    stage_info[dataset][stage] = info
                    save_jobs()
                    del data
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                if budget_exhausted:
                    break

            final_datasets = [dataset for dataset in datasets if all(stage_info[dataset].get(stage, {}).get("status") == "complete" for stage in stages)]
            for dataset in final_datasets:
                core.check_stop(stop, f"before verification {dataset}")
                if time.time() >= total_deadline_wall:
                    raise RuntimeError("total deadline exhausted before protected verification completed")
                data, meta = load_data(dataset)
                incumbent = incumbents[dataset]
                dataset_dir = out / dataset
                dataset_dir.mkdir(parents=True, exist_ok=True)
                core.setup_seed(args.global_seed)
                verify = core.replay_v3(data, incumbent["config"], stop, args.stop_check_interval,
                                        dataset_dir / "winner_predictions_replay.npz")
                core.assert_reproduced(incumbent, verify, f"{dataset} winner")
                winner = {**{key: incumbent[key] for key in ("candidate_id", "fingerprint", "stage", "label", "config")},
                          **core.compact_result(incumbent), "verification": core.compact_result(verify),
                          "cache_sha256": meta["sha256"], "sample_order_seed": meta["sample_order_seed"],
                          "sample_order_sha256": meta["full_permutation_sha256"], "order_sha256": meta["order_sha256"],
                          "selection_scope": "fixed_order_tuning_oracle" if dataset in {"A", "V"} else "canonical_order_tuning_oracle"}
                winners[dataset] = winner

                original_identity = core.stable_sha({"run": identity_sha, "dataset": dataset, "kind": "original", "base": original_bases[dataset]})
                fixed = original_results.get(dataset)
                if fixed is None:
                    core.setup_seed(args.global_seed)
                    metric = core.replay_dota(data, original_bases[dataset], stop, args.stop_check_interval,
                                              dataset_dir / "original_dota_predictions.npz")
                    fixed = {"identity_sha256": original_identity, "base": original_bases[dataset], **core.compact_result(metric)}
                    original_results[dataset] = fixed
                elif fixed.get("identity_sha256") != original_identity:
                    raise RuntimeError(f"original baseline identity mismatch: {dataset}")
                matched_identity = core.stable_sha({"run": identity_sha, "dataset": dataset, "kind": "matched", "base": incumbent["config"]["base"]})
                paired = matched_results.get(dataset)
                if paired is None:
                    core.setup_seed(args.global_seed)
                    metric = core.replay_dota(data, incumbent["config"]["base"], stop, args.stop_check_interval,
                                              dataset_dir / "matched_base_dota_predictions.npz")
                    paired = {"identity_sha256": matched_identity, "base": incumbent["config"]["base"], **core.compact_result(metric)}
                    matched_results[dataset] = paired
                elif paired.get("identity_sha256") != matched_identity:
                    raise RuntimeError(f"matched baseline identity mismatch: {dataset}")
                core.atomic_json(baselines_path, {"status": "running", "original_dota_fixed": original_results, "matched_base_dota": matched_results})
                write_summary(out, datasets, original_results, matched_results, winners, "running")
                del data
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            formal_complete = args.max_samples is None and len(winners) == len(datasets)
            status = "complete_budget_limited" if formal_complete and budget_exhausted else "complete" if formal_complete else "smoke_complete" if args.max_samples is not None and len(winners) == len(datasets) else "partial"
            core.atomic_json(baselines_path, {"status": status, "original_dota_fixed": original_results, "matched_base_dota": matched_results})
            write_summary(out, datasets, original_results, matched_results, winners, status)
            core.atomic_json(state_path, {"status": status, "datasets": len(winners), "budget_exhausted": budget_exhausted,
                                          "finished_at": time.time(), "total_deadline_wall": total_deadline_wall})
            final_manifest = {"status": status, **identity_record, "started_at": started_at,
                              "search_deadline_wall": search_deadline_wall, "total_deadline_wall": total_deadline_wall,
                              "jobs_sha256": core.sha256_file(jobs_path), "summary_sha256": core.sha256_file(out / "summary.json")}
            core.atomic_json(manifest_path, final_manifest)
            if formal_complete and set(datasets) == set(DATASETS):
                payload = {"version": VERSION, "status": status, "parent_run": str(parent),
                           "protocol": "A seed152 / V seed20 fixed-order continuation oracle; R canonical-order continuation oracle",
                           "datasets": winners, "summary_sha256": final_manifest["summary_sha256"]}
                core.atomic_json(V3_ROOT / "BEST_SINGLE_CONFIG_IMAGENET_AVR_ROUND2.json", payload)
                lines = [f"{dataset}: {winners[dataset]['correct']}/{winners[dataset]['num_samples']} = {winners[dataset]['accuracy']:.6f}% | seed={REGISTRY[dataset]['sample_order_seed']}" for dataset in datasets]
                (V3_ROOT / "BEST_SINGLE_CONFIG_IMAGENET_AVR_ROUND2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            return 0
        except core.StopRequested as exc:
            core.atomic_json(state_path, {"status": "interrupted", "reason": str(exc), "updated_at": time.time()})
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
