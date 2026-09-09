#!/usr/bin/env python3
"""Two-stage true-online compatibility-source search; isolated from OCR-DOTA-V3."""

from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V3 = REPO / "OCR-DOTA-V3"
sys.path[:0] = [str(REPO), str(V3), str(HERE)]

import tune_nonimagenet21 as core  # noqa: E402
from compat_risk_model import OCRDOTAV3CompatibilitySearch, SOURCE_COMPONENTS  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402

VERSION = "v3-compat-risk-two-stage-v1"
SOURCES = ("D", "E", "O", "EO", "DO")
STAGE1 = ("dtd", "eurosat", "ucf101")
CLASSIC10 = core.TEN_DATASETS
WINNERS = V3 / "BEST_SINGLE_CONFIG_NONIMAGENET21_AFTER_CONTINUATION.json"
DEFAULT_OUT = HERE / "results" / "classic10_compat_risk_20260903"


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_sha(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def append_fsync(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def candidates() -> list[dict[str, Any]]:
    rows = [{"candidate_id": source, "source": source, "prediction_strength": 0.1, "update_power": 1.0, "tau_rank": 0.15} for source in SOURCES]
    assert len(rows) == 5
    for row in rows:
        row["candidate_sha256"] = stable_sha(row)
    return rows


def compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("correct", "num_samples", "accuracy", "prediction_sha256", "trajectory_sha256", "state_sha256",
            "compatibility_sha256", "health_status", "state_health", "elapsed_sec", "peak_cuda_bytes",
            "compatibility_floor_rate", "mean_rank_compatibility", "mean_update_gate", "mean_update_mass")
    return {key: metrics[key] for key in keys if key in metrics}


def replay(data: Mapping[str, Any], config: Mapping[str, Any], arm: Mapping[str, Any], stop: Path, interval: int) -> dict[str, Any]:
    core.validate_config(config)
    device = str(data["features"].device)
    dim, classes = map(int, data["clip_shape"])
    model = OCRDOTAV3CompatibilitySearch(
        config["base"], {"rank": config["rank"], "update": config["update"]},
        dim, classes, data["text_prototypes"], device=device,
        source=arm["source"], prediction_strength=arm["prediction_strength"], update_power=arm["update_power"],
    )
    model.eval()
    count = int(data["features"].shape[0])
    dtype = legacy.compact_dtype(classes)
    targets = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    sample_ids = np.asarray(data["sample_ids"], dtype=np.int64)
    predictions = np.empty(count, dtype=dtype)
    # Ranking metadata is added to promoted arms; hash only executable fields
    # so Stage1 and cold replay have an identical trajectory identity.
    executable_arm = {key: arm[key] for key in ("candidate_id", "source", "prediction_strength", "update_power", "tau_rank")}
    trajectory = hashlib.sha256(canonical({"config": config, "arm": executable_arm}).encode())
    compat_digest = hashlib.sha256()
    sums = {"compat": 0.0, "gate": 0.0, "mass": 0.0}
    floor_hits = floor_total = 0
    started = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for index in range(count):
            if index % max(1, interval) == 0 and stop.exists():
                raise core.StopRequested(f"STOP at {index}/{count}")
            views = data["features"][index].to(device=device, dtype=torch.float32)
            clip_logits = data["clip_logits"][index:index + 1].to(device=device, dtype=torch.float32)
            prob_map = data["prob_maps"][index].to(device=device, dtype=torch.float32)
            z = views.mean(0, keepdim=True)
            parts = model.posterior_and_responsibility(z, return_parts=True)
            pre_cap = float(config["base"]["rho"]) * model.C.mean() / views.size(0)
            weight = torch.clamp(pre_cap, max=float(config["base"]["eta"]))
            final = clip_logits + weight * parts["rank_logits"]
            if not bool(torch.isfinite(final).all()):
                raise FloatingPointError(f"non-finite prediction at {index}")
            predictions[index] = int(final.argmax(-1).item())
            features = core.update_features(views, z, config["update_views"])
            gaussian = model.gaussian_logits(features)
            residual = model._residual_magnitude(features)
            p_geometry = F.softmax(gaussian - model.residual_strength * residual, -1)
            allocation = core.allocation_by_rule(config["update_rule"], prob_map, p_geometry)
            gate = parts["update_gate"]
            weights = gate * allocation
            if not bool(torch.isfinite(weights).all()):
                raise FloatingPointError(f"non-finite update at {index}")
            trajectory.update(index.to_bytes(8, "little"))
            trajectory.update(weights.detach().contiguous().cpu().numpy().tobytes())
            model.fit_ocr(features, weights)
            model.update()
            compat = parts["rank_compatibility"]
            compat_digest.update(index.to_bytes(8, "little"))
            compat_digest.update(compat.detach().contiguous().cpu().numpy().tobytes())
            sums["compat"] += float(compat.mean().cpu())
            sums["gate"] += float(gate.mean().cpu())
            sums["mass"] += float(weights.sum(-1).mean().cpu())
            floor_hits += int((compat <= model.rank_cfg.eps * 1.0001).sum().cpu())
            floor_total += compat.numel()
    health = core.health(model)
    correct = int((predictions == targets).sum())
    return {
        "correct": correct, "num_samples": count, "accuracy": 100.0 * correct / count,
        "prediction_sha256": core.prediction_sha(sample_ids, targets, predictions),
        "trajectory_sha256": trajectory.hexdigest(), "state_sha256": core.state_sha(model),
        "compatibility_sha256": compat_digest.hexdigest(), "health_status": health["health_status"],
        "state_health": health, "compatibility_floor_rate": floor_hits / max(1, floor_total),
        "mean_rank_compatibility": sums["compat"] / count, "mean_update_gate": sums["gate"] / count,
        "mean_update_mass": sums["mass"] / count, "elapsed_sec": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0,
    }


def rank_stage1(rows: list[dict[str, Any]], stage1_datasets=STAGE1) -> list[dict[str, Any]]:
    table = {(row["dataset"], row["candidate_id"]): row for row in rows if row.get("status") == "ok" and row.get("stage") == "stage1"}
    result = []
    for arm in candidates():
        selected = [table.get((dataset, arm["candidate_id"])) for dataset in stage1_datasets]
        if any(row is None for row in selected):
            continue
        result.append({**arm, "macro_accuracy": sum(row["accuracy"] for row in selected) / len(selected),
                       "per_dataset": {row["dataset"]: row["accuracy"] for row in selected}})
    return sorted(result, key=lambda x: (-x["macro_accuracy"], len(SOURCE_COMPONENTS[x["source"]]), x["candidate_id"]))


def write_reports(out: Path, rows: list[dict[str, Any]], ranking: list[dict[str, Any]], top3: list[dict[str, Any]], verified: int, stage2_datasets=CLASSIC10) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    table = {(r["dataset"], r["candidate_id"]): r for r in ok}
    final_rank = []
    for arm in top3:
        ds = [table[(d, arm["candidate_id"])] for d in stage2_datasets if (d, arm["candidate_id"]) in table]
        final_rank.append({**arm, "datasets_complete": len(ds), "macro_accuracy": sum(r["accuracy"] for r in ds) / len(ds) if ds else None,
                           "micro_correct": sum(r["correct"] for r in ds), "num_samples": sum(r["num_samples"] for r in ds)})
    final_rank.sort(key=lambda x: (-(x["macro_accuracy"] if x["macro_accuracy"] is not None else -1), x["candidate_id"]))
    expected_verified = 3 * len(stage2_datasets)
    payload = {"status": "complete" if len(final_rank) == 3 and all(x["datasets_complete"] == len(stage2_datasets) for x in final_rank) and verified == expected_verified else "running",
               "stage1_ranking": ranking, "top3": top3, "stage2_ranking": final_rank, "verified": verified}
    core.atomic_json(out / "summary.json", payload)
    core.atomic_json(out / "winner_ranking.json", final_rank)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("candidate", "source", "dataset", "correct", "N", "accuracy"))
        for arm in top3:
            for dataset in stage2_datasets:
                row = table.get((dataset, arm["candidate_id"]))
                if row: writer.writerow((arm["candidate_id"], arm["source"], dataset, row["correct"], row["num_samples"], row["accuracy"]))
    lines = ["V3 compatibility-risk真实在线搜索", f"状态: {payload['status']}", f"复验: {verified}/{expected_verified}", "", "Stage1 top3:"]
    lines += [f"- {x['candidate_id']}: macro={x['macro_accuracy']:.6f}%" for x in top3]
    lines += ["", "Stage2 ranking:"]
    lines += [f"- {x['candidate_id']}: macro={x['macro_accuracy']:.6f}% ({x['datasets_complete']}/{len(stage2_datasets)})" for x in final_rank if x["macro_accuracy"] is not None]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(DEFAULT_OUT)); p.add_argument("--device", default="cuda")
    p.add_argument("--global-seed", type=int, default=1); p.add_argument("--stop-check-interval", type=int, default=25)
    p.add_argument("--max-samples", type=int); p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-dtd32", action="store_true", help="run all five sources and top3 replay on DTD first 32 samples")
    return p.parse_args()


def main() -> int:
    args = parse_args(); out = Path(args.output).resolve(); stop = out / "STOP"
    if args.smoke_dtd32:
        if args.max_samples not in (None, 32): raise ValueError("--smoke-dtd32 requires --max-samples 32 or omitted")
        args.max_samples = 32
        stage1_datasets, stage2_datasets = ("dtd",), ("dtd",)
    else:
        stage1_datasets, stage2_datasets = STAGE1, CLASSIC10
    winners_path = WINNERS.resolve(); winner_rows = json.loads(winners_path.read_text())["datasets"]
    cache_paths = {d: core.cache_path(d, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT) for d in CLASSIC10}
    expected = json.loads(core.EXPECTED_CACHE_MANIFEST.read_text())
    identity = {"version": VERSION, "global_seed": args.global_seed, "max_samples": args.max_samples, "precision": "fp32",
                "protocol": {"sources": SOURCES, "stage1": stage1_datasets, "stage2": stage2_datasets, "tau": .15, "prediction_strength": .1, "update_power": 1.0, "top_k": 3,
                    "formula": "D=1-cos^2; E=clip(old_semantic,0,1); O=.5*relu(max_other-cos); EO=(E+O)/2; DO=(D+O)/2; compatibility=exp(-risk/.15)", "candidates": candidates()},
                "winner_source": str(winners_path), "winner_sha256": core.sha256_file(winners_path),
                "code_sha256": {"runner": core.sha256_file(Path(__file__)), "sidecar_model": core.sha256_file(HERE / "compat_risk_model.py"),
                    "v3_model": core.sha256_file(V3 / "ocr_dota_v3/model.py"), "rank": core.sha256_file(V3 / "ocr_dota_v3/rank_compatibility.py"),
                    "tuner": core.sha256_file(V3 / "tune_nonimagenet21.py"), "legacy": core.sha256_file(REPO / "scripts/run_cross_benchmark_ocr_ablation.py")},
                "caches": {d: {"path": str(cache_paths[d]), "sha256": core.sha256_file(cache_paths[d]), "num_samples": expected[d]["num_samples"], "order_sha256": expected[d]["order_sha256"]} for d in CLASSIC10}}
    identity_sha = stable_sha(identity); out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if not args.resume or old.get("identity_sha256") != identity_sha: raise RuntimeError("resume identity mismatch or --resume missing")
    else: core.atomic_json(manifest, {"status": "running", "identity": identity, "identity_sha256": identity_sha, "started_at": time.time()})
    results_path, verify_path = out / "results.jsonl", out / "verification.jsonl"
    rows = core.load_jsonl(results_path); vr = core.load_jsonl(verify_path)
    done = {(r.get("dataset"), r.get("candidate_id")): r for r in rows if r.get("status") == "ok" and r.get("identity_sha256") == identity_sha}
    verified = {(r.get("dataset"), r.get("candidate_id")): r for r in vr if r.get("status") == "reproduced" and r.get("identity_sha256") == identity_sha}

    def load(dataset):
        core.setup_seed(args.global_seed); data, meta = legacy.load_cache(cache_paths[dataset], args.device, args.max_samples)
        exp = identity["caches"][dataset]
        if meta["sha256"] != exp["sha256"]: raise RuntimeError(f"cache SHA mismatch {dataset}")
        if args.max_samples is None and (meta["num_samples"] != exp["num_samples"] or meta["order_sha256"] != exp["order_sha256"]): raise RuntimeError(f"cache identity mismatch {dataset}")
        return data, meta

    def run_one(dataset, arm, stage, data, meta):
        key = (dataset, arm["candidate_id"])
        if key in done: return done[key]
        core.atomic_json(out / "state.json", {"status": "running", "stage": stage, "dataset": dataset, "candidate": arm["candidate_id"], "updated_at": time.time()})
        config = json.loads(canonical(winner_rows[dataset]["config"]))
        config["rank"]["tau_rank"] = arm["tau_rank"]; config["rank"]["prediction_strength"] = arm["prediction_strength"]; config["rank"]["update_power"] = arm["update_power"]
        core.setup_seed(args.global_seed); metrics = replay(data, config, arm, stop, args.stop_check_interval)
        row = {"status": "ok", "identity_sha256": identity_sha, "stage": stage, "dataset": dataset, **arm,
               "config": config, "config_sha256": stable_sha(config), "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"], **compact(metrics)}
        append_fsync(results_path, row); rows.append(row); done[key] = row; return row

    try:
        with core.PidLock(out / "RUNNING.pid"):
            for dataset in stage1_datasets:
                if stop.exists(): raise core.StopRequested("STOP before stage1")
                data, meta = load(dataset)
                for arm in candidates(): run_one(dataset, arm, "stage1", data, meta)
                del data; torch.cuda.empty_cache()
            ranking = rank_stage1(rows, stage1_datasets); top3 = ranking[:3]
            if len(top3) != 3: raise RuntimeError("stage1 incomplete")
            core.atomic_json(out / "stage1_ranking.json", ranking); core.atomic_json(out / "top3.json", top3)
            for dataset in stage2_datasets:
                data, meta = load(dataset)
                for arm in top3: run_one(dataset, arm, "stage2", data, meta)
                for arm in top3:
                    key = (dataset, arm["candidate_id"])
                    if key not in verified:
                        config = done[key]["config"]; core.setup_seed(args.global_seed)
                        again = replay(data, config, arm, stop, args.stop_check_interval)
                        fields = ("correct", "num_samples", "prediction_sha256", "trajectory_sha256", "state_sha256", "compatibility_sha256")
                        mismatch = {f: [done[key].get(f), again.get(f)] for f in fields if done[key].get(f) != again.get(f)}
                        if mismatch: raise RuntimeError(f"cold replay mismatch {dataset}/{arm['candidate_id']}: {mismatch}")
                        rec = {"status": "reproduced", "identity_sha256": identity_sha, "dataset": dataset, "candidate_id": arm["candidate_id"], "fields": fields, "elapsed_sec": again["elapsed_sec"]}
                        append_fsync(verify_path, rec); verified[key] = rec
                write_reports(out, rows, ranking, top3, len(verified), stage2_datasets)
                del data; torch.cuda.empty_cache()
            write_reports(out, rows, ranking, top3, len(verified), stage2_datasets)
            status = "complete_smoke" if args.max_samples is not None else "complete"
            core.atomic_json(out / "state.json", {"status": status, "verified": len(verified), "finished_at": time.time()})
            doc = json.loads(manifest.read_text()); doc.update({"status": status, "finished_at": time.time(), "summary_sha256": core.sha256_file(out / "summary.json")}); core.atomic_json(manifest, doc)
    except core.StopRequested as exc:
        core.atomic_json(out / "state.json", {"status": "stopped", "reason": str(exc), "updated_at": time.time()}); return 75
    return 0


if __name__ == "__main__": raise SystemExit(main())
