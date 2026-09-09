#!/usr/bin/env python3
"""Targeted full-stream continuation for the sealed OCR-DOTA V3 campaign."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch


V3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V3_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V3_ROOT))

import tune_nonimagenet21 as core  # noqa: E402


VERSION = "ocr-dota-v3-targeted-continuation-v1"
SOURCE = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_nonimagenet21_full_oracle_24h_20260830"
DEFAULT_OUTPUT = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_nonimagenet21_continuation_20260901"
TARGETS = (
    "eurosat", "domainnet_sketch", "domainnet_real", "visda2017_validation",
    "domainnet_quickdraw", "food101", "sun397", "ucf101",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate(config: Mapping[str, Any], candidate_id: str, **changes: Any) -> dict[str, Any]:
    row = core.changed(config, candidate_id, **changes)
    row["candidate_id"] = candidate_id
    return row


def candidate_specs(dataset: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    c = copy.deepcopy(dict(config))
    if dataset == "eurosat":
        specs = [
            candidate(c, "E1_eps_0.90", base__epsilon=7.10775e-5),
            candidate(c, "E2_eps_0.95", base__epsilon=7.502625e-5),
            candidate(c, "E3_count_0.75", update__init_count=.28125),
            candidate(c, "E4_count_0.875", update__init_count=.328125),
            candidate(c, "E5_power_3", rank__update_power=3.0),
            candidate(c, "E6_eps_eta_joint", base__epsilon=7.10775e-5, base__eta=3.498),
        ]
    elif dataset == "domainnet_sketch":
        specs = [
            candidate(c, "S1_sigma_0.80", base__sigma=.00144),
            candidate(c, "S2_sigma_0.85", base__sigma=.00153),
            candidate(c, "S3_residual_0.75", update__residual_strength=.75),
            candidate(c, "S4_pred_2.5", rank__prediction_strength=2.5),
            candidate(c, "S5_pred_3", rank__prediction_strength=3.0),
            candidate(c, "S6_sigma_residual_joint", base__sigma=.00153, update__residual_strength=.75),
        ]
    elif dataset == "domainnet_real":
        specs = [
            candidate(c, "R1_eta_1.05", base__eta=.259875),
            candidate(c, "R2_eta_1.10", base__eta=.27225),
            candidate(c, "R3_count_1.10", update__init_count=13.28584125),
            candidate(c, "R4_count_1.25", update__init_count=15.097546875),
            candidate(c, "R5_eta_count_joint", base__eta=.27225, update__init_count=13.28584125),
        ]
    elif dataset == "visda2017_validation":
        specs = [
            candidate(c, "V1_sigma_1.05", base__sigma=.010522281),
            candidate(c, "V2_sigma_1.10", base__sigma=.011023342),
            candidate(c, "V3_count_1.10", update__init_count=14.171564),
            candidate(c, "V4_eta_1.10", base__eta=.2475),
            candidate(c, "V5_sigma_count_joint", base__sigma=.011023342, update__init_count=14.171564),
        ]
    elif dataset == "domainnet_quickdraw":
        specs = [
            candidate(c, "Q1_eta_1.05", base__eta=2.84992554),
            candidate(c, "Q2_eps_0.90", base__epsilon=3.9285e-5),
            candidate(c, "Q3_residual_0.75", update__residual_strength=.75),
            candidate(c, "Q4_rule_clip", update_rule="clip"),
            candidate(c, "Q5_rule_mix05", update_rule="mix05"),
            candidate(c, "Q6_eta_eps_joint", base__eta=2.98563628, base__epsilon=3.9285e-5),
        ]
    elif dataset == "food101":
        specs = [
            candidate(c, "F1_sigma_1.05", base__sigma=.00231),
            candidate(c, "F2_sigma_1.10", base__sigma=.00242),
            candidate(c, "F3_sigma_eps_joint", base__sigma=.00242, base__epsilon=2.2e-5),
        ]
    elif dataset == "sun397":
        specs = [
            candidate(c, "N1_eps_1.05", base__epsilon=.0001155),
            candidate(c, "N2_eps_1.10", base__epsilon=.000121),
            candidate(c, "N3_rho_0.90", base__rho=.007875),
            candidate(c, "N4_rho_1.10", base__rho=.009625),
        ]
    elif dataset == "ucf101":
        specs = [
            candidate(c, "U1_power_3", rank__update_power=3.0),
            candidate(c, "U2_power_4", rank__update_power=4.0),
            candidate(c, "U3_tau_pred_power", rank__tau_rank=.30,
                      rank__prediction_strength=2.0, rank__update_power=2.0),
        ]
    else:
        raise ValueError(dataset)
    return core.unique_candidates(dataset, specs)


def validate_parent(source: Path) -> dict[str, Any]:
    paths = {name: source / name for name in (
        "manifest.json", "state.json", "summary.json", "winners.json",
        "baselines.json", "jobs.json", "candidate_results.jsonl",
    )}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    manifest, state = read_json(paths["manifest.json"]), read_json(paths["state.json"])
    summary, winners = read_json(paths["summary.json"]), read_json(paths["winners.json"])
    if manifest.get("status") != "complete" or state.get("status") != "complete" or summary.get("status") != "complete":
        raise RuntimeError("parent campaign is not sealed complete")
    if core.sha256_file(paths["summary.json"]) != manifest.get("summary_sha256"):
        raise RuntimeError("parent summary SHA mismatch")
    table = winners.get("datasets", {})
    if set(table) != set(core.ALL_DATASETS):
        raise RuntimeError("parent winner set is not the canonical 21 streams")
    for dataset, row in table.items():
        if core.fingerprint(dataset, row["config"]) != row["fingerprint"]:
            raise RuntimeError(f"parent winner fingerprint mismatch: {dataset}")
        core.assert_reproduced(row, row["verification"], f"parent {dataset}")
    rows = core.load_jsonl(paths["candidate_results.jsonl"])
    if len(rows) != 282 or any(r.get("status") != "ok" or r.get("health_status") != "healthy" for r in rows):
        raise RuntimeError("parent candidate registry is not 282 healthy rows")
    if core.sha256_file(V3_ROOT / "tune_nonimagenet21.py") != manifest["identity"]["code_sha256"]["runner"]:
        raise RuntimeError("parent core runner source drifted")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "file_sha256": {name: core.sha256_file(path) for name, path in paths.items()},
        "manifest": manifest, "summary": summary, "winners": table,
        "baselines": read_json(paths["baselines.json"]), "history": rows,
    }


def build_jobs(parent: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    history = {(r["dataset"], r["fingerprint"]): r for r in parent["history"]}
    jobs: dict[str, Any] = {"version": VERSION, "datasets": {}}
    registry: dict[str, Any] = {"accepted_parent_rows": len(history), "reused": {}, "new": {}}
    for dataset in TARGETS:
        anchor = parent["winners"][dataset]
        candidates = candidate_specs(dataset, anchor["config"])
        new, reused = [], []
        for row in candidates:
            key = (dataset, row["fingerprint"])
            if key in history:
                reused.append({"candidate": row, "source_row_sha256": core.stable_sha(history[key])})
            else:
                new.append(row)
        jobs["datasets"][dataset] = {
            "anchor_fingerprint": anchor["fingerprint"], "candidates": new,
            "reused_parent": reused,
        }
        registry["reused"][dataset], registry["new"][dataset] = len(reused), len(new)
    jobs["jobs_sha256"] = core.stable_sha(jobs["datasets"])
    return jobs, registry


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--global-seed", type=int, default=1)
    p.add_argument("--time-budget-hours", type=float, default=6.0)
    p.add_argument("--verification-reserve-hours", type=float, default=1.0)
    p.add_argument("--stop-check-interval", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    repo, source, out = Path(args.repo_root).resolve(), Path(args.source).resolve(), Path(args.output).resolve()
    if repo != REPO_ROOT:
        raise RuntimeError(f"repo identity mismatch: {repo}")
    if args.verification_reserve_hours <= 0 or args.time_budget_hours <= args.verification_reserve_hours:
        raise ValueError("invalid time budget")
    parent = validate_parent(source)
    jobs, history_registry = build_jobs(parent)
    if sum(history_registry["new"].values()) != 38:
        raise RuntimeError(f"expected 38 new candidates, got {history_registry['new']}")
    cache_identity = parent["manifest"]["identity"]["caches"]
    code_sha = {
        "continuation": core.sha256_file(Path(__file__).resolve()),
        "core_runner": core.sha256_file(V3_ROOT / "tune_nonimagenet21.py"),
        "model": core.sha256_file(V3_ROOT / "ocr_dota_v3/model.py"),
        "rank": core.sha256_file(V3_ROOT / "ocr_dota_v3/rank_compatibility.py"),
        "legacy": core.sha256_file(REPO_ROOT / "scripts/run_cross_benchmark_ocr_ablation.py"),
    }
    identity = {
        "version": VERSION, "parent_identity_sha256": parent["manifest"]["identity_sha256"],
        "parent_file_sha256": parent["file_sha256"], "jobs_sha256": jobs["jobs_sha256"],
        "cache_identity": cache_identity, "code_sha256": code_sha,
        "global_seed": args.global_seed, "precision": "fp32", "worker_count": 1,
        "time_budget_hours": args.time_budget_hours,
        "verification_reserve_hours": args.verification_reserve_hours,
    }
    identity_sha = core.stable_sha(identity)
    if args.dry_run:
        print(json.dumps({"identity_sha256": identity_sha, "history": history_registry,
                          "candidate_counts": {d: len(jobs["datasets"][d]["candidates"]) for d in TARGETS}}, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    stop = out / "STOP"
    manifest_path, jobs_path = out / "manifest.json", out / "jobs.json"
    results_path, state_path = out / "candidate_results.jsonl", out / "state.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if not args.resume or old.get("identity_sha256") != identity_sha:
            raise RuntimeError("continuation resume identity mismatch")
        started_at = float(old["started_at"])
        search_deadline, total_deadline = float(old["search_deadline"]), float(old["total_deadline"])
    else:
        started_at = time.time()
        search_deadline = started_at + 3600 * (args.time_budget_hours - args.verification_reserve_hours)
        total_deadline = started_at + 3600 * args.time_budget_hours
        core.atomic_json(manifest_path, {"status": "running", "identity": identity,
            "identity_sha256": identity_sha, "started_at": started_at,
            "search_deadline": search_deadline, "total_deadline": total_deadline})
        core.atomic_json(out / "parent_handoff.json", {
            "source": str(source), "parent_identity_sha256": identity["parent_identity_sha256"],
            "parent_file_sha256": parent["file_sha256"]})
        core.atomic_json(out / "history_registry.json", history_registry)
        core.atomic_json(jobs_path, jobs)
    if read_json(jobs_path).get("jobs_sha256") != jobs["jobs_sha256"]:
        raise RuntimeError("frozen jobs mismatch")

    prior_results = core.load_jsonl(results_path)
    done = {(r["dataset"], r["fingerprint"]): r for r in prior_results
            if r.get("campaign_identity_sha256") == identity_sha and r.get("status") == "ok"
            and r.get("health_status") == "healthy"}
    winners = copy.deepcopy(parent["winners"])
    improved: set[str] = set()
    budget_exhausted = False

    def load_data(dataset: str):
        cache = cache_identity[dataset]
        data, meta = core.legacy.load_cache(Path(cache["path"]), args.device, None)
        if (meta["sha256"], meta["order_sha256"], int(meta["num_samples"])) != (
            cache["sha256"], cache["order_sha256"], int(cache["num_samples"])):
            raise RuntimeError(f"cache/order/N identity mismatch: {dataset}")
        return data, meta

    with core.PidLock(out / "RUNNING.pid"):
        try:
            for dataset_index, dataset in enumerate(TARGETS):
                core.check_stop(stop, f"before {dataset}")
                data, meta = load_data(dataset)
                rows = []
                for candidate_index, job in enumerate(jobs["datasets"][dataset]["candidates"]):
                    key = (dataset, job["fingerprint"])
                    row = done.get(key)
                    if row is None:
                        observed = [float(r["elapsed_sec"]) for r in prior_results if r.get("dataset") == dataset and r.get("elapsed_sec")]
                        estimate = max(observed[-5:]) if observed else core.HISTORICAL_SECONDS[dataset] * 1.25
                        if time.time() + estimate >= search_deadline:
                            budget_exhausted = True
                            break
                        core.atomic_json(state_path, {"status": "running", "dataset": dataset,
                            "dataset_index": dataset_index, "candidate_index": candidate_index,
                            "candidate_count": len(jobs["datasets"][dataset]["candidates"]),
                            "updated_at": time.time()})
                        core.setup_seed(args.global_seed)
                        started = time.time()
                        try:
                            metrics = core.replay_v3(data, job["config"], stop, args.stop_check_interval)
                            status, error = "ok", None
                        except core.StopRequested:
                            raise
                        except Exception as exc:
                            status, error = "error", f"{type(exc).__name__}: {exc}"
                            metrics = {"elapsed_sec": time.time() - started, "health_status": "unhealthy"}
                        row = {"status": status, "error": error, "campaign_identity_sha256": identity_sha,
                            "dataset": dataset, "stage": "targeted_continuation", **job,
                            "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"],
                            "global_seed": args.global_seed, "precision": "fp32", **metrics}
                        core.append_jsonl(results_path, row)
                        prior_results.append(row)
                        if status == "ok":
                            done[key] = row
                    rows.append(row)
                if budget_exhausted:
                    del data
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    break
                healthy = [r for r in rows if r.get("status") == "ok" and r.get("health_status") == "healthy"]
                anchor = parent["winners"][dataset]
                best = core.select_best(healthy, anchor["config"]) if healthy else anchor
                if int(best["correct"]) > int(anchor["correct"]):
                    # A new winner is unusable unless there is enough protected
                    # time for its replay and, when base changed, matched DOTA.
                    reserve_need = core.HISTORICAL_SECONDS[dataset] * 2.5
                    if time.time() + reserve_need >= total_deadline:
                        budget_exhausted = True
                        del data
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        break
                    core.setup_seed(args.global_seed)
                    verification = core.replay_v3(data, best["config"], stop, args.stop_check_interval,
                        out / dataset / "winner_predictions_replay.npz")
                    core.assert_reproduced(best, verification, f"continuation {dataset}")
                    winners[dataset] = {**best, "verification": core.compact_result(verification),
                        "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"],
                        "source_campaign": str(out)}
                    improved.add(dataset)
                del data
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Reuse sealed baselines whenever their semantic base identity is unchanged.
            source_baselines = parent["baselines"]
            original = copy.deepcopy(source_baselines["original_dota_fixed"])
            matched = copy.deepcopy(source_baselines["matched_base_dota"])
            for dataset in improved:
                old_base = parent["winners"][dataset]["config"]["base"]
                new_base = winners[dataset]["config"]["base"]
                if core.canonical(old_base) != core.canonical(new_base):
                    core.check_stop(stop, f"before matched baseline {dataset}")
                    if time.time() + core.HISTORICAL_SECONDS[dataset] * 1.25 >= total_deadline:
                        raise RuntimeError(f"total deadline cannot safely fit matched baseline: {dataset}")
                    data, meta = load_data(dataset)
                    core.setup_seed(args.global_seed)
                    value = core.replay_dota(data, new_base, stop, args.stop_check_interval,
                        out / dataset / "matched_base_dota_predictions.npz")
                    matched[dataset] = {"inherited": False, "base": new_base, **core.compact_result(value)}
                    del data
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            for dataset in core.ALL_DATASETS:
                if dataset not in improved:
                    winners[dataset]["inherited_verified_parent"] = True
                original[dataset]["inherited_verified_parent"] = True
                if dataset not in improved or core.canonical(parent["winners"][dataset]["config"]["base"]) == core.canonical(winners[dataset]["config"]["base"]):
                    matched[dataset]["inherited_verified_parent"] = True

            final_status = "complete_budget_limited" if budget_exhausted else "complete"
            core.atomic_json(out / "baselines.json", {"status": final_status,
                "original_dota_fixed": original, "matched_base_dota": matched})
            core.write_summary(out, core.ALL_DATASETS, original, matched, winners, final_status)
            core.atomic_json(state_path, {"status": final_status, "improved": sorted(improved),
                "budget_exhausted": budget_exhausted, "finished_at": time.time()})
            final_manifest = {"status": final_status, "identity": identity,
                "identity_sha256": identity_sha, "started_at": started_at,
                "search_deadline": search_deadline, "total_deadline": total_deadline,
                "jobs_sha256": core.sha256_file(jobs_path),
                "summary_sha256": core.sha256_file(out / "summary.json")}
            core.atomic_json(manifest_path, final_manifest)
            payload = {"version": VERSION, "status": final_status, "datasets": winners,
                "summary_sha256": final_manifest["summary_sha256"], "parent": str(source)}
            core.atomic_json(V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21_AFTER_CONTINUATION.json", payload)
            core.atomic_json(V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21.json", payload)
            text = "\n".join(f"{d}: {winners[d]['correct']}/{winners[d]['num_samples']} = {winners[d]['accuracy']:.6f}%"
                for d in core.ALL_DATASETS) + "\n"
            (V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21_AFTER_CONTINUATION.txt").write_text(text, encoding="utf-8")
            (V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21.txt").write_text(text, encoding="utf-8")
            return 0
        except core.StopRequested as exc:
            core.atomic_json(state_path, {"status": "interrupted", "reason": str(exc), "updated_at": time.time()})
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
