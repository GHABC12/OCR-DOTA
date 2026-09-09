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


VERSION = "ocr-dota-v3-nonimagenet21-round2-v1"
INITIAL = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_nonimagenet21_full_oracle_24h_20260830"
SOURCE = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_nonimagenet21_continuation_20260901"
DEFAULT_OUTPUT = REPO_ROOT / "log/all_dataset_perf/ocr_dota_v3_nonimagenet21_round2_tiered_20260904"
TARGETS = tuple(core.ALL_DATASETS)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate(config: Mapping[str, Any], candidate_id: str, **changes: Any) -> dict[str, Any]:
    row = core.changed(config, candidate_id, **changes)
    row["candidate_id"] = candidate_id
    return row


def candidate_specs(dataset: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    c = copy.deepcopy(dict(config))
    b, r, u = c["base"], c["rank"], c["update"]
    # Evidence-weighted budget: six directed candidates only for the five
    # streams that responded in the previous continuation, four elsewhere.
    if dataset == "eurosat":
        specs = [candidate(c,"E_eta_m5",base__eta=b["eta"]*.95),candidate(c,"E_eta_p5",base__eta=b["eta"]*1.05),
                 candidate(c,"E_eps_m5",base__epsilon=b["epsilon"]*.95),candidate(c,"E_eps_p5",base__epsilon=b["epsilon"]*1.05),
                 candidate(c,"E_count_m10",update__init_count=u["init_count"]*.90),candidate(c,"E_count_p10",update__init_count=u["init_count"]*1.10)]
    elif dataset == "visda2017_validation":
        specs = [candidate(c,"V_sigma_m5",base__sigma=b["sigma"]*.95),candidate(c,"V_sigma_p5",base__sigma=b["sigma"]*1.05),
                 candidate(c,"V_count_m10",update__init_count=u["init_count"]*.90),candidate(c,"V_count_p10",update__init_count=u["init_count"]*1.10),
                 candidate(c,"V_eta_m5",base__eta=b["eta"]*.95),candidate(c,"V_eta_p5",base__eta=b["eta"]*1.05)]
    elif dataset == "domainnet_quickdraw":
        specs = [candidate(c,"Q_eta_m5",base__eta=b["eta"]*.95),candidate(c,"Q_eta_p5",base__eta=b["eta"]*1.05),
                 candidate(c,"Q_eps_m5",base__epsilon=b["epsilon"]*.95),candidate(c,"Q_eps_p5",base__epsilon=b["epsilon"]*1.05),
                 candidate(c,"Q_res_m20",update__residual_strength=u["residual_strength"]*.80),candidate(c,"Q_res_p20",update__residual_strength=u["residual_strength"]*1.20)]
    elif dataset == "domainnet_real":
        specs = [candidate(c,"R_eta_m5",base__eta=b["eta"]*.95),candidate(c,"R_eta_p5",base__eta=b["eta"]*1.05),
                 candidate(c,"R_count_m10",update__init_count=u["init_count"]*.90),candidate(c,"R_count_p10",update__init_count=u["init_count"]*1.10),
                 candidate(c,"R_eps_m5",base__epsilon=b["epsilon"]*.95),candidate(c,"R_eps_p5",base__epsilon=b["epsilon"]*1.05)]
    elif dataset == "domainnet_sketch":
        specs = [candidate(c,"S_sigma_m10",base__sigma=b["sigma"]*.90),candidate(c,"S_sigma_m5",base__sigma=b["sigma"]*.95),
                 candidate(c,"S_sigma_p5",base__sigma=b["sigma"]*1.05),candidate(c,"S_res_p25",update__residual_strength=u["residual_strength"]*1.25),
                 candidate(c,"S_res_p50",update__residual_strength=u["residual_strength"]*1.50),candidate(c,"S_pred_p10",rank__prediction_strength=r["prediction_strength"]*1.10)]
    elif dataset == "domainnet_clipart":
        specs = [candidate(c,"C_sigma_m20",base__sigma=b["sigma"]*.80),candidate(c,"C_sigma_m10",base__sigma=b["sigma"]*.90),
                 candidate(c,"C_res_p25",update__residual_strength=u["residual_strength"]*1.25),candidate(c,"C_res_p50",update__residual_strength=u["residual_strength"]*1.50)]
    elif dataset == "food101":
        specs = [candidate(c,"F_sigma_m5",base__sigma=b["sigma"]*.95),candidate(c,"F_sigma_p5",base__sigma=b["sigma"]*1.05),
                 candidate(c,"F_eps_m5",base__epsilon=b["epsilon"]*.95),candidate(c,"F_eps_p5",base__epsilon=b["epsilon"]*1.05)]
    else:
        specs = [candidate(c,"B_eta_m5",base__eta=b["eta"]*.95),candidate(c,"B_eta_p5",base__eta=b["eta"]*1.05),
                 candidate(c,"B_rho_m5",base__rho=b["rho"]*.95),candidate(c,"B_rho_p5",base__rho=b["rho"]*1.05)]
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
    if manifest.get("status") not in {"complete", "complete_budget_limited"} or state.get("status") not in {"complete", "complete_budget_limited"} or summary.get("status") not in {"complete", "complete_budget_limited"}:
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
    if any(r.get("status") != "ok" or r.get("health_status") != "healthy" for r in rows):
        raise RuntimeError("parent candidate registry contains an unhealthy row")
    initial_rows = core.load_jsonl(INITIAL / "candidate_results.jsonl")
    if len(initial_rows) != 282 or any(r.get("status") != "ok" or r.get("health_status") != "healthy" for r in initial_rows):
        raise RuntimeError("initial historical registry identity mismatch")
    rows = initial_rows + rows
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
    p.add_argument("--time-budget-hours", type=float, default=8.0)
    p.add_argument("--verification-reserve-hours", type=float, default=1.5)
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
    if not 85 <= sum(history_registry["new"].values()) <= 94:
        raise RuntimeError(f"unexpected new candidate count: {history_registry['new']}")
    parent_identity = parent["manifest"]["identity"]
    cache_identity = parent_identity.get("caches", parent_identity.get("cache_identity"))
    if not isinstance(cache_identity, dict) or set(cache_identity) != set(core.ALL_DATASETS):
        raise RuntimeError("parent cache identity is not the canonical 21 streams")
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
                            "dataset": dataset, "stage": "round2_local_refine", **job,
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
            core.atomic_json(V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21_ROUND2.json", payload)
            text = "\n".join(f"{d}: {winners[d]['correct']}/{winners[d]['num_samples']} = {winners[d]['accuracy']:.6f}%"
                for d in core.ALL_DATASETS) + "\n"
            (V3_ROOT / "BEST_SINGLE_CONFIG_NONIMAGENET21_ROUND2.txt").write_text(text, encoding="utf-8")
            return 0
        except core.StopRequested as exc:
            core.atomic_json(state_path, {"status": "interrupted", "reason": str(exc), "updated_at": time.time()})
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
