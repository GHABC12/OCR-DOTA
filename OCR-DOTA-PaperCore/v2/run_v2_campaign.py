#!/usr/bin/env python3
"""Small, resumable driver for the PaperCore V2 ten-dataset protocol.

The driver intentionally keeps V2 state in a new output tree and never calls
or edits the V1 campaign. It can be run phase by phase for fast diagnostics or
with ``--phase all`` for the complete evidence -> offline P -> responsibility
screen -> paired final pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
REPO = V2_ROOT.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(V2_ROOT)); sys.path.insert(0, str(REPO / "OCR-DOTA-V3")); sys.path.insert(0, str(REPO))

import tune_nonimagenet21 as core  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402
from build_posterior_evidence import build_evidence  # noqa: E402
from evaluate_posterior_offline import evaluate_directory, evaluate_candidate, load_evidence  # noqa: E402
from replay_pairs import replay_pair  # noqa: E402
from screen_responsibility import screen  # noqa: E402
from run_paper_ablation import replay_paper  # noqa: E402
from export_v2_results import export as export_results  # noqa: E402
from protocol_v2 import (  # noqa: E402
    DEV_DATASETS, EVAL_DATASETS, PROTOCOL_VERSION, TEN_DATASETS,
    digest, protocol_dict,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""): h.update(block)
    return h.hexdigest()


def _write_state(out: Path, phase: str, **extra: Any) -> None:
    _write_json(out / "state.json", {"status": "running", "phase": phase, "time": time.time(), **extra})


def _evidence_ready(out: Path, dataset: str) -> bool:
    manifest = out / "evidence" / f"{dataset}.json"; artifact = out / "evidence" / f"{dataset}.npz"
    if not manifest.exists() or not artifact.exists(): return False
    try:
        item = json.loads(manifest.read_text(encoding="utf-8"))
        return item.get("evidence_sha256") == _sha(artifact) and item.get("protocol_version") == PROTOCOL_VERSION
    except (OSError, ValueError, KeyError): return False


def phase_exact(out: Path, device: str) -> dict[str, Any]:
    """Check V2 Base against the independent legacy replay at 32/500/full."""
    path = out / "exact_validation.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("status") == "passed": return existing
        except (OSError, ValueError):
            pass
    stop = out / "STOP"
    if stop.exists(): raise RuntimeError(f"STOP marker exists: {stop}")
    dataset = "dtd"; cache = core.cache_path(dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT)
    base = core.load_original_base(REPO, dataset)
    config = {"tau_rank": .15, "alpha": 0., "lambda_max": 0., "prediction_power": 1., "posterior_mode": "gated_log_prior", "delta": 1., "beta_resp": 0., "responsibility_mode": "dota"}
    records = []
    for limit in (32, 500, None):
        data, meta = legacy.load_cache(cache, device, limit); core.setup_seed(1)
        reference = core.replay_dota(data, base, stop, 25)
        core.setup_seed(1)
        paper = replay_paper(data, base, config, "base", stop, save_path=None)
        keys = ("correct", "num_samples", "prediction_sha256", "trajectory_sha256", "state_sha256")
        mismatches = [key for key in keys if reference.get(key) != paper.get(key)]
        records.append({"num_samples": int(data["features"].shape[0]), "limit": limit, "cache_sha256": meta["sha256"], "order_sha256": meta["order_sha256"], "status": "passed" if not mismatches else "failed", "mismatches": mismatches, "reference": {k: reference.get(k) for k in keys}, "paper": {k: paper.get(k) for k in keys}})
        if mismatches: raise RuntimeError(f"V2 exact Base mismatch at limit={limit}: {mismatches}")
    result = {"status": "passed", "protocol_version": PROTOCOL_VERSION, "dataset": dataset, "records": records, "precision": "fp32", "label_policy": "targets read only in metrics after update"}
    _write_json(path, result); return result


def phase_evidence(out: Path, device: str, max_samples: int | None = None) -> dict[str, Any]:
    done = []
    for index, dataset in enumerate(TEN_DATASETS, 1):
        _write_state(out, "evidence", dataset=dataset, completed=done, total=len(TEN_DATASETS), index=index)
        if not _evidence_ready(out, dataset):
            build_evidence(dataset, out, device=device, max_samples=max_samples)
        done.append(dataset)
    result = {"status": "complete", "datasets": done, "num_samples_override": max_samples}
    _write_json(out / "evidence_phase.json", result)
    return result


def _select_responsibility(screen_result: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in screen_result.get("candidates", []) if r.get("not_obviously_failed")]
    if not rows: return {"mode": "gate_only", "delta": 0.0, "beta": 0.0, "status": "negative_fallback_dota"}
    rows.sort(key=lambda r: (-float(r["aggregate"].get("macro_accuracy", -1e30)), -int(r["aggregate"].get("correct", -1)), float(r["aggregate"].get("wrm", 1e30)), r["candidate_id"]))
    row = rows[0]
    return {"mode": row["mode"], "delta": row.get("delta", 0.0), "beta": row.get("beta", 0.0), "status": "selected", "candidate_id": row["candidate_id"]}


def phase_development(out: Path, device: str, prefix: int = 1000) -> dict[str, Any]:
    phase_exact(out, device)
    evidence_result = phase_evidence(out, device)
    _write_state(out, "posterior_offline")
    posterior_result = evaluate_directory(out / "evidence", out)
    # Required DTD equivalence check: offline V2 posterior must agree with an
    # online V2 replay for both preregistered posterior families before the
    # offline grid is treated as a valid search result.
    offline_online = []
    dtd_cache = core.cache_path("dtd", core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT)
    dtd_data, _ = legacy.load_cache(dtd_cache, device, None)
    dtd_base = core.load_original_base(REPO, "dtd")
    dtd_evidence = load_evidence(out / "evidence" / "dtd.npz")
    for mode in ("direct_prior", "entropy_gated_prior"):
        cfg = {"posterior_mode": mode, "gamma": .1, "entropy_power": 1., "tau_rank": .15}
        offline_metric, _, _ = evaluate_candidate(dtd_evidence, {"mode": mode, "gamma": .1, "entropy_power": 1.})
        core.setup_seed(1)
        online_pair = replay_pair(dtd_data, dtd_base, cfg, {"mode": "dota", "delta": 0., "beta": 0.}, "base_posterior", out / "STOP", out / "offline_online" / mode)
        online = online_pair["posterior"]
        checks = {
            "correct": int(offline_metric["correct"]) == int(online["correct"]),
            "prediction_sha256": offline_metric["prediction_sha256"] == online["prediction_sha256"],
            "wrong_to_correct": int(offline_metric["wrong_to_correct"]) == int(online["wrong_to_correct"]),
            "correct_to_wrong": int(offline_metric["correct_to_wrong"]) == int(online["correct_to_wrong"]),
            "nll": abs(float(offline_metric["nll"]) - float(online["nll"])) <= 1e-6,
            "brier": abs(float(offline_metric["brier"]) - float(online["brier"])) <= 1e-6,
            "ece": abs(float(offline_metric["ece"]) - float(online["ece"])) <= 1e-6,
        }
        offline_online.append({"mode": mode, "gamma": .1, "checks": checks, "status": "passed" if all(checks.values()) else "failed", "offline": {k: offline_metric[k] for k in ("correct", "prediction_sha256", "wrong_to_correct", "correct_to_wrong", "nll", "brier", "ece")}, "online": {k: online[k] for k in ("correct", "prediction_sha256", "wrong_to_correct", "correct_to_wrong", "nll", "brier", "ece")}})
        if not all(checks.values()): raise RuntimeError(f"offline/online posterior mismatch for {mode}: {checks}")
    _write_json(out / "offline_online_verification.json", {"status": "passed", "dataset": "dtd", "checks": offline_online})
    _write_state(out, "responsibility_quick_screen")
    responsibility_result = screen(tuple(DEV_DATASETS), out, prefix, device)
    selected_p = posterior_result.get("selected")
    posterior = (selected_p or {}).get("config", {"mode": "direct_prior", "gamma": 0.0, "entropy_power": 1.0})
    # At most two candidates survive the prefix screen and are then run on
    # the complete Dev-5 streams. This is the only responsibility result used
    # for freezing V2.
    quick_rows = [r for r in responsibility_result.get("candidates", []) if r.get("not_obviously_failed")]
    quick_rows.sort(key=lambda r: (-float(r["aggregate"].get("macro_accuracy", -1e30)), -int(r["aggregate"].get("correct", -1)), float(r["aggregate"].get("wrm", 1e30)), r["candidate_id"]))
    finalist_candidates = tuple({"mode": r["mode"], "delta": r.get("delta", 0.), "beta": r.get("beta", 0.)} for r in quick_rows[:2])
    if finalist_candidates:
        responsibility_full = screen(tuple(DEV_DATASETS), out, None, device, finalist_candidates)
    else:
        responsibility_full = {"status": "negative", "candidates": []}
    selected_u = _select_responsibility(responsibility_full if responsibility_full.get("candidates") else responsibility_result)
    frozen = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": protocol_dict(),
        "posterior": posterior,
        "responsibility": selected_u,
        "dev_datasets": list(DEV_DATASETS), "eval_datasets": list(EVAL_DATASETS),
        "selection_status": {"posterior": posterior_result.get("status"), "responsibility": selected_u.get("status")},
        "code_sha256": {
            str(p.relative_to(REPO)): _sha(p)
            for p in sorted(list((V2_ROOT / "v2").glob("*.py")) + [V2_ROOT / "ocr_dota_paper" / "posterior_v2.py", V2_ROOT / "ocr_dota_paper" / "responsibility_v2.py"])
        },
        "frozen_at": time.time(),
    }
    frozen["config_sha256"] = digest({"posterior": posterior, "responsibility": selected_u})
    (out / "FROZEN_V2_CONFIG.yaml").write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    _write_json(out / "V2_PROTOCOL.json", protocol_dict())
    _write_json(out / "development_summary.json", {"evidence": evidence_result, "posterior": posterior_result, "offline_online": offline_online, "responsibility_quick": responsibility_result, "responsibility_full": responsibility_full, "frozen": frozen})
    _write_json(out / "state.json", {"status": "development_complete", "phase": "development", "frozen_config_sha256": frozen["config_sha256"], "time": time.time()})
    return frozen


def phase_final(out: Path, frozen: dict[str, Any], device: str, datasets: tuple[str, ...]) -> dict[str, Any]:
    final = out / "final"; final.mkdir(parents=True, exist_ok=True)
    posterior = frozen["posterior"]
    responsibility = frozen["responsibility"]
    all_results = {}
    for index, dataset in enumerate(datasets, 1):
        _write_state(out, "final_pairs", dataset=dataset, index=index, total=len(datasets))
        cache = core.cache_path(dataset, core.TEN_CACHE_ROOT, core.DOMAIN_CACHE_ROOT)
        data, _ = legacy.load_cache(cache, device, None)
        base = core.load_original_base(REPO, dataset); core.setup_seed(1)
        target_dir = final / "pairs" / dataset
        pair_a = replay_pair(data, base, {"posterior_mode": posterior.get("mode", "direct_prior"), "gamma": posterior.get("gamma", 0.0), "entropy_power": posterior.get("entropy_power", 1.0), "tau_rank": .15}, responsibility, "base_posterior", out / "STOP", target_dir / "base_posterior")
        core.setup_seed(1)
        pair_b = replay_pair(data, base, {"posterior_mode": posterior.get("mode", "direct_prior"), "gamma": posterior.get("gamma", 0.0), "entropy_power": posterior.get("entropy_power", 1.0), "tau_rank": .15}, responsibility, "responsibility_full", out / "STOP", target_dir / "responsibility_full")
        all_results[dataset] = {"base_posterior": pair_a, "responsibility_full": pair_b}
        _write_json(target_dir / "pairs.json", all_results[dataset])
    _write_json(final / "pair_results.json", all_results)
    # Export is part of the final phase so a single ``--phase all`` invocation
    # leaves the complete preregistered result tree behind.
    export_results(final / "pairs", final)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("evidence", "development", "final", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", type=int, default=1000)
    parser.add_argument("--datasets", default=",".join(TEN_DATASETS))
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    if set(datasets) - set(TEN_DATASETS): raise ValueError("only the ten V2 datasets are permitted")
    _write_json(out / "V2_PROTOCOL.json", protocol_dict())
    if args.phase == "evidence": result = phase_evidence(out, args.device)
    elif args.phase == "development": result = phase_development(out, args.device, args.prefix)
    elif args.phase == "final":
        if not (out / "exact_validation.json").exists(): raise RuntimeError("exact_validation.json missing; run development first")
        frozen_file = out / "FROZEN_V2_CONFIG.yaml"
        if not frozen_file.exists(): raise RuntimeError(f"missing frozen config: {frozen_file}")
        result = phase_final(out, yaml.safe_load(frozen_file.read_text()), args.device, datasets)
    else:
        frozen = phase_development(out, args.device, args.prefix)
        result = phase_final(out, frozen, args.device, datasets)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
