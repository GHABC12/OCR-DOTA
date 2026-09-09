#!/usr/bin/env python3
"""Wait for sealed AVR Round2, then run up to three adaptive fine rounds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ("A", "R", "V")
FINAL = {"A": (7500, 152, "b265ef0e9aeac95702d27d21a0aef53588c137d6b93200ddbd5b97bceed082ac"),
         "R": (30000, None, "7146f11166efda28ca9ca43d607a7b85de395dda7fb78283479068313b757972"),
         "V": (10000, 20, "5c72e762340104ee441aca8a53022d3206ff8cc838b559e4721e75d666c496d2")}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def atomic(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def validate(run: Path, datasets: tuple[str, ...]) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "complete_budget_limited"}:
        raise RuntimeError(f"unsealed run {run}: {manifest.get('status')}")
    if manifest.get("summary_sha256") != sha(run / "summary.json"):
        raise RuntimeError(f"summary SHA mismatch: {run}")
    payload = json.loads((run / "winners.json").read_text(encoding="utf-8"))
    winners = payload.get("datasets", {})
    for dataset in datasets:
        row = winners[dataset]; n, seed, order = FINAL[dataset]
        if row["num_samples"] != n or row.get("sample_order_seed") != seed or row.get("sample_order_sha256") != order:
            raise RuntimeError(f"identity mismatch {run}/{dataset}")
        if row.get("health_status") != "healthy": raise RuntimeError(f"unhealthy winner {run}/{dataset}")
        verify = row.get("verification", {})
        for key in ("correct", "prediction_sha256", "state_sha256", "trajectory_sha256"):
            if verify.get(key) != row.get(key): raise RuntimeError(f"replay mismatch {run}/{dataset}/{key}")
    return winners

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True); p.add_argument("--parent", required=True)
    p.add_argument("--output", required=True); p.add_argument("--python", required=True)
    p.add_argument("--poll-seconds", type=int, default=30); p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args(); repo, parent, out = map(lambda x: Path(x).resolve(), (args.repo_root, args.parent, args.output))
    out.mkdir(parents=True, exist_ok=True); stop, state = out / "STOP", out / "state.json"
    atomic(state, {"status":"waiting_parent", "parent":str(parent), "at":time.time()})
    while True:
        if stop.exists(): atomic(state, {"status":"interrupted_waiting", "at":time.time()}); return 130
        try: winners = validate(parent, DATASETS); break
        except (FileNotFoundError, json.JSONDecodeError):
            atomic(state, {"status":"waiting_parent", "parent":str(parent), "at":time.time()}); time.sleep(args.poll_seconds)
        except RuntimeError as exc:
            if (parent / "RUNNING.pid").exists():
                atomic(state, {"status":"waiting_parent", "parent":str(parent), "detail":str(exc), "at":time.time()}); time.sleep(args.poll_seconds)
            else: atomic(state, {"status":"failed_parent_validation", "error":str(exc)}); return 2
    latest = {d: parent for d in DATASETS}; active = list(DATASETS)
    atomic(state, {"status":"parent_validated", "parent":str(parent), "active":active, "at":time.time()})
    for round_index in range(3, 3 + args.rounds):
        next_active = []
        for dataset in active:
            if stop.exists(): atomic(state, {"status":"interrupted", "round":round_index, "dataset":dataset}); return 130
            prior = latest[dataset]; prior_winner = validate(prior, (dataset,))[dataset]
            target = out / f"round{round_index}" / dataset
            cmd = [args.python, str(repo / "OCR-DOTA-V3/tune_imagenet_avr_chain_round.py"),
                   "--repo-root", str(repo), "--parent", str(prior), "--output", str(target),
                   "--datasets", dataset, "--round-index", str(round_index), "--time-budget-hours", "4",
                   "--verification-reserve-hours", "0.75", "--stop", str(stop)]
            if (target / "manifest.json").exists(): cmd.append("--resume")
            atomic(state, {"status":"running", "round":round_index, "dataset":dataset, "command":cmd, "at":time.time()})
            if subprocess.run(cmd, cwd=repo).returncode != 0:
                atomic(state, {"status":"failed_child", "round":round_index, "dataset":dataset}); return 3
            winner = validate(target, (dataset,))[dataset]; latest[dataset] = target
            if int(winner["correct"]) > int(prior_winner["correct"]): next_active.append(dataset)
        active = next_active
        atomic(state, {"status":"round_complete", "round":round_index, "active_next":active,
                       "latest":{d:str(p) for d,p in latest.items()}, "at":time.time()})
        if not active: break
    final = {d: validate(latest[d], (d,))[d] for d in DATASETS}
    payload = {"status":"complete", "protocol":"fixed A152/V20/R identity; adaptive three-round chain",
               "latest_runs":{d:str(p) for d,p in latest.items()}, "datasets":final}
    atomic(out / "BEST_SINGLE_CONFIG_IMAGENET_AVR_CHAIN.json", payload)
    atomic(state, {"status":"complete", "latest":payload["latest_runs"], "at":time.time()})
    return 0

if __name__ == "__main__": raise SystemExit(main())
