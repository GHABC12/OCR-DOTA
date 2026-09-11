#!/usr/bin/env python3
"""Dataset-specific oracle tuning for the two OCR-DOTA V3 core modules.

This sidecar intentionally does not change the V3 model, rank compatibility
implementation, or defaults.  It reuses the cache/anchor/replay contracts
from ``run_core2_ablation10.py`` and chooses one posterior, responsibility,
and full configuration per dataset.  Labels are only consumed after a full
online replay.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import run_core2_ablation10 as core

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATASETS = core.DATASETS
DISPLAY = core.DISPLAY
VERSION = "ocr-dota-v3-dataset-specific-oracle10-v1"
P_TAUS = (0.075, 0.15, 0.30, 0.60)
P_RATIOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
U_RULES = ("mix05", "mix08clip")
HISTORY_DEFAULT = REPO / "OCR-DOTA-V3/log/core2_ablation10_20260910_rankfree"


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def cfg_key(cfg: Mapping[str, Any]) -> str:
    return core.canonical(cfg)


def result_row(dataset: str, phase: str, cid: str, cfg: Mapping[str, Any], metric: Mapping[str, Any],
               base: Mapping[str, Any], identity: str, source: str) -> dict[str, Any]:
    row = {"phase": phase, "status": "ok", "source": source, "identity_sha256": identity,
           "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": cid,
           "candidate_fingerprint": core.candidate_fingerprint(dataset, cfg, phase),
           "config": cfg, "num_samples": int(metric["num_samples"]), "correct": int(metric["correct"]),
           "accuracy": float(metric["accuracy"]), "base_correct": int(base["correct"]),
           "base_accuracy": float(base["accuracy"]), "last50_correct": int(metric["last50_correct"]),
           "last50_accuracy": float(metric["last50_accuracy"]),
           "base_last50_correct": int(base["last50_correct"]),
           "base_last50_accuracy": float(base["last50_accuracy"]),
           "delta_correct": int(metric["correct"])-int(base["correct"]),
           "delta_pp": float(metric["accuracy"])-float(base["accuracy"]),
           "delta_last50_correct": int(metric["last50_correct"])-int(base["last50_correct"]),
           "delta_last50_pp": float(metric["last50_accuracy"])-float(base["last50_accuracy"]),
           "health_status": metric.get("health", {}).get("health_status", "healthy")}
    for k in ("prediction_sha256", "trajectory_sha256", "state_sha256", "mean_gate", "median_gate",
              "gate_p10", "gate_p25", "gate_p75", "gate_p90", "mean_effective_update_mass"):
        if k in metric: row[k] = metric[k]
    return row


def load_history(path: Path) -> list[dict[str, Any]]:
    return core.load_jsonl(path / "candidate_results.jsonl") if path.exists() else []


def historical_match(history: list[dict[str, Any]], dataset: str, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    key = cfg_key(cfg)
    hits = [r for r in history if r.get("status") == "ok" and r.get("dataset") == dataset and cfg_key(r.get("config", {})) == key
            and r.get("phase") in {"update_u1", "update_u2"}]
    return hits[-1] if hits else None


def make_identity(repo: Path, anchor_path: Path, anchors: Mapping[str, Any], caches: Mapping[str, Any], seed: int,
                  history_path: Path) -> dict[str, Any]:
    files = {"runner": core.sha256_file(Path(__file__).resolve()),
             "shared_runner": core.sha256_file(ROOT / "run_core2_ablation10.py"),
             "model": core.sha256_file(ROOT / "ocr_dota_v3/model.py"),
             "rank_compatibility": core.sha256_file(ROOT / "ocr_dota_v3/rank_compatibility.py"),
             "anchor": core.sha256_file(anchor_path),
             "history_results": core.sha256_file(history_path / "candidate_results.jsonl") if (history_path / "candidate_results.jsonl").exists() else ""}
    value = {"version": VERSION, "datasets": list(anchors), "global_seed": seed, "precision": "fp32",
             "anchor_source": str(anchor_path), "anchor_sha256": files["anchor"], "anchors_sha256": core.stable_sha(anchors),
             "caches": caches, "history_path": str(history_path), "code_sha256": files}
    return {"identity": value, "identity_sha256": core.stable_sha(value)}


def compatible_resume_identity(old_manifest: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Allow a bug-fix resume when only this sidecar's runner hash changed."""
    old = dict(old_manifest.get("identity", {})); new = dict(current.get("identity", {}))
    oc = dict(old.get("code_sha256", {})); nc = dict(new.get("code_sha256", {}))
    # Model/rank/cache/anchor identity must remain byte-identical.  The
    # runner hash is allowed to change only for a compatible bug fix.
    if oc.get("model") != nc.get("model") or oc.get("rank_compatibility") != nc.get("rank_compatibility"):
        return False
    old["code_sha256"] = {k:v for k,v in oc.items() if k != "runner"}
    new["code_sha256"] = {k:v for k,v in nc.items() if k != "runner"}
    return core.canonical(old) == core.canonical(new)


def posterior_replay(data: Mapping[str, Any], anchor: Mapping[str, Any], taus: tuple[float, ...], ratios: tuple[float, ...],
                     stop: Path, interval: int) -> dict[str, Any]:
    """One state trajectory, all tau/ratio prediction heads."""
    cfg0 = core.make_config(anchor, prediction_strength=0.0, power=0.0, tau=0.15)
    device = str(data["features"].device); dim, classes = map(int, data["clip_shape"])
    model = core.OCRDOTAV3(cfg0["base"], {"rank": cfg0["rank"], "update": cfg0["update"]}, dim, classes,
                           data["text_prototypes"], device=device)
    n = int(data["features"].shape[0]); dtype = core.legacy.compact_dtype(classes)
    target = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    ids = np.asarray(data["sample_ids"], dtype=np.int64)
    specs = [(float(t), float(r), float(t*r)) for t in taus for r in ratios]
    preds = {f"t{t:g}_r{r:g}": np.empty(n, dtype=dtype) for t, r, _ in specs}
    disp_sha = hashlib.sha256(); traj_sha = hashlib.sha256(core.canonical(cfg0).encode())
    compat_sample = np.empty(n, dtype=np.float32); stable = np.empty(n, dtype=np.int32); dynamic = np.empty(n, dtype=np.int32)
    with torch.no_grad():
        for i in range(n):
            if i % max(1, interval) == 0: core.check_stop(stop, f"posterior {i}")
            views = data["features"][i].to(device=device, dtype=torch.float32); z = views.mean(0, keepdim=True)
            clip_logits = data["clip_logits"][i:i+1].to(device=device, dtype=torch.float32)
            prob_map = data["prob_maps"][i].to(device=device, dtype=torch.float32)
            parts = model.posterior_and_responsibility(z, return_parts=True)
            disp = parts["rank_displacement"]
            geometry = parts["geometry_logits"]
            for tau, ratio, strength in specs:
                comp = torch.exp(-disp / tau).clamp_min(float(cfg0["rank"]["eps"]))
                rank_logits = geometry + strength * torch.log(comp)
                cap = float(cfg0["base"]["rho"]) * model.C.mean() / views.size(0)
                weight = torch.clamp(cap, max=float(cfg0["base"]["eta"]))
                preds[f"t{tau:g}_r{ratio:g}"][i] = int((clip_logits + weight * rank_logits).argmax(-1).item())
            compat_sample[i] = float(torch.exp(-disp / 0.15).mean().item())
            stable[i] = int(parts["stable_rank"].argmax(-1).item()); dynamic[i] = int(parts["dynamic_rank"].argmax(-1).item())
            disp_sha.update(disp.detach().cpu().numpy().tobytes())
            allocation = core.tune.allocation_by_rule(cfg0["update_rule"], prob_map, parts["p_geometry"])
            assert torch.allclose(parts["update_gate"], torch.ones_like(parts["update_gate"]), atol=0.0, rtol=0.0)
            model.fit_ocr(views, allocation); model.update(); traj_sha.update(allocation.detach().cpu().numpy().tobytes())
    return {"target": target, "sample_id": ids, "predictions": preds, "compat_sample": compat_sample,
            "stable_top1": stable, "dynamic_top1": dynamic, "state_sha256": core.model_state_sha(model),
            "trajectory_sha256": traj_sha.hexdigest(), "compatibility_sha256": disp_sha.hexdigest(),
            "health": core.tune.health(model)}


def posterior_rows(dataset: str, data: Mapping[str, Any], anchor: Mapping[str, Any], base: Mapping[str, Any], identity: str,
                   out: Path, stop: Path, interval: int, taus: tuple[float, ...] = P_TAUS,
                   ratios: tuple[float, ...] = P_RATIOS, prefix: str = "P") -> list[dict[str, Any]]:
    # P2 local refinement may omit ratio=0.  Always replay the Base head as an
    # internal reference, while emitting only the requested candidates.
    requested_ratios = tuple(ratios)
    replay_ratios = tuple(sorted(set((0.0,) + requested_ratios)))
    grid = posterior_replay(data, anchor, taus, replay_ratios, stop, interval); target = grid["target"]; ids = grid["sample_id"]
    base_pred = grid["predictions"][f"t{taus[0]:g}_r0"]
    bc, ba = core._accuracy(base_pred, target); start = len(target)//2; blc, bla = core._accuracy(base_pred, target, start)
    old_sha = str(base.get("prediction_sha256", ""))
    new_sha = core.pred_sha(ids, target, base_pred)
    if old_sha and old_sha != new_sha: raise RuntimeError(f"baseline reproduction mismatch {dataset}: {new_sha} != {old_sha}")
    if int(base.get("correct", 0)) > 0 and bc != int(base["correct"]): raise RuntimeError(f"baseline count mismatch {dataset}: {bc} != {base['correct']}")
    rows = []
    for tau in taus:
        for ratio in requested_ratios:
            cid = f"{prefix}_t{tau:g}_r{ratio:g}"; key = f"t{tau:g}_r{ratio:g}"; pred = grid["predictions"][key]
            cc, ca = core._accuracy(pred, target); lc, la = core._accuracy(pred, target, start); diag = core._diagnostics(base_pred, pred, target, grid["compat_sample"])
            cfg = core.make_config(anchor, prediction_strength=ratio*tau, power=0.0, tau=tau)
            row = {"phase": "posterior", "status": "ok", "source": "shared_trajectory", "identity_sha256": identity,
                   "dataset": dataset, "display_name": DISPLAY[dataset], "candidate_id": cid,
                   "candidate_fingerprint": core.candidate_fingerprint(dataset, cfg, "posterior"), "config": cfg,
                   "tau_rank": tau, "ratio": ratio, "num_samples": len(target), "correct": cc, "accuracy": ca,
                   "base_correct": bc, "base_accuracy": ba, "last50_correct": lc, "last50_accuracy": la,
                   "base_last50_correct": blc, "base_last50_accuracy": bla, "delta_correct": cc-bc, "delta_pp": ca-ba,
                   "delta_last50_correct": lc-blc, "delta_last50_pp": la-bla, "prediction_sha256": core.pred_sha(ids, target, pred),
                   "state_sha256": grid["state_sha256"], "trajectory_sha256": grid["trajectory_sha256"],
                   "compatibility_sha256": grid["compatibility_sha256"], "health_status": grid["health"]["health_status"], **diag}
            rows.append(row)
    # paired artifacts are intentionally light and contain no feature tensor
    core.atomic_npz(out / dataset / "posterior_dataset_specific.npz", sample_id=ids, target=target,
                    base_prediction=base_pred, compat_sample=grid["compat_sample"], stable_top1=grid["stable_top1"],
                    dynamic_top1=grid["dynamic_top1"], **{k: v for k, v in grid["predictions"].items()})
    return rows


def local_update_specs(history: list[dict[str, Any]], dataset: str) -> list[tuple[float, float, str]]:
    old = [r for r in history if r.get("status") == "ok" and r.get("dataset") == dataset and r.get("phase") in {"update_u1", "update_u2"}
           and r.get("config", {}).get("update_rule") in U_RULES]
    if old: best = max(old, key=lambda r: (int(r.get("correct", 0)), float(r.get("last50_accuracy", 0))))
    else: best = {"config": {"rank": {"tau_rank": 0.15, "update_power": 1.0}, "update_rule": "mix08clip"}}
    tau = float(best["config"]["rank"]["tau_rank"]); power = float(best["config"]["rank"]["update_power"]); rule = str(best["config"].get("update_rule", "mix08clip"))
    taus = sorted({round(max(0.02, min(1.2, tau*m)), 8) for m in (0.75, 1.0, 1.25)})
    powers = sorted({round(max(0.05, min(4.0, power*m)), 8) for m in (0.75, 1.0, 1.25, 1.5)})
    rules = tuple(dict.fromkeys((rule, "mix05" if rule == "mix08clip" else "mix08clip")))
    return [(t, p, r) for t in taus for p in powers for r in rules]


def full_specs(p: Mapping[str, Any], u: Mapping[str, Any]) -> list[tuple[float, float, float, str]]:
    tau = float(u["config"]["rank"]["tau_rank"]); power = float(u["config"]["rank"]["update_power"]); rule = str(u["config"].get("update_rule", "mix08clip")); ps = float(p["config"]["rank"]["prediction_strength"])
    strengths = [0.0, 0.75*ps, ps, 1.25*ps] if ps else [0.0, 0.0375, 0.075]
    powers = [0.75*power, power, 1.25*power]; rules = tuple(dict.fromkeys((rule, "mix05" if rule == "mix08clip" else "mix08clip")))
    return [(tau, float(s), float(q), r) for s in strengths for q in powers for r in rules]


def copy_or_metric(history: list[dict[str, Any]], dataset: str, cfg: Mapping[str, Any], out: Path, cid: str,
                   base: Mapping[str, Any], identity: str, args: argparse.Namespace, data: Mapping[str, Any], phase: str,
                   stop: Path) -> dict[str, Any]:
    # A truncated smoke stream must never inherit a full-stream historical
    # metric.  Historical reuse is only valid for the formal complete run.
    old = None if args.max_samples is not None else historical_match(history, dataset, cfg)
    if old is not None:
        row = dict(old); row.update({"phase": phase, "candidate_id": cid, "candidate_fingerprint": core.candidate_fingerprint(dataset, cfg, phase), "source": "historical_reuse", "identity_sha256": identity, "config": cfg}); return row
    metric = core.replay_update(data, cfg, stop, args.stop_interval, out / dataset / f"{cid}.npz")
    return result_row(dataset, phase, cid, cfg, metric, base, identity, "fresh_replay")


def eurosat_state_diagnostics(data: Mapping[str, Any], config: Mapping[str, Any], stop: Path,
                             interval: int) -> dict[str, Any]:
    """Capture lightweight state checkpoints for the EuroSAT winner."""
    core.tune.validate_config(config)
    device = str(data["features"].device); dim, classes = map(int, data["clip_shape"])
    model = core.OCRDOTAV3(config["base"], {"rank": config["rank"], "update": config["update"]},
                           dim, classes, data["text_prototypes"], device=device)
    n = int(data["features"].shape[0]); targets = data["targets"].detach().cpu().numpy().reshape(-1)
    pred = np.empty(n, dtype=np.int64); gates = np.empty(n, dtype=np.float32)
    checkpoints = {int(n*f): f for f in (0.0, 0.25, 0.50, 0.75)}; snaps = {}
    def snap(frac: float) -> dict[str, Any]:
        tensors = {"C": model.C, "mu": model.mu, "pi": model.pi}
        if model.covariance_mode == "full": tensors.update({"Q": model.Q, "Sigma": model.Sigma, "Lambda": model.Lambda})
        else: tensors.update({"Q_diag": model.Q_diag, "Sigma_diag": model.Sigma_diag, "Lambda_diag": model.Lambda_diag})
        stats = {name: {"mean": float(t.detach().float().mean().item()), "std": float(t.detach().float().std().item()), "norm": float(t.detach().float().norm().item())} for name,t in tensors.items()}
        return {"fraction": frac, "sample_index": int(round(n*frac)), "state_sha256": core.model_state_sha(model), "state_stats": stats}
    with torch.no_grad():
        for i in range(n):
            if i in checkpoints: snaps[str(checkpoints[i])] = snap(checkpoints[i])
            if i % max(1, interval) == 0: core.check_stop(stop, f"eurosat diagnostics {i}")
            views=data["features"][i].to(device=device,dtype=torch.float32); z=views.mean(0,keepdim=True)
            clip_logits=data["clip_logits"][i:i+1].to(device=device,dtype=torch.float32); prob_map=data["prob_maps"][i].to(device=device,dtype=torch.float32)
            parts=model.posterior_and_responsibility(z,return_parts=True); cap=float(config["base"]["rho"])*model.C.mean()/views.size(0); weight=torch.clamp(cap,max=float(config["base"]["eta"]))
            pred[i]=int((clip_logits+weight*parts["rank_logits"]).argmax(-1).item()); gates[i]=float(parts["update_gate"].mean())
            alloc=core.tune.allocation_by_rule(config["update_rule"],prob_map,parts["p_geometry"]); model.fit_ocr(views,parts["update_gate"]*alloc); model.update()
    snaps["1.0"] = snap(1.0)
    return {"num_samples": n, "checkpoints": snaps, "mean_gate": float(gates.mean()), "gate_p10": float(np.percentile(gates,10)), "gate_p50": float(np.percentile(gates,50)), "gate_p90": float(np.percentile(gates,90)), "prediction_sha256": core.pred_sha(np.arange(n,dtype=np.int64),targets,pred)}


def select_best(rows: list[dict[str, Any]], dataset: str, phase: str) -> dict[str, Any]:
    rr = [r for r in rows if r.get("dataset") == dataset and r.get("phase") == phase and r.get("status") == "ok"]
    if not rr: raise RuntimeError(f"no {phase} rows for {dataset}")
    return max(rr, key=lambda r: (int(r["correct"]), int(r.get("last50_correct", 0)), -abs(float(r.get("config", {}).get("rank", {}).get("prediction_strength", 0.0)))))


def write_outputs(out: Path, datasets: tuple[str, ...], rows: list[dict[str, Any]], bases: list[dict[str, Any]], pbest: dict[str, Any], ubest: dict[str, Any], fbest: dict[str, Any], identity: Mapping[str, Any]) -> None:
    core.atomic_json(out / "posterior_best_by_dataset.json", {"selection": "per-dataset correct, corrected-regressed, local stability", "rows": pbest})
    core.atomic_json(out / "update_best_by_dataset.json", {"selection": "per-dataset overall correct, then last50", "rows": ubest})
    core.atomic_json(out / "full_best_by_dataset.json", {"selection": "per-dataset overall correct, then last50", "rows": fbest})
    p_stab=[]; p_corr=[]
    for d in datasets:
        rr=[r for r in rows if r.get("dataset")==d and r.get("phase")=="posterior"]; best=pbest[d]
        pos=[r for r in rr if int(r["delta_correct"])>0]; p_stab.append({"dataset":d,"display_name":DISPLAY[d],"best_delta_correct":best["delta_correct"],"positive_parameter_count":len(pos),"parameter_count":len(rr),"positive_parameter_ratio":len(pos)/max(1,len(rr)),"median_positive_delta":float(np.median([r["delta_correct"] for r in pos])) if pos else 0.0})
        p_corr.append({"dataset":d,"candidate":best["candidate_id"],"tau_rank":best["config"]["rank"]["tau_rank"],"prediction_strength":best["config"]["rank"]["prediction_strength"],"base_correct":best["base_correct"],"candidate_correct":best["correct"],"delta_correct":best["delta_correct"],"delta_pp":best["delta_pp"],"corrected":best.get("corrected",0),"regressed":best.get("regressed",0),"net_correction":best.get("net_correction",0)})
    write_csv(out/"POSTERIOR_PARAMETER_STABILITY.csv",p_stab,list(p_stab[0])); write_csv(out/"POSTERIOR_CORRECTION_ANALYSIS.csv",p_corr,list(p_corr[0]))
    u_an=[]
    for d in datasets:
        r=ubest[d]; u_an.append({"dataset":d,"candidate":r["candidate_id"],"tau_rank":r["config"]["rank"]["tau_rank"],"update_power":r["config"]["rank"]["update_power"],"update_rule":r["config"].get("update_rule"),"base_all_correct":r["base_correct"],"candidate_all_correct":r["correct"],"delta_correct_all":r["delta_correct"],"delta_all_pp":r["delta_pp"],"base_last50_accuracy":r.get("base_last50_accuracy"),"candidate_last50_accuracy":r.get("last50_accuracy"),"delta_last50_pp":r.get("delta_last50_pp"),"mean_gate":r.get("mean_gate"),"gate_p10":r.get("gate_p10"),"median_gate":r.get("median_gate"),"gate_p90":r.get("gate_p90")})
    write_csv(out/"UPDATE_RESPONSIBILITY_ANALYSIS.csv",u_an,list(u_an[0]))
    # EuroSAT receives the requested stream-segment diagnostic.  The replay
    # stores predictions/targets; gate quantiles come from the same cold
    # replay.  State checkpoints are intentionally represented by hashes only
    # (no full covariance tensors are written).
    if "eurosat" in datasets:
        er = ubest["eurosat"]; pred_path = out/"eurosat"/f"{er['candidate_id']}.npz"
        if not pred_path.exists(): pred_path = HISTORY_DEFAULT/"eurosat"/f"{er['candidate_id']}.npz"
        base_path = out/"eurosat"/"posterior_dataset_specific.npz"
        if pred_path.exists() and base_path.exists():
            q=np.load(pred_path); bp=np.load(base_path); pred=q["prediction"]; target=q["target"]; bpred=bp["base_prediction"]
            seg=[]
            for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.0)):
                a=int(len(target)*lo); z=int(len(target)*hi); seg.append({"start_fraction":lo,"end_fraction":hi,"num_samples":z-a,"base_correct":int((bpred[a:z]==target[a:z]).sum()),"responsibility_correct":int((pred[a:z]==target[a:z]).sum()),"base_accuracy":float(100*(bpred[a:z]==target[a:z]).mean()),"responsibility_accuracy":float(100*(pred[a:z]==target[a:z]).mean())})
            state_diag = {}
            state_file = out/"EUROSAT_STATE_DIAGNOSTICS.json"
            if state_file.exists(): state_diag = json.loads(state_file.read_text(encoding="utf-8"))
            core.atomic_json(out/"EUROSAT_RESPONSIBILITY_MECHANISM.json",{"candidate":er["candidate_id"],"config":er["config"],"segments":seg,"mean_gate":er.get("mean_gate"),"median_gate":er.get("median_gate"),"gate_p10":er.get("gate_p10"),"gate_p25":er.get("gate_p25"),"gate_p75":er.get("gate_p75"),"gate_p90":er.get("gate_p90"),"state_checkpoints":state_diag.get("checkpoints",{}),"state_checkpoint_note":"Each checkpoint stores state SHA and scalar norms/statistics; no full covariance tensor is persisted."})
    param=[]; main=[]
    for d in datasets:
        b=next(x for x in bases if x["dataset"]==d); p=pbest[d]; u=ubest[d]; f=fbest.get(d)
        main.append({"dataset":d,"display_name":DISPLAY[d],"base":f"{b['correct']}/{b['num_samples']} ({b['accuracy']:.4f}%)","posterior":f"{p['correct']}/{p['num_samples']} ({p['accuracy']:.4f}%)","delta_p":p['delta_correct'],"responsibility":f"{u['correct']}/{u['num_samples']} ({u['accuracy']:.4f}%)","delta_r":u['delta_correct'],"full":f"{f['correct']}/{f['num_samples']} ({f['accuracy']:.4f}%)" if f else "not_run","delta_full":f['delta_correct'] if f else ""})
        param.append({"dataset":d,"p_tau":p["config"]["rank"]["tau_rank"],"p_strength":p["config"]["rank"]["prediction_strength"],"u_tau":u["config"]["rank"]["tau_rank"],"u_power":u["config"]["rank"]["update_power"],"u_rule":u["config"].get("update_rule"),"full_tau":f["config"]["rank"]["tau_rank"] if f else "","full_strength":f["config"]["rank"]["prediction_strength"] if f else "","full_power":f["config"]["rank"]["update_power"] if f else "","full_rule":f["config"].get("update_rule") if f else ""})
    fields=list(main[0]); write_csv(out/"FINAL_DATASET_SPECIFIC_ABLATION.csv",main,fields)
    lines=["# OCR-DOTA V3 Dataset-specific Oracle Ablation", "", "Results are per-dataset oracle tuning on complete streams; they are not held-out generalization.", "", "| Dataset | Rank-free Base | Posterior | ΔP | Responsibility | ΔR | Full | ΔFull |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['display_name']} | {r['base']} | {r['posterior']} | {r['delta_p']:+d} | {r['responsibility']} | {r['delta_r']:+d} | {r['full']} | {r['delta_full'] if r['delta_full'] != '' else ''} |" for r in main]
    (out/"FINAL_DATASET_SPECIFIC_ABLATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); write_csv(out/"FINAL_DATASET_SPECIFIC_PARAMS.csv",param,list(param[0]))
    supported=[d for d in datasets if pbest[d]["delta_correct"]>0 and ubest[d]["delta_correct"]>0 and fbest.get(d,{}).get("delta_correct",-1)>0]
    report=["# OCR-DOTA V3 Dataset-specific Oracle Report", "", f"Identity: `{identity['identity_sha256']}`", "", "All parameter choices are dataset-specific oracle tuning on complete datasets. Labels are post-hoc only.", "", f"Posterior positive datasets: {sum(pbest[d]['delta_correct']>0 for d in datasets)}/10", f"Responsibility positive datasets: {sum(ubest[d]['delta_correct']>0 for d in datasets)}/10", f"Full positive datasets: {sum(fbest.get(d,{}).get('delta_correct',-1)>0 for d in datasets)}/10", f"Simultaneously positive: {', '.join(DISPLAY[d] for d in supported) if supported else 'none'}", "", "## Main table", "", *lines[4:]]
    (out/"FINAL_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    repo=Path(args.repo_root).resolve(); out=Path(args.output).resolve(); datasets=tuple(core.parse_datasets(args.datasets)); anchor_path=Path(args.anchor_source).resolve(); history_path=Path(args.history).resolve(); out.mkdir(parents=True,exist_ok=True)
    anchors, expected, sources=core.resolve_anchor(anchor_path,datasets,repo/"OCR-DOTA-V3/BEST_SINGLE_CONFIG_NONIMAGENET21.json"); caches={}
    for d in datasets:
        p=core.cache_path(repo,d); raw=torch.load(p,map_location="cpu",weights_only=False); n=int(raw["features"].shape[0]); ids=raw.get("sample_ids",torch.arange(n)); ids=ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids); caches[d]={"path":str(p),"sha256":core.sha256_file(p),"num_samples":n,"order_sha256":hashlib.sha256(ids.tobytes()).hexdigest()}
        e=expected.get(d,{})
        for k,v in (("cache_sha256",caches[d]["sha256"]),("order_sha256",caches[d]["order_sha256"])):
            if e.get(k) and str(e[k])!=str(v): raise RuntimeError(f"{d} {k} mismatch")
        if e.get("num_samples") and int(e["num_samples"])!=n: raise RuntimeError(f"{d} sample count mismatch")
    identity=make_identity(repo,anchor_path,anchors,caches,args.global_seed,history_path)
    if args.dry_run:
        print(json.dumps({"identity_sha256": identity["identity_sha256"], "datasets": list(datasets),
                          "posterior_heads": len(P_TAUS)*len(P_RATIOS),
                          "update_local_candidates_per_dataset": "derived from historical oracle",
                          "full_local_candidates_per_dataset": "derived from posterior/responsibility winners"}, ensure_ascii=False, indent=2))
        return 0
    man=out/"manifest.json"
    if man.exists() and not args.resume: raise RuntimeError(f"output exists: {out}; use --resume")
    if man.exists():
        old_manifest=json.loads(man.read_text())
        if old_manifest.get("identity_sha256")!=identity["identity_sha256"]:
            if not compatible_resume_identity(old_manifest, identity): raise RuntimeError("identity mismatch")
            core.atomic_json(man,{"status":"running",**identity,"started_at":old_manifest.get("started_at",time.time()),"resumed_from_identity":old_manifest.get("identity_sha256")})
    else: core.atomic_json(man,{"status":"running",**identity,"started_at":time.time()})
    hist=load_history(history_path); result_path=out/"candidate_results.jsonl"; rows=core.load_jsonl(result_path)
    # Re-index only prior posterior/U1/U2 outcomes.  Full-V3 history is not
    # used as a final configuration in this run.
    if not rows and args.max_samples is None:
        for h in hist:
            if h.get("status") != "ok" or h.get("dataset") not in datasets: continue
            if h.get("phase") == "posterior": phase = "posterior"
            elif h.get("phase") in {"update_u1", "update_u2"}: phase = "responsibility"
            else: continue
            q = dict(h); q["phase"] = phase; q["source"] = "historical_reuse"; q["identity_sha256"] = identity["identity_sha256"]
            q["candidate_fingerprint"] = core.candidate_fingerprint(q["dataset"], q["config"], phase)
            rows.append(q); core.append_jsonl(result_path, q)
    done={(r.get("dataset"),r.get("candidate_fingerprint")):r for r in rows if r.get("status")=="ok"}; stop=Path(args.stop).resolve() if args.stop else out/"STOP"; bases=[]; pbest={}; ubest={}; fbest={}
    with core.PidLock(out/"RUNNING.pid"):
        # Every dataset gets one fresh shared trajectory; this is also the strict Base reproduction.
        for d in datasets:
            core.check_stop(stop,d); core.setup_seed(args.global_seed); data,meta=core.legacy.load_cache(Path(caches[d]["path"]),args.device,args.max_samples); anchor=anchors[d]; base_cfg=core.make_config(anchor,prediction_strength=0,power=0,tau=.15)
            # posterior_rows performs the single shared full replay and emits
            # the P0 head used as the Rank-free Base reproduction.
            prow=posterior_rows(d,data,anchor,{"correct":0,"accuracy":0.0,"last50_correct":0,"last50_accuracy":0},identity["identity_sha256"],out,stop,args.stop_interval)
            p0=next(r for r in prow if r["tau_rank"]==0.15 and r["ratio"]==0.0)
            b={"dataset":d,"display_name":DISPLAY[d],"num_samples":p0["num_samples"],"correct":p0["correct"],"accuracy":p0["accuracy"],"last50_correct":p0["last50_correct"],"last50_accuracy":p0["last50_accuracy"],"prediction_sha256":p0["prediction_sha256"],"state_sha256":p0["state_sha256"]}; bases.append(b)
            old_base=next((r for r in hist if r.get("dataset")==d and r.get("phase")=="base"),None)
            if args.max_samples is None and old_base and (int(old_base["correct"])!=b["correct"] or old_base.get("prediction_sha256")!=b["prediction_sha256"]): raise RuntimeError(f"historical Base mismatch {d}")
            # Replace the provisional baseline fields in the paired rows.
            for r in prow:
                r["base_correct"]=b["correct"]; r["base_accuracy"]=b["accuracy"]; r["base_last50_correct"]=b["last50_correct"]; r["base_last50_accuracy"]=b["last50_accuracy"]
                r["delta_correct"]=int(r["correct"])-b["correct"]; r["delta_pp"]=float(r["accuracy"])-b["accuracy"]; r["delta_last50_correct"]=int(r["last50_correct"])-b["last50_correct"]; r["delta_last50_pp"]=float(r["last50_accuracy"])-b["last50_accuracy"]
            # P2: one bounded local ratio refinement around this dataset's
            # coarse winner; it shares a trajectory as well.
            coarse = max(prow, key=lambda x: (int(x["correct"]), -abs(float(x["ratio"]))))
            r0 = float(coarse["ratio"])
            # Always include the ratio=0 control: posterior_rows uses it to
            # derive a tau-independent Rank-free Base prediction head.
            local_ratios = sorted(({0.0} | {round(max(0.0, r0*m), 8) for m in (0.70, 0.85, 1.0, 1.15, 1.30)})) if r0 else (0.0, 0.05, 0.10, 0.15, 0.25)
            local = posterior_rows(d, data, anchor, b, identity["identity_sha256"], out, stop, args.stop_interval,
                                   taus=(float(coarse["tau_rank"]),), ratios=tuple(local_ratios), prefix="P2")
            for r in local:
                r["base_correct"]=b["correct"]; r["base_accuracy"]=b["accuracy"]; r["base_last50_correct"]=b["last50_correct"]; r["base_last50_accuracy"]=b["last50_accuracy"]
                r["delta_correct"]=int(r["correct"])-b["correct"]; r["delta_pp"]=float(r["accuracy"])-b["accuracy"]; r["delta_last50_correct"]=int(r["last50_correct"])-b["last50_correct"]; r["delta_last50_pp"]=float(r["last50_accuracy"])-b["last50_accuracy"]
                prow.append(r)
            for r in prow:
                if (d,r["candidate_fingerprint"]) not in done: core.append_jsonl(result_path,r); rows.append(r); done[(d,r["candidate_fingerprint"])]=r
            del data
        # Dataset-specific posterior best and local responsibility candidates.
        for d in datasets: pbest[d]=select_best(rows,d,"posterior")
        for d in datasets:
            core.check_stop(stop,f"update {d}"); core.setup_seed(args.global_seed); data,_=core.legacy.load_cache(Path(caches[d]["path"]),args.device,args.max_samples); b=next(x for x in bases if x["dataset"]==d); anchor=anchors[d]
            for i,(tau,power,rule) in enumerate(local_update_specs(hist,d)):
                cfg=core.make_config(anchor,prediction_strength=0,tau=tau,power=power,update_rule=rule); cid=f"U_t{tau:g}_p{power:g}_{rule}"; fp=core.candidate_fingerprint(d,cfg,"responsibility")
                if (d,fp) in done: continue
                r=copy_or_metric(hist,d,cfg,out,cid,b,identity["identity_sha256"],args,data,"responsibility",stop); r["phase"]="responsibility"; r["candidate_id"]=cid; r["candidate_fingerprint"]=fp; core.append_jsonl(result_path,r); rows.append(r); done[(d,fp)]=r
            del data
        for d in datasets: ubest[d]=select_best(rows,d,"responsibility")
        # Per-dataset Full local joint search; no global configuration is consulted.
        for d in datasets:
            core.check_stop(stop,f"full {d}"); core.setup_seed(args.global_seed); data,_=core.legacy.load_cache(Path(caches[d]["path"]),args.device,args.max_samples); b=next(x for x in bases if x["dataset"]==d); anchor=anchors[d]; p=pbest[d]; u=ubest[d]
            for tau,strength,power,rule in full_specs(p,u):
                cfg=core.make_config(anchor,prediction_strength=strength,tau=tau,power=power,update_rule=rule); cid=f"F_t{tau:g}_s{strength:g}_p{power:g}_{rule}"; fp=core.candidate_fingerprint(d,cfg,"full")
                if (d,fp) in done: continue
                metric=core.replay_update(data,cfg,stop,args.stop_interval,out/d/f"{cid}.npz"); r=result_row(d,"full",cid,cfg,metric,b,identity["identity_sha256"],"fresh_replay"); core.append_jsonl(result_path,r); rows.append(r); done[(d,fp)]=r
            del data
        for d in datasets: fbest[d]=select_best(rows,d,"full")
        if "eurosat" in datasets:
            core.setup_seed(args.global_seed); edata,_=core.legacy.load_cache(Path(caches["eurosat"]["path"]),args.device,args.max_samples)
            diag=eurosat_state_diagnostics(edata,ubest["eurosat"]["config"],stop,args.stop_interval)
            core.atomic_json(out/"EUROSAT_STATE_DIAGNOSTICS.json",diag); del edata
        write_outputs(out,datasets,rows,bases,pbest,ubest,fbest,identity)
        core.atomic_json(man,{"status":"complete",**identity,"finished_at":time.time(),"candidate_count":len(rows),"positive": {"posterior":sum(pbest[d]["delta_correct"]>0 for d in datasets),"responsibility":sum(ubest[d]["delta_correct"]>0 for d in datasets),"full":sum(fbest[d]["delta_correct"]>0 for d in datasets)}})
    print(json.dumps({"status":"complete","output":str(out),"rows":len(rows),"posterior_positive":sum(pbest[d]["delta_correct"]>0 for d in datasets),"responsibility_positive":sum(ubest[d]["delta_correct"]>0 for d in datasets),"full_positive":sum(fbest[d]["delta_correct"]>0 for d in datasets)},ensure_ascii=False,indent=2)); return 0


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo-root",default=str(REPO)); p.add_argument("--output",default=str(ROOT/"log/core2_dataset_specific_oracle10_20260910")); p.add_argument("--anchor-source",default=str(REPO/"V3-P0-ablation/results/nonimagenet21_p0_20260906/results.jsonl")); p.add_argument("--history",default=str(HISTORY_DEFAULT)); p.add_argument("--device",default="cuda"); p.add_argument("--datasets"); p.add_argument("--global-seed",type=int,default=1); p.add_argument("--stop-interval",type=int,default=25); p.add_argument("--stop"); p.add_argument("--max-samples",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--dry-run",action="store_true"); return p


if __name__ == "__main__": raise SystemExit(run(parser().parse_args()))
