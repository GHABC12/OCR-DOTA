#!/usr/bin/env python3
"""P0 true-online ablation for OCR-DOTA-V3, isolated from the model tree."""

from __future__ import annotations

import argparse, copy, csv, hashlib, json, math, os, sys, time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V3 = REPO / "OCR-DOTA-V3"
sys.path[:0] = [str(REPO), str(V3)]
import tune_nonimagenet21 as core  # noqa: E402
from ocr_dota_v3 import OCRDOTAV3  # noqa: E402
from scripts import run_cross_benchmark_ocr_ablation as legacy  # noqa: E402

VERSION = "ocr-dota-v3-p0-online-v1"
WINNERS = V3 / "BEST_SINGLE_CONFIG_NONIMAGENET21_CHAIN_ROUND3.json"
DEFAULT_OUT = HERE / "results" / "nonimagenet21_p0_20260906"
VARIANTS = {
    "rank_free": (0, 0, 0), "prediction_only": (0, 1, 0),
    "update_only": (0, 0, 1), "pure_rank": (0, 1, 1),
    "geometry_only": (1, 0, 0), "full_v3": (1, 1, 1),
}

def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)

def stable_sha(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def config_for(full: Mapping[str, Any], mask: tuple[int, int, int]) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(full))
    cfg["update"]["residual_strength"] = float(full["update"]["residual_strength"]) if mask[0] else 0.0
    cfg["rank"]["prediction_strength"] = float(full["rank"]["prediction_strength"]) if mask[1] else 0.0
    cfg["rank"]["update_power"] = float(full["rank"]["update_power"]) if mask[2] else 0.0
    core.validate_config(cfg)
    return cfg

def trace_sha(arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(canonical(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    with tmp.open("rb") as f: os.fsync(f.fileno())
    os.replace(tmp, path)

def top2_margin(logits: torch.Tensor) -> float:
    if logits.shape[-1] < 2: return float("inf")
    v = logits.topk(2, dim=-1).values
    return float((v[:, 0] - v[:, 1]).item())

def replay(data: Mapping[str, Any], config: Mapping[str, Any], stop: Path, interval: int,
           trace_path: Path | None = None) -> dict[str, Any]:
    """Same online order as core.replay_v3; diagnostics do not affect model tensors."""
    core.validate_config(config)
    device = str(data["features"].device); dim, classes = map(int, data["clip_shape"])
    model = OCRDOTAV3(config["base"], {"rank": config["rank"], "update": config["update"]},
                      dim, classes, data["text_prototypes"], device=device).eval()
    n = int(data["features"].shape[0]); dtype = legacy.compact_dtype(classes)
    targets = data["targets"].detach().cpu().numpy().reshape(-1).astype(dtype, copy=False)
    ids = np.asarray(data["sample_ids"], dtype=np.int64); pred = np.empty(n, dtype=dtype)
    arrays = {
        "sample_id": ids, "target": targets, "prediction": pred,
        "geometry_prediction": np.empty(n, dtype=dtype), "allocation_prediction": np.empty(n, dtype=dtype),
        "sample_compatibility": np.empty(n, np.float32), "base_margin": np.empty(n, np.float32),
        "gate": np.empty(n, np.float32), "true_weight": np.empty(n, np.float32),
        "total_weight": np.empty(n, np.float32),
    }
    traj = hashlib.sha256(canonical(config).encode()); compat_h = hashlib.sha256(); started = time.time()
    if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i in range(n):
            if i % max(1, interval) == 0 and stop.exists(): raise core.StopRequested(f"STOP at {i}/{n}")
            views = data["features"][i].to(device=device, dtype=torch.float32)
            clip = data["clip_logits"][i:i+1].to(device=device, dtype=torch.float32)
            prob = data["prob_maps"][i].to(device=device, dtype=torch.float32); z = views.mean(0, keepdim=True)
            parts = model.posterior_and_responsibility(z, return_parts=True)
            pre = float(config["base"]["rho"]) * model.C.mean() / views.size(0)
            weight = torch.clamp(pre, max=float(config["base"]["eta"]))
            final = clip + weight * parts["rank_logits"]
            geom_final = clip + weight * parts["geometry_logits"]
            if not bool(torch.isfinite(final).all()): raise FloatingPointError(f"non-finite prediction at {i}")
            pred[i] = int(final.argmax(-1)); arrays["geometry_prediction"][i] = int(geom_final.argmax(-1))
            arrays["base_margin"][i] = top2_margin(geom_final)
            features = core.update_features(views, z, config["update_views"])
            gu = model.gaussian_logits(features); ru = model._residual_magnitude(features)
            pg = F.softmax(gu - model.residual_strength * ru, dim=-1)
            allocation = core.allocation_by_rule(config["update_rule"], prob, pg)
            gate = parts["update_gate"]; weights = gate * allocation
            if not bool(torch.isfinite(weights).all()): raise FloatingPointError(f"non-finite update at {i}")
            mean_alloc = allocation.mean(0)
            arrays["allocation_prediction"][i] = int(mean_alloc.argmax())
            arrays["sample_compatibility"][i] = float(parts["sample_compatibility"].item())
            arrays["gate"][i] = float(gate.item())
            arrays["true_weight"][i] = float(weights[:, int(targets[i])].sum().item())
            arrays["total_weight"][i] = float(weights.sum().item())
            traj.update(i.to_bytes(8, "little")); traj.update(weights.detach().contiguous().cpu().numpy().tobytes())
            compat_h.update(i.to_bytes(8, "little")); compat_h.update(parts["rank_compatibility"].detach().contiguous().cpu().numpy().tobytes())
            model.fit_ocr(features, weights); model.update()
    health = core.health(model); sha = trace_sha(arrays)
    if trace_path is not None: atomic_npz(trace_path, arrays)
    correct = int((pred == targets).sum())
    return {"correct": correct, "num_samples": n, "accuracy": 100.0*correct/n,
            "prediction_sha256": core.prediction_sha(ids, targets, pred), "trajectory_sha256": traj.hexdigest(),
            "state_sha256": core.state_sha(model), "compatibility_sha256": compat_h.hexdigest(),
            "analysis_trace_sha256": sha, "health_status": health["health_status"], "state_health": health,
            "mean_update_gate": float(arrays["gate"].mean()), "elapsed_sec": time.time()-started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0}

def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = int(labels.sum()); neg = len(labels)-pos
    if not pos or not neg: return None
    order = np.argsort(scores, kind="stable"); ranks = np.empty(len(scores), float)
    i = 0
    while i < len(scores):
        j=i+1
        while j<len(scores) and scores[order[j]]==scores[order[i]]: j+=1
        ranks[order[i:j]]=(i+j+1)/2; i=j
    return float((ranks[labels].sum()-pos*(pos+1)/2)/(pos*neg))

def rankdata(x: np.ndarray) -> np.ndarray:
    order=np.argsort(x, kind="stable"); ranks=np.empty(len(x), float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        ranks[order[i:j]]=(i+j-1)/2; i=j
    return ranks

def summarize_trace(path: Path) -> dict[str, Any]:
    z=np.load(path); y=z["target"]; p=z["prediction"]; c=z["sample_compatibility"].astype(float)
    ok=p==y; rc=rankdata(c); ry=rankdata(ok.astype(float)); den=np.std(rc)*np.std(ry)
    bins=[]
    for idx in np.array_split(np.argsort(c, kind="stable"), 10):
        bins.append({"n":len(idx),"mean_compatibility":float(c[idx].mean()),"accuracy":float(ok[idx].mean()*100)})
    tw=z["true_weight"].astype(float); total=z["total_weight"].astype(float); alloc_ok=z["allocation_prediction"]==y
    windows=[]
    for idx in np.array_split(np.arange(len(y)),10): windows.append({"n":len(idx),"accuracy":float((p[idx]==y[idx]).mean()*100)})
    return {"correct_compatibility_mean":float(c[ok].mean()) if ok.any() else None,
            "wrong_compatibility_mean":float(c[~ok].mean()) if (~ok).any() else None,
            "compatibility_auroc":auc(c,ok), "compatibility_spearman":None if den==0 else float(np.corrcoef(rc,ry)[0,1]),
            "reliability_bins":bins, "true_weight_sum":float(tw.sum()), "total_weight_sum":float(total.sum()),
            "wrm":float(1-tw.sum()/total.sum()), "crm":float(tw.sum()/total.sum()),
            "allocation_correct_count":int(alloc_ok.sum()), "allocation_wrong_count":int((~alloc_ok).sum()),
            "gate_correct_allocation_sum":float(z["gate"][alloc_ok].sum()), "gate_wrong_allocation_sum":float(z["gate"][~alloc_ok].sum()),
            "gate_correct_allocation_mean":float(z["gate"][alloc_ok].mean()) if alloc_ok.any() else None,
            "gate_wrong_allocation_mean":float(z["gate"][~alloc_ok].mean()) if (~alloc_ok).any() else None,
            "gate_correct_allocation_quantiles":np.quantile(z["gate"][alloc_ok],[0,.25,.5,.75,1]).tolist() if alloc_ok.any() else None,
            "gate_wrong_allocation_quantiles":np.quantile(z["gate"][~alloc_ok],[0,.25,.5,.75,1]).tolist() if (~alloc_ok).any() else None,
            "windows":windows,"late50_accuracy":float((p[len(p)//2:]==y[len(y)//2:]).mean()*100)}

def pair_analysis(base_path: Path, pred_path: Path) -> dict[str, Any]:
    a=np.load(base_path); b=np.load(pred_path); y=a["target"]; pa=a["prediction"]; pb=b["prediction"]
    if not np.array_equal(y,b["target"]) or not np.array_equal(a["sample_id"],b["sample_id"]): raise RuntimeError("pair identity mismatch")
    w2c=(pa!=y)&(pb==y); c2w=(pa==y)&(pb!=y); c2c=(pa==y)&(pb==y); w2w=(pa!=y)&(pb!=y)
    result={"wrong_to_correct":int(w2c.sum()),"correct_to_wrong":int(c2w.sum()),
            "correct_to_correct":int(c2c.sum()),"wrong_to_wrong":int(w2w.sum()),
            "ncr":float((w2c.sum()-c2w.sum())/len(y))}
    for field in ("sample_compatibility","base_margin"):
        bins=[]
        for idx in np.array_split(np.argsort(a[field],kind="stable"),10):
            bins.append({"n":len(idx),"mean":float(a[field][idx].mean()),"w2c_rate":float(w2c[idx].mean()),"c2w_rate":float(c2w[idx].mean()),"ncr":float(w2c[idx].mean()-c2w[idx].mean())})
        result[field+"_bins"]=bins
    return result

def compact(m: Mapping[str,Any]) -> dict[str,Any]:
    return {k:m[k] for k in ("correct","num_samples","accuracy","prediction_sha256","trajectory_sha256","state_sha256","compatibility_sha256","analysis_trace_sha256","health_status","state_health","mean_update_gate","elapsed_sec","peak_cuda_bytes") if k in m}

def parser():
    p=argparse.ArgumentParser(); p.add_argument("--output",default=str(DEFAULT_OUT)); p.add_argument("--datasets")
    p.add_argument("--device",default="cuda"); p.add_argument("--global-seed",type=int,default=1); p.add_argument("--max-samples",type=int)
    p.add_argument("--stop-check-interval",type=int,default=25); p.add_argument("--winner-source",default=str(WINNERS)); p.add_argument("--resume",action="store_true")
    return p.parse_args()

def main() -> int:
    args=parser(); datasets=tuple(core.normalize_dataset(x) for x in args.datasets.split(",")) if args.datasets else core.ALL_DATASETS
    if len(set(datasets))!=len(datasets): raise ValueError("duplicate datasets")
    out=Path(args.output).resolve(); stop=out/"STOP"; source=Path(args.winner_source).resolve(); winners=json.loads(source.read_text())["datasets"]
    paths={d:core.cache_path(d,core.TEN_CACHE_ROOT,core.DOMAIN_CACHE_ROOT) for d in datasets}; cm=json.loads(core.EXPECTED_CACHE_MANIFEST.read_text())
    original={d:core.load_original_base(REPO,d) for d in datasets}
    identity={"version":VERSION,"datasets":datasets,"seed":args.global_seed,"max_samples":args.max_samples,"precision":"fp32","winner_source":str(source),"winner_sha256":core.sha256_file(source),
              "code_sha256":{"runner":core.sha256_file(Path(__file__)),"model":core.sha256_file(V3/"ocr_dota_v3/model.py"),"rank":core.sha256_file(V3/"ocr_dota_v3/rank_compatibility.py"),"core":core.sha256_file(V3/"tune_nonimagenet21.py"),"legacy":core.sha256_file(REPO/"scripts/run_cross_benchmark_ocr_ablation.py")},
              "caches":{d:{"sha256":core.sha256_file(paths[d]),"num_samples":cm[d]["num_samples"],"order_sha256":cm[d]["order_sha256"]} for d in datasets},
              "winner_configs":{d:stable_sha(winners[d]["config"]) for d in datasets},"original_bases":original,"variants":["original_dota","matched_base_dota",*VARIANTS]}
    iid=stable_sha(identity); out.mkdir(parents=True,exist_ok=True); manifest=out/"manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if not args.resume or old.get("identity_sha256")!=iid: raise RuntimeError("resume identity mismatch or --resume missing")
    else: core.atomic_json(manifest,{"status":"running","identity":identity,"identity_sha256":iid,"started_at":time.time()})
    rp=out/"results.jsonl"; vp=out/"verification.jsonl"; rows=core.load_jsonl(rp); vrows=core.load_jsonl(vp)
    done={(r.get("dataset"),r.get("variant")):r for r in rows if r.get("status")=="ok" and r.get("identity_sha256")==iid}
    verified={(r.get("dataset"),r.get("variant")) for r in vrows if r.get("status")=="reproduced" and r.get("identity_sha256")==iid}
    try:
      with core.PidLock(out/"RUNNING.pid"):
       for di,d in enumerate(datasets):
        if stop.exists(): raise core.StopRequested("STOP before dataset")
        core.setup_seed(args.global_seed); data,meta=legacy.load_cache(paths[d],args.device,args.max_samples); exp=identity["caches"][d]
        if meta["sha256"]!=exp["sha256"] or (args.max_samples is None and (meta["num_samples"]!=exp["num_samples"] or meta["order_sha256"]!=exp["order_sha256"])): raise RuntimeError(f"cache identity mismatch {d}")
        full=winners[d]["config"]; specs=[("original_dota","dota",original[d]),("matched_base_dota","dota",full["base"])]+[(n,"v3",config_for(full,m)) for n,m in VARIANTS.items()]
        for vi,(name,kind,cfg) in enumerate(specs):
            key=(d,name); tpath=out/"traces"/d/f"{name}.npz"
            if key not in done:
                core.atomic_json(out/"state.json",{"status":"running","dataset":d,"dataset_index":di,"variant":name,"variant_index":vi,"updated_at":time.time()}); core.setup_seed(args.global_seed)
                if kind=="dota": metrics=core.replay_dota(data,cfg,stop,args.stop_check_interval); metrics["analysis_trace_sha256"]="not_applicable"
                else: metrics=replay(data,cfg,stop,args.stop_check_interval,tpath)
                row={"status":"ok","identity_sha256":iid,"dataset":d,"variant":name,"kind":kind,"config":cfg,"config_sha256":stable_sha(cfg),"cache_sha256":meta["sha256"],"order_sha256":meta["order_sha256"],**compact(metrics)}
                if kind=="v3" and VARIANTS[name][2]==0 and abs(row["mean_update_gate"]-1)>1e-7: raise RuntimeError(f"U0 gate not identity {d}/{name}")
                core.append_jsonl(rp,row); rows.append(row); done[key]=row
            elif kind=="v3" and (not tpath.exists() or trace_sha(dict(np.load(tpath)))!=done[key]["analysis_trace_sha256"]): raise RuntimeError(f"trace missing/mismatch {d}/{name}")
            if key not in verified:
                core.setup_seed(args.global_seed)
                again=core.replay_dota(data,cfg,stop,args.stop_check_interval) if kind=="dota" else replay(data,cfg,stop,args.stop_check_interval,None)
                if kind=="dota": again["analysis_trace_sha256"]="not_applicable"
                fields=("correct","num_samples","prediction_sha256","trajectory_sha256","state_sha256","analysis_trace_sha256")+(("compatibility_sha256",) if kind=="v3" else ())
                mm={f:[done[key].get(f),again.get(f)] for f in fields if done[key].get(f)!=again.get(f)}
                if mm: raise RuntimeError(f"cold replay mismatch {d}/{name}: {mm}")
                core.append_jsonl(vp,{"status":"reproduced","identity_sha256":iid,"dataset":d,"variant":name,"fields":fields,"elapsed_sec":again["elapsed_sec"]}); verified.add(key)
        # P must not change update state when G/U are equal.
        for a,b in (("rank_free","prediction_only"),("update_only","pure_rank")):
            for f in ("state_sha256","compatibility_sha256"):
                if done[(d,a)][f]!=done[(d,b)][f]: raise RuntimeError(f"prediction polluted update {d}/{a}/{b}/{f}")
        analyses={n:summarize_trace(out/"traces"/d/f"{n}.npz") for n in VARIANTS}
        analyses["prediction_correction"]=pair_analysis(out/"traces"/d/"rank_free.npz",out/"traces"/d/"prediction_only.npz")
        base_windows=analyses["rank_free"]["windows"]; base_late=analyses["rank_free"]["late50_accuracy"]
        for n in VARIANTS:
            analyses[n]["window_gain_vs_rank_free_pp"]=[analyses[n]["windows"][j]["accuracy"]-base_windows[j]["accuracy"] for j in range(10)]
            analyses[n]["late50_gain_vs_rank_free_pp"]=analyses[n]["late50_accuracy"]-base_late
        core.atomic_json(out/"analysis"/f"{d}.json",analyses)
        del data
        if torch.cuda.is_available(): torch.cuda.empty_cache()
       payload=build_report(datasets,done,out)
       core.atomic_json(out/"summary.json",payload); write_report(out,payload)
       status="complete_smoke" if args.max_samples is not None else "complete"; core.atomic_json(out/"state.json",{"status":status,"datasets":len(datasets),"verified":len(verified),"finished_at":time.time()})
       m=json.loads(manifest.read_text()); m.update({"status":status,"finished_at":time.time(),"summary_sha256":core.sha256_file(out/"summary.json")}); core.atomic_json(manifest,m)
    except core.StopRequested as e:
        core.atomic_json(out/"state.json",{"status":"stopped","reason":str(e),"updated_at":time.time()}); return 75
    except Exception as e:
        core.atomic_json(out/"state.json",{"status":"failed","error":f"{type(e).__name__}: {e}","updated_at":time.time()});
        m=json.loads(manifest.read_text()); m.update({"status":"failed","error":f"{type(e).__name__}: {e}","updated_at":time.time()}); core.atomic_json(manifest,m); raise
    return 0

def build_report(datasets,done,out):
    per={}
    for d in datasets:
        if not all((d,n) in done for n in ["original_dota","matched_base_dota",*VARIANTS]): continue
        full=done[(d,"full_v3")]["config"]; active={"geometry":full["update"]["residual_strength"]>0,"prediction":full["rank"]["prediction_strength"]>0,"update":full["rank"]["update_power"]>0}
        stages={n:{"correct":done[(d,n)]["correct"],"accuracy":done[(d,n)]["accuracy"]} for n in ["original_dota","matched_base_dota",*VARIANTS]}
        a=stages; interaction=a["pure_rank"]["accuracy"]-a["prediction_only"]["accuracy"]-a["update_only"]["accuracy"]+a["rank_free"]["accuracy"]
        per[d]={"num_samples":done[(d,"original_dota")]["num_samples"],"active":active,
                "effect_status":{"prediction":"active" if active["prediction"] else "N/A_champion_strength_zero",
                                 "update":"active" if active["update"] else "N/A_champion_strength_zero",
                                 "geometry":"active" if active["geometry"] else "N/A_champion_strength_zero"},
                "stages":stages,"rank_interaction_pp":interaction,"analysis":json.loads((out/"analysis"/f"{d}.json").read_text())}
    aggregate=None
    if len(per)==len(datasets):
        names=["original_dota","matched_base_dota",*VARIANTS]; total=sum(x["num_samples"] for x in per.values()); micro={n:sum(x["stages"][n]["correct"] for x in per.values()) for n in names}; macro={n:sum(x["stages"][n]["accuracy"] for x in per.values())/len(per) for n in names}
        corr=[x["analysis"]["prediction_correction"] for x in per.values()]
        w2c=sum(x["wrong_to_correct"] for x in corr); c2w=sum(x["correct_to_wrong"] for x in corr)
        diagnostic={}
        for arm in VARIANTS:
            aa=[x["analysis"][arm] for x in per.values()]; tw=sum(x["true_weight_sum"] for x in aa); mass=sum(x["total_weight_sum"] for x in aa)
            diagnostic[arm]={"macro_compatibility_auroc":sum(x["compatibility_auroc"] for x in aa if x["compatibility_auroc"] is not None)/max(1,sum(x["compatibility_auroc"] is not None for x in aa)),
                             "macro_compatibility_spearman":sum(x["compatibility_spearman"] for x in aa if x["compatibility_spearman"] is not None)/max(1,sum(x["compatibility_spearman"] is not None for x in aa)),
                             "micro_wrm":1-tw/mass,"micro_crm":tw/mass,
                             "macro_late50_accuracy":sum(x["late50_accuracy"] for x in aa)/len(aa),
                             "macro_window_accuracy":[sum(x["windows"][j]["accuracy"] for x in aa)/len(aa) for j in range(10)]}
        aggregate={"num_samples":total,"micro_correct":micro,"micro_accuracy":{n:100*v/total for n,v in micro.items()},"macro_accuracy":macro,
                   "active_counts":{f:sum(x["active"][f] for x in per.values()) for f in ("geometry","prediction","update")},
                   "prediction_correction":{"wrong_to_correct":w2c,"correct_to_wrong":c2w,"ncr":(w2c-c2w)/total},"diagnostic":diagnostic}
    return {"status":"complete" if aggregate else "running","protocol":"frozen-winner full-stream true-online P0 diagnostic","per_dataset":per,"aggregate":aggregate}

def write_report(out,p):
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); names=["original_dota","matched_base_dota",*VARIANTS]; w.writerow(["dataset","N",*names,"G_active","P_active","U_active","rank_interaction_pp"])
        for d,x in p["per_dataset"].items(): w.writerow([d,x["num_samples"],*[x["stages"][n]["accuracy"] for n in names],x["active"]["geometry"],x["active"]["prediction"],x["active"]["update"],x["rank_interaction_pp"]])
    lines=["OCR-DOTA-V3 P0真实在线消融","冻结每流最新冠军；关闭项只置零，原冠军为零时记N/A，不擅自改强度。",""]
    if p["aggregate"]:
        a=p["aggregate"]; lines += [f"总样本: {a['num_samples']}",f"Active counts: {a['active_counts']}",
            f"Prediction纠错 W2C/C2W/NCR: {a['prediction_correction']['wrong_to_correct']}/{a['prediction_correction']['correct_to_wrong']}/{a['prediction_correction']['ncr']:.8f}","", "Macro accuracy:"]+[f"- {k}: {v:.6f}%" for k,v in a["macro_accuracy"].items()]
    (out/"report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__ == "__main__": raise SystemExit(main())
