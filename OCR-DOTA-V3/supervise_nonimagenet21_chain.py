#!/usr/bin/env python3
"""Run three sealed adaptive continuation rounds for the 21 non-ImageNet streams."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

ALL=("fgvc","caltech101","stanford_cars","dtd","eurosat","oxford_flowers","food101","oxford_pets","sun397","ucf101","office_home_art","office_home_clipart","office_home_product","office_home_real_world","visda2017_validation","domainnet_clipart","domainnet_infograph","domainnet_painting","domainnet_quickdraw","domainnet_real","domainnet_sketch")

def read(p): return json.loads(p.read_text(encoding="utf-8"))
def atomic(p,v):
    t=p.with_name(f".{p.name}.tmp.{os.getpid()}");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(t,p)
def sealed(run:Path):
    m=read(run/"manifest.json"); s=read(run/"state.json"); w=read(run/"winners.json")["datasets"]
    if m.get("status") not in {"complete","complete_budget_limited"} or s.get("status") not in {"complete","complete_budget_limited"}: raise RuntimeError(f"unsealed {run}")
    if set(w)!=set(ALL): raise RuntimeError("winner set mismatch")
    for d,r in w.items():
        if r.get("health_status")!="healthy": raise RuntimeError(f"unhealthy {d}")
        v=r.get("verification",{})
        if any(v.get(k)!=r.get(k) for k in ("correct","prediction_sha256","state_sha256","trajectory_sha256")): raise RuntimeError(f"unverified {d}")
    return w,s

def main():
    p=argparse.ArgumentParser();p.add_argument("--repo-root",required=True);p.add_argument("--parent",required=True);p.add_argument("--output",required=True);p.add_argument("--python",required=True);p.add_argument("--total-budget-hours",type=float,default=10.0);a=p.parse_args()
    repo,parent,out=Path(a.repo_root).resolve(),Path(a.parent).resolve(),Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);stop=out/"STOP";state=out/"state.json";deadline=time.time()+a.total_budget_hours*3600
    prior_winners,_=sealed(parent); active=list(ALL); latest=parent
    for rnd in (1,2,3):
        if not active or stop.exists(): break
        remaining=(deadline-time.time())/3600
        if remaining<=1.5: break
        target=out/f"round{rnd}"; cmd=[a.python,str(repo/"OCR-DOTA-V3/tune_nonimagenet21_chain_round.py"),"--repo-root",str(repo),"--source",str(latest),"--output",str(target),"--datasets",",".join(active),"--round-index",str(rnd),"--time-budget-hours",str(remaining),"--verification-reserve-hours","1.25"]
        if (target/"manifest.json").exists(): cmd.append("--resume")
        atomic(state,{"status":"running","round":rnd,"active":active,"command":cmd,"updated_at":time.time()})
        proc=subprocess.Popen(cmd,cwd=repo)
        while proc.poll() is None:
            if stop.exists(): (target/"STOP").touch()
            time.sleep(10)
        if proc.returncode!=0:
            atomic(state,{"status":"interrupted" if stop.exists() else "failed","round":rnd,"returncode":proc.returncode,"updated_at":time.time()});return proc.returncode
        winners,child_state=sealed(target)
        improved=[d for d in active if int(winners[d]["correct"])>int(prior_winners[d]["correct"])]
        atomic(out/f"round{rnd}_handoff.json",{"parent":str(latest),"child":str(target),"active":active,"improved":improved,"parent_correct":{d:prior_winners[d]["correct"] for d in active},"child_correct":{d:winners[d]["correct"] for d in active},"child_state":child_state})
        latest=target;prior_winners=winners;active=improved
    winners,_=sealed(latest)
    payload={"status":"complete","protocol":"three-round adaptive, full-stream online FP32 seed1","latest_run":str(latest),"active_after_last_round":active,"datasets":winners}
    atomic(out/"BEST_SINGLE_CONFIG_NONIMAGENET21_CHAIN.json",payload)
    (out/"BEST_SINGLE_CONFIG_NONIMAGENET21_CHAIN.txt").write_text("\n".join(f"{d}: {winners[d]['correct']}/{winners[d]['num_samples']} = {winners[d]['accuracy']:.6f}%" for d in ALL)+"\n",encoding="utf-8")
    atomic(state,{"status":"complete","latest_run":str(latest),"active_after_last_round":active,"updated_at":time.time()});return 0
if __name__=="__main__": raise SystemExit(main())
