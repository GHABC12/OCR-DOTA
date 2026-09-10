"""DTD offline-vs-online identity check for the two V2 posterior modes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent; V1=HERE.parent; REPO=V1.parent
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'OCR-DOTA-V3'));sys.path.insert(0,str(V1));sys.path.insert(0,str(HERE))
import tune_nonimagenet21 as core
from scripts import run_cross_benchmark_ocr_ablation as legacy
from ocr_dota_paper.compatibility import compute_ocr_compatibility
from ocr_dota_paper.responsibility_v2 import compute_clean_mass_v2
from ocr_dota_paper.model import PaperDOTA
from evaluate_posterior_offline import evaluate_candidate, load_evidence
from ocr_dota_paper.posterior_v2 import compute_ocr_posterior_v2
from protocol_v2 import sha256_bytes
import yaml
def cache_path(d): return REPO / "log/all_dataset_perf/cache" / f"{d}_vitb16.pt"
def load_base(d):
    raw=yaml.safe_load((REPO/"configs/vit"/f"{d}.yaml").read_text()) or {}
    return {k:float(raw[k]) for k in ("epsilon","sigma","eta","rho")}
setup_seed=core.setup_seed

def online(data,base,cfg):
 dim,k=map(int,data['clip_shape']); state=legacy.LegacyState(base,dim,k,'cuda'); text=data['text_prototypes'].float();
 if tuple(text.shape)==(dim,k):text=text.t()
 text=torch.nn.functional.normalize(text,dim=-1); n=len(data['features']); ids=np.asarray(data['sample_ids'],dtype=np.int64); target=np.asarray(data['targets'].detach().cpu().numpy().reshape(-1),dtype=legacy.compact_dtype(k)); pred=np.empty(n,dtype=legacy.compact_dtype(k)); nll=[];brier=[]
 with torch.no_grad():
  for i in range(n):
   views=data['features'][i].float();clip=data['clip_logits'][i:i+1].float();prob=data['prob_maps'][i].float();z=views.mean(0,keepdim=True);g=state.scores(z,use_prior=False);stable=torch.nn.functional.normalize(z,dim=-1)@text.t();e=compute_ocr_compatibility(stable,g,.15);p=compute_clean_mass_v2(torch.softmax(g,-1),e['compatibility']);w=torch.clamp(float(base['rho'])*state.count.mean()/len(views),max=float(base['eta']));o=compute_ocr_posterior_v2(g,e['compatibility'],cfg['gamma'],cfg['mode'],cfg.get('entropy_power',1.));final=clip+w*o['ocr_logits']; weights=legacy.align_prob_map(prob,len(views));state.fit(views,weights);state.refresh_inverse();y=int(target[i]); pred[i]=int(final.argmax(-1));pf=torch.softmax(final,-1);nll.append(float(-torch.log(pf[0,y])));brier.append(float(pf.square().sum()-2*pf[0,y]+1))
 return {'prediction':pred,'target':target,'ids':ids,'correct':int((pred==target).sum()),'nll':float(np.mean(nll)),'brier':float(np.mean(brier)),'wrong_to_correct':None,'correct_to_wrong':None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence-root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args(); root=Path(a.evidence_root); out=Path(a.output); ev=load_evidence(root/'dtd.npz'); data,_=legacy.load_cache(cache_path('dtd'),'cuda',None);base=load_base('dtd'); results=[]
 for cfg in ({'mode':'direct_prior','gamma':.1,'entropy_power':1.},{'mode':'entropy_gated_prior','gamma':.1,'entropy_power':1.}):
  off,_,_=evaluate_candidate(ev,cfg); on=online(data,base,cfg); b=ev['base_prediction'].astype(np.int64); y=ev['target'].astype(np.int64); on['wrong_to_correct']=int(((b!=y)&(on['prediction']==y)).sum());on['correct_to_wrong']=int(((b==y)&(on['prediction']!=y)).sum()); equal={'prediction_sha256':core.prediction_sha(on['ids'],on['target'],on['prediction']),'offline_correct':off['correct'],'correct':on['correct'],'nll_abs_diff':abs(on['nll']-off['nll']),'brier_abs_diff':abs(on['brier']-off['brier']),'wrong_to_correct':on['wrong_to_correct'],'correct_to_wrong':on['correct_to_wrong']}; results.append({'config':cfg,'offline':off,'online':{k:v for k,v in on.items() if k not in ('prediction','target','ids')},'checks':equal})
 write={'status':'passed' if all(r['checks']['correct']==r['offline']['correct'] and r['checks']['nll_abs_diff']<=1e-6 and r['checks']['brier_abs_diff']<=1e-6 for r in results) else 'failed','dataset':'dtd','records':results}; out.mkdir(parents=True,exist_ok=True);(out/'offline_online_validation.json').write_text(json.dumps(write,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(write,indent=2))
if __name__=='__main__':main()
