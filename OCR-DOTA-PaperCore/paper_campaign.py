"""Pre-registered sequential development and frozen evaluation of Paper Core."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from run_paper_ablation import replay_paper, core, legacy

ARMS = ('base', 'posterior', 'responsibility', 'full')
DEFAULT = dict(tau_rank=.15, alpha=.25, lambda_max=.25, prediction_power=1,
               posterior_mode='gated_log_prior', delta=1., beta_resp=1.,
               responsibility_mode='gate_only')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x): core.atomic_json(Path(p),x)

def validate_frozen(f,c):
    if f['identity_sha256']!=c.sig or f['config_sha256']!=digest(f['config']):
        raise RuntimeError('frozen identity/config mismatch')
    if f['code_sha256']!=digest(c.identity['code']):
        raise RuntimeError('frozen code mismatch')
    if f['base_configs']!={d:x['base'] for d,x in c.identity['datasets'].items()}:
        raise RuntimeError('frozen base mismatch')
    return f

def healthy(m):
    if m.get('health_status') not in ('healthy', 'ok'):
        raise RuntimeError('unhealthy or missing candidate health status')
    for key in ('prediction_sha256','trajectory_sha256','state_sha256','compatibility_sha256'):
        if not isinstance(m.get(key),str) or len(m[key])!=64:
            raise RuntimeError('missing SHA: '+key)
    for key in ('accuracy','late50_accuracy','nll','brier','ece','wrm','crm','total_update_mass'):
        if key not in m or not np.isfinite(m[key]):
            raise RuntimeError('nonfinite/missing metric: '+key)

def paired_trace(base,row):
    with np.load(base['trace']) as b,np.load(row['trace']) as r:
        for key in ('sample_id','target'):
            if not np.array_equal(b[key],r[key]):raise RuntimeError('paired trace mismatch: '+key)
        y=r['target'].reshape(-1);bp=b['prediction'].reshape(-1);rp=r['prediction'].reshape(-1)
        bc=bp==y;rc=rp==y;n=len(y)
        wc=int((~bc & rc).sum());cw=int((bc & ~rc).sum())
        correction=dict(wrong_to_correct=wc,correct_to_wrong=cw,net_correction=wc-cw,
            changed_prediction_rate=100*float((bp!=rp).mean()),ncr=100*(wc-cw)/n)
        windows=[]
        for i,index in enumerate(np.array_split(np.arange(n),10)):
            if not len(index):continue
            windows.append(dict(window=i+1,start=int(index[0]),end_exclusive=int(index[-1]+1),
                num_samples=len(index),accuracy=100*float(rc[index].mean()),
                base_accuracy=100*float(bc[index].mean()),delta_accuracy=100*float(rc[index].mean()-bc[index].mean())))
        return correction,windows

def diagnostic_rows(d,base,row):
    m=row['metrics'];corr,windows=paired_trace(base,row)
    shared=dict(dataset=d,arm=row['arm'],config_sha256=digest(row['config']))
    posterior=dict(shared,**corr,**{k:m[k] for k in ('accuracy','nll','brier','ece')})
    resp=dict(shared,**{k:m[k] for k in ('accuracy','wrm','crm','total_update_mass',
        'true_class_responsibility_mass','wrong_class_responsibility_mass','mean_update_gate',
        'correct_allocation_gate','wrong_allocation_gate','late50_accuracy')})
    temporal=[dict(shared,late50_accuracy=m['late50_accuracy'],**w) for w in windows]
    return posterior,resp,temporal

def equal_updates(left,right):
    for key in ('state_sha256','trajectory_sha256'):
        if left['metrics'][key]!=right['metrics'][key]:
            raise RuntimeError('cross-arm update pollution: '+left['dataset']+'/'+key)

def choose(group, n):
    return sorted(group,key=lambda d:hashlib.sha256(('paper-core-v1-dev-split|'+d).encode()).hexdigest())[:n]

def protocol():
    dev=choose(core.TEN_DATASETS,3)+choose(core.OFFICE_HOME,1)+choose(core.DOMAINNET,2)
    return dict(version='paper-core-v1-prereg-1',development=dev,
        heldout=[d for d in core.ALL_DATASETS if d not in dev],all_datasets=list(core.ALL_DATASETS),
        split_rule='SHA256(paper-core-v1-dev-split|dataset), lowest 3 classic,1 Office,2 DomainNet; VisDA held out',
        base_policy='Original configs/vit YAML frozen before development; all four arms identical base; no base tuning',
        posterior_grid=dict(alpha=[.25,.5,1.],lambda_max=[.25,.5,1.],prediction_power=[1,2]),
        posterior_modes=['gated_log_prior','probability_blend'],tau_rank=.15,
        posterior_accept='development macro and micro >= base; net correction strictly positive',
        posterior_selection='macro accuracy, total correct, lower alpha*lambda_max, deterministic id',
        responsibility=dict(modes=['gate_only','calibrated_clip','posterior'],delta=1.,beta_resp=1.,
            selection='largest macro then correct; prefer R3 over best if within .01 macro pp and nonnegative micro/macro gain'),
        negative_posterior_policy='stop after both 18-point grids if neither yields acceptable posterior; do not run or retune on heldout',
        rank='normalized ordinal ranks in [0,1], stable descending tie break',
        precision='fp32',global_seed=1,geometry=False,per_dataset_module_selection=False,
        limitations='Heldout means not used in this campaign selection. All datasets were exposed in historical research; not pristine external validation.')

def write_csv(p, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

class Campaign:
    def __init__(self,out):
        self.out=out;out.mkdir(parents=True,exist_ok=True);self.stop=out/'STOP'
        self.p=protocol();self.data=None;self.dataset=None
        sources=list(HERE.glob('*.py'))+list((HERE/'ocr_dota_paper').glob('*.py'))
        sources += [REPO/'scripts/run_cross_benchmark_ocr_ablation.py',REPO/'OCR-DOTA-V3/tune_nonimagenet21.py']
        self.identity=dict(protocol=self.p,code={str(p.relative_to(REPO)):sha(p) for p in sorted(sources)},datasets={})
        old_manifest=out/'manifest.json'
        for d in core.ALL_DATASETS:
            cp=core.cache_path(d,core.TEN_CACHE_ROOT,core.DOMAIN_CACHE_ROOT)
            self.identity['datasets'][d]=dict(cache_path=str(cp),cache_sha256=sha(cp),
                base=core.load_original_base(REPO,d),base_file_sha=sha(REPO/'configs/vit'/f'{d}.yaml'))
        self.sig=digest(self.identity)
        if old_manifest.exists() and read(old_manifest)['identity_sha256']!=self.sig:
            raise RuntimeError('manifest identity changed; refusing mixed resume')
        write(old_manifest,dict(identity=self.identity,identity_sha256=self.sig,status='running'))
        write(out/'DEVELOPMENT_PROTOCOL.json',self.p)

    def check(self):
        if self.stop.exists():raise core.StopRequested('campaign STOP')

    def status(self,phase,**kw):
        write(self.out/'state.json',dict(status='running',phase=phase,time=time.time(),**kw))

    def load(self,d):
        self.check()
        if self.dataset!=d:
            self.data=None;torch.cuda.empty_cache()
            ident=self.identity['datasets'][d]
            if sha(ident['cache_path'])!=ident['cache_sha256']:raise RuntimeError('cache changed')
            self.data,self.meta=legacy.load_cache(Path(ident['cache_path']),'cuda',None)
            self.dataset=d
            write(self.out/'data_identity'/f'{d}.json',self.meta)
        return self.data

    def run(self,phase,d,cfg,arm,verify=False):
        self.check();base=self.identity['datasets'][d]['base']
        job=digest(dict(signature=self.sig,dataset=d,base=base,config=cfg,arm=arm))
        folder=self.out/'runs'/d/job;folder.mkdir(parents=True,exist_ok=True)
        result_file=folder/'result.json';trace=folder/'trace.npz'
        self.status(phase,dataset=d,arm=arm,job=job,verification=verify)
        if result_file.exists():
            row=read(result_file)
            if row['identity_sha256']!=self.sig or row.get('job')!=job or row.get('config')!=cfg or row.get('arm')!=arm or row.get('status')!='ok' or not trace.exists() or sha(trace)!=row['trace_file_sha256']:
                raise RuntimeError('cached artifact identity mismatch')
        else:
            data=self.load(d);core.setup_seed(1)
            metrics=replay_paper(data,base,cfg,arm,self.stop,save_path=trace)
            if metrics['num_samples']!=len(data['features']):raise RuntimeError('missing samples')
            healthy(metrics)
            row=dict(identity_sha256=self.sig,job=job,dataset=d,arm=arm,config=cfg,
                metrics=metrics,trace=str(trace),trace_file_sha256=sha(trace),status='ok')
            write(result_file,row);core.append_jsonl(self.out/'candidate_results.jsonl',row)
        healthy(row['metrics'])
        if verify:
            vf=folder/'verification.json'
            if vf.exists():
                v=read(vf)
                if v['reference_result_sha256']!=sha(result_file) or v.get('identity_sha256')!=self.sig or v.get('status')!='reproduced':raise RuntimeError('verification reference changed')
            else:
                data=self.load(d);self.check();core.setup_seed(1)
                repeated=replay_paper(data,base,cfg,arm,self.stop,save_path=folder/'replay.npz')
                healthy(repeated)
                fields=['correct','num_samples','prediction_sha256','trajectory_sha256','state_sha256','compatibility_sha256','analysis_trace_sha256']
                for key in fields:
                    if key not in repeated or key not in row['metrics'] or repeated[key]!=row['metrics'][key]:raise RuntimeError(f'cold replay mismatch: {d}/{arm}/{key}')
                v=dict(status='reproduced',reference_result_sha256=sha(result_file),fields=fields,
                    identity_sha256=self.sig,dataset=d,arm=arm,job=job)
                write(vf,v);core.append_jsonl(self.out/'verification.jsonl',v)
        return row

def aggregate(rows):
    return dict(correct=sum(r['metrics']['correct'] for r in rows),
        num_samples=sum(r['metrics']['num_samples'] for r in rows),
        micro_accuracy=100*sum(r['metrics']['correct'] for r in rows)/sum(r['metrics']['num_samples'] for r in rows),
        macro_accuracy=float(np.mean([r['metrics']['accuracy'] for r in rows])))

def acceptable(a,b):return a['correct']>b['correct'] and a['macro_accuracy']>=b['macro_accuracy']

def development(c):
    locked=c.out/'FROZEN_PAPER_CONFIG.yaml'
    if locked.exists():
        f=yaml.safe_load(locked.read_text())
        return validate_frozen(f,c)
    dev=c.p['development']
    baserows=[c.run('posterior_base',d,DEFAULT,'base',True) for d in dev]
    base=aggregate(baserows);posterior_candidates=[];chosen=None;posterior_diag=[]
    base_by_dataset={r['dataset']:r for r in baserows}
    for mode in c.p['posterior_modes']:
        configs=[dict(DEFAULT,posterior_mode=mode,alpha=a,lambda_max=l,prediction_power=q)
            for a,l,q in itertools.product([.25,.5,1.],[.25,.5,1.],[1,2])]
        # Dataset-first keeps each cache resident while all independent candidates cold-start.
        allrows={digest(cfg):[] for cfg in configs}
        for d in dev:
            for cfg in configs:
                row=c.run('posterior_search',d,cfg,'posterior')
                equal_updates(base_by_dataset[d],row)
                allrows[digest(cfg)].append(row)
                pdiag,_,_=diagnostic_rows(d,base_by_dataset[d],row)
                posterior_diag.append(pdiag)
        scores=[dict(config=cfg,aggregate=aggregate(allrows[digest(cfg)])) for cfg in configs]
        posterior_candidates.extend(scores)
        write(c.out/'posterior_search.json',dict(base=base,candidates=posterior_candidates))
        write_csv(c.out/'posterior_development_correction.csv',posterior_diag)
        good=[s for s in scores if acceptable(s['aggregate'],base)]
        if good:
            good.sort(key=lambda s:(-s['aggregate']['macro_accuracy'],-s['aggregate']['correct'],
                s['config']['alpha']*s['config']['lambda_max'],digest(s['config'])))
            chosen=good[0]['config'];break
    if chosen is None:
        write(c.out/'state.json',dict(status='research_negative',phase='posterior',
            reason='Neither pre-registered posterior family improved both development aggregate metrics. Heldout remains unused.'))
        (c.out/'research_negative.md').write_text(
            '# Posterior development result\n\nBoth pre-registered 18-point searches failed the acceptance criterion. '
            'No held-out labels were used for selection. Base was not redefined. '
            'Responsibility and formal 2x2 phases are not complete. See posterior_search.json.\n',encoding='utf-8')
        return None
    for d in dev:c.run('posterior_verify',d,chosen,'posterior',True)
    modes=['gate_only','calibrated_clip','posterior'];res=[];responsibility_diag=[]
    for d in dev:
        _,rdiag,_=diagnostic_rows(d,base_by_dataset[d],base_by_dataset[d]);responsibility_diag.append(rdiag)
    for mode in modes:
        cfg=dict(chosen,responsibility_mode=mode)
        rows=[c.run('responsibility_development',d,cfg,'responsibility',True) for d in dev]
        for row in rows:
            _,rdiag,_=diagnostic_rows(row['dataset'],base_by_dataset[row['dataset']],row)
            responsibility_diag.append(rdiag)
        res.append(dict(config=cfg,aggregate=aggregate(rows)))
    res.sort(key=lambda s:(-s['aggregate']['macro_accuracy'],-s['aggregate']['correct']))
    winner=res[0];r3=next(s for s in res if s['config']['responsibility_mode']=='posterior')
    if r3['aggregate']['macro_accuracy']>=winner['aggregate']['macro_accuracy']-.01 and acceptable(r3['aggregate'],base):winner=r3
    write(c.out/'responsibility_development.json',dict(base=base,candidates=res,selected=winner))
    write_csv(c.out/'responsibility_development_analysis.csv',responsibility_diag)
    frozen=dict(identity_sha256=c.sig,protocol=c.p,config=winner['config'],base_configs={d:x['base'] for d,x in c.identity['datasets'].items()},
        code_sha256=digest(c.identity['code']),config_sha256=digest(winner['config']),frozen_at=time.time())
    tmp=locked.with_suffix('.tmp');tmp.write_text(yaml.safe_dump(frozen,sort_keys=True),encoding='utf-8');os.replace(tmp,locked)
    return frozen

def export(c, allrows, frozen):
    out=c.out/'results';out.mkdir(exist_ok=True)
    per=[];posterior=[];resp=[];temporal=[]
    for d,rows in allrows.items():
        ms={a:r['metrics'] for a,r in rows.items()}
        r=dict(dataset=d,split='development' if d in c.p['development'] else 'heldout',num_samples=ms['base']['num_samples'])
        for a in ARMS:r[a]=ms[a]['accuracy'];r[a+'_correct']=ms[a]['correct']
        for a in ARMS[1:]:r['delta_'+a]=r[a]-r['base']
        r['interaction']=r['full']-r['posterior']-r['responsibility']+r['base'];per.append(r)
        for a in ARMS:
            pd,rd,td=diagnostic_rows(d,rows['base'],rows[a])
            posterior.append(pd);resp.append(rd);temporal.extend(td)
    groups={}
    for name,ds in [('all',list(allrows)),('development',c.p['development']),('heldout',c.p['heldout'])]:
        groups[name]={}
        for a in ARMS:
            rs=[allrows[d][a] for d in ds];v=aggregate(rs);v['micro_accuracy']=100*v['correct']/v['num_samples']
            late=[r['metrics'].get('late50_accuracy') for r in rs]
            v['late50_macro']=float(np.mean(late)) if all(x is not None for x in late) else None
            groups[name][a]=v
    contrasts={}
    for name,arms in groups.items():
        contrasts[name]={}
        for metric in ('correct','micro_accuracy','macro_accuracy','late50_macro'):
            b=arms['base'][metric];p=arms['posterior'][metric];u=arms['responsibility'][metric];f=arms['full'][metric]
            contrasts[name][metric]=dict(delta_posterior=p-b,delta_responsibility=u-b,delta_full=f-b,interaction=f-p-u+b)
    write(out/'paper_2x2_summary.json',dict(status='complete',groups=groups,contrasts=contrasts,per_dataset=per,frozen=frozen))
    write_csv(out/'per_dataset.csv',per);write_csv(out/'paper_2x2_summary.csv',[dict(group=g,arm=a,**v) for g,arms in groups.items() for a,v in arms.items()])
    write_csv(out/'posterior_correction.csv',posterior);write_csv(out/'responsibility_analysis.csv',resp);write_csv(out/'temporal_analysis.csv',temporal)
    verifications=[]
    for rows in allrows.values():
        for row in rows.values():
            vf=Path(row['trace']).parent/'verification.json';v=read(vf)
            if v['status']!='reproduced' or v['identity_sha256']!=c.sig:raise RuntimeError('missing final verification')
            verifications.append(dict(v,result_sha256=sha(vf.parent/'result.json'),
                metrics={k:row['metrics'][k] for k in ('prediction_sha256','trajectory_sha256','state_sha256','compatibility_sha256','analysis_trace_sha256')}))
    write(out/'verification.json',dict(status='reproduced',datasets=len(allrows),arms=len(ARMS),
        identity_sha256=c.sig,config_sha256=frozen['config_sha256'],code_sha256=frozen['code_sha256'],runs=verifications))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(HERE/'results/paper_core_v1_20260909'));p.add_argument('--phase',choices=['development','evaluate','all'],default='all');args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    try:
        with core.PidLock(out/'RUNNING.pid'):
            c=Campaign(out)
            # Exact-reproduction validation must be completed by independent phase0/1 runner first.
            gate=HERE/'results/exact_validation.json'
            if not gate.exists() or read(gate).get('status')!='passed':raise RuntimeError('Phase 1 exact validation has not passed')
            gate_data=read(gate);gate_code=gate_data.get('code_sha256')
            if not isinstance(gate_code,dict) or not gate_code:raise RuntimeError('exact gate lacks bound code hashes')
            for name,value in gate_code.items():
                path=Path(name)
                if not path.is_absolute():
                    path=(REPO/path) if (REPO/path).is_file() else (HERE/path)
                if not path.is_file() or sha(path)!=value:raise RuntimeError('exact gate code changed: '+name)
            if args.phase=='evaluate':
                f=yaml.safe_load((out/'FROZEN_PAPER_CONFIG.yaml').read_text())
                validate_frozen(f,c)
            else:f=development(c)
            if f is None:return 2
            if args.phase=='development':return 0
            rows={}
            for d in core.ALL_DATASETS:
                rows[d]={a:c.run('formal_2x2',d,f['config'],a,True) for a in ARMS}
                for a,b in [('base','posterior'),('responsibility','full')]:
                    equal_updates(rows[d][a],rows[d][b])
            export(c,rows,f)
            write(out/'state.json',dict(status='complete',datasets=21,arms=4,finished_at=time.time()))
            manifest=read(out/'manifest.json');manifest['status']='complete';write(out/'manifest.json',manifest)
    except Exception as e:
        write(out/'state.json',dict(status='stopped' if isinstance(e,core.StopRequested) else 'failed',error=str(e),time=time.time()))
        raise
    return 0

if __name__=='__main__':raise SystemExit(main())
