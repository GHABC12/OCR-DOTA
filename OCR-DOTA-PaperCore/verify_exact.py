"""Phase 1 exact legacy contracts at 32, 500 and full DTD."""
from pathlib import Path
import json
from run_paper_ablation import core,legacy,replay_paper

def main():
    root=Path(__file__).resolve().parent
    out=root/'results'/'exact_contracts';out.mkdir(parents=True,exist_ok=True)
    base=core.load_original_base(root.parent,'dtd')
    path=core.cache_path('dtd',core.TEN_CACHE_ROOT,core.DOMAIN_CACHE_ROOT)
    records=[]
    for n in (32,500,None):
        core.setup_seed(1);data,meta=legacy.load_cache(path,'cuda',n)
        original=core.replay_dota(data,base,out/'STOP',25)
        b=replay_paper(data,base,{},'base',out/'STOP')
        p=replay_paper(data,base,{},'posterior',out/'STOP')
        for k in ('prediction_sha256','trajectory_sha256','state_sha256'):
            assert original[k]==b[k],(n,k)
        for k in ('trajectory_sha256','state_sha256','compatibility_sha256'):
            assert b[k]==p[k],(n,k)
        u=replay_paper(data,base,{},'responsibility',out/'STOP',save_path=out/f'u_{n}.npz')
        full=replay_paper(data,base,{},'full',out/'STOP')
        for k in ('trajectory_sha256','state_sha256','compatibility_sha256'):
            assert u[k]==full[k],(n,k)
        again=replay_paper(data,base,{},'full',out/'STOP')
        core.assert_reproduced(full,again,'full')
        assert u['trajectory_sha256'] != b['trajectory_sha256']
        import numpy as np
        trace=np.load(out/f'u_{n}.npz')
        assert trace['prediction'][0]==trace['base_prediction'][0]
        records.append(dict(n=n,cache=meta,original=original,base=b,posterior=p,responsibility=u,full=full,status='passed'))
        core.atomic_json(out/'verification.json',records)
        print('PASS',n,flush=True)
    code_paths=[root/'run_paper_ablation.py',root/'verify_exact.py',*sorted((root/'ocr_dota_paper').glob('*.py')),root.parent/'scripts/run_cross_benchmark_ocr_ablation.py',root.parent/'OCR-DOTA-V3/tune_nonimagenet21.py']
    core.atomic_json(root/'results/exact_validation.json',dict(status='passed',base=base,records=records,code_sha256={str(p):core.sha256_file(p) for p in code_paths}))

if __name__=='__main__':main()
