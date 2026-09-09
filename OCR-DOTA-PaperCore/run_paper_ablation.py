"""Real sequential FP32 replay. Labels are accessed only after update."""
from pathlib import Path
import sys, hashlib, time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / 'OCR-DOTA-V3'))
sys.path.insert(0, str(REPO))
import tune_nonimagenet21 as core
from scripts import run_cross_benchmark_ocr_ablation as legacy
from ocr_dota_paper.model import PaperDOTA

def _array(t):
    return t.detach().contiguous().cpu().numpy()

def _margin(logits):
    v = logits.topk(min(2, logits.shape[-1]), -1).values
    return float((v[..., 0] - v[..., -1]).item())

def replay_paper(data, base, config, arm, stop, stop_interval=25, save_path=None):
    if arm not in ('base', 'posterior', 'responsibility', 'full'):
        raise ValueError(arm)
    device = str(data['features'].device)
    dim, classes = map(int, data['clip_shape'])
    model = PaperDOTA(base, config, dim, classes, data['text_prototypes'], device)
    count = data['features'].shape[0]
    trajectory = hashlib.sha256(core.canonical(base).encode())
    compat_sha = hashlib.sha256()
    trace = {k: [] for k in ('prediction','base_prediction','ocr_prediction','p_dota_top1','p_ocr_top1','rank_displacement','mean_compatibility','clean_mass','lambda_pred','base_margin','ocr_margin','update_mass','true_class_weight','wrong_class_weight','confidence','nll','brier','gate','allocation_correct')}
    started = time.time()
    with torch.no_grad():
        for i in range(count):
            if i % max(1, stop_interval) == 0 and Path(stop).exists():
                raise core.StopRequested(f'STOP at {i}/{count}')
            views = data['features'][i].to(dtype=torch.float32)
            clip_logits = data['clip_logits'][i:i+1].to(dtype=torch.float32)
            prob = data['prob_maps'][i].to(dtype=torch.float32)
            parts = model.compute_ocr_evidence(views.mean(0, keepdim=True))
            final, posterior, weight = model.compute_prediction(clip_logits, parts, len(views), arm in ('posterior','full'))
            weights = model.compute_responsibility(views, prob, parts, arm in ('responsibility','full'))
            if not all(bool(torch.isfinite(x).all()) for x in (final, weights, parts['compatibility'])):
                raise FloatingPointError(f'nonfinite at {i}')
            trajectory.update(i.to_bytes(8,'little')); trajectory.update(_array(weights).tobytes())
            compat_sha.update(i.to_bytes(8,'little')); compat_sha.update(_array(parts['compatibility']).tobytes())
            model.update(views, weights)
            # Evaluation boundary: no target has been read before prediction/update.
            target = int(data['targets'][i].item())
            base_final = clip_logits + weight * parts['gaussian_logits']
            ocr_final = clip_logits + weight * posterior['ocr_logits']
            pred = int(final.argmax(-1).item())
            p_final = torch.softmax(final, -1)
            true_w = float(weights[:, target].sum())
            total_w = float(weights.sum())
            gate = float(parts['clean_mass'].pow(model.config['delta']).item()) if arm in ('responsibility','full') and model.config['responsibility_mode'] != 'dota' else 1.
            values = dict(prediction=pred,base_prediction=int(base_final.argmax()),ocr_prediction=int(ocr_final.argmax()),p_dota_top1=int(parts['p_dota'].argmax()),p_ocr_top1=int(posterior['p_ocr'].argmax()),rank_displacement=float(parts['rank_displacement'].mean()),mean_compatibility=float(parts['compatibility'].mean()),clean_mass=float(parts['clean_mass'].item()),lambda_pred=float(posterior['lambda_pred'].item()),base_margin=_margin(base_final),ocr_margin=_margin(ocr_final),update_mass=total_w,true_class_weight=true_w,wrong_class_weight=total_w-true_w,confidence=float(p_final.max()),nll=float(-torch.log_softmax(final,-1)[0,target]),brier=float(p_final.square().sum()-2*p_final[0,target]+1),gate=gate,allocation_correct=int(weights.sum(0).argmax())==target)
            for k,v in values.items(): trace[k].append(v)
    dtype = legacy.compact_dtype(classes)
    target = _array(data['targets']).reshape(-1).astype(dtype)
    ids = np.asarray(data['sample_ids'], dtype=np.int64)
    arrays = {k:np.asarray(v) for k,v in trace.items()}
    arrays['prediction'] = arrays['prediction'].astype(dtype)
    arrays.update(target=target,sample_id=ids)
    correct = arrays['prediction'] == target
    ece = 0.
    for j in range(15):
        mask = (arrays['confidence'] >= j/15) & ((arrays['confidence'] < (j+1)/15) if j<14 else (arrays['confidence'] <= 1))
        if mask.any(): ece += mask.mean()*abs(correct[mask].mean()-arrays['confidence'][mask].mean())
    mass = float(arrays['update_mass'].sum()); true_mass=float(arrays['true_class_weight'].sum()); wrong_mass=mass-true_mass
    good = arrays['allocation_correct'].astype(bool)
    finite = all(bool(torch.isfinite(t).all()) for _,t in model.state.state_tensors())
    if not finite or float(model.state.count.min()) <= 0: raise FloatingPointError('invalid final state')
    result = dict(correct=int(correct.sum()),num_samples=int(count),accuracy=100*float(correct.mean()),late50_accuracy=100*float(correct[count//2:].mean()),nll=float(arrays['nll'].mean()),brier=float(arrays['brier'].mean()),ece=float(ece),wrm=100*wrong_mass/max(mass,1e-30),crm=100*true_mass/max(mass,1e-30),total_update_mass=mass,true_class_responsibility_mass=true_mass,wrong_class_responsibility_mass=wrong_mass,mean_update_gate=float(arrays['gate'].mean()),correct_allocation_gate=float(arrays['gate'][good].mean()) if good.any() else None,wrong_allocation_gate=float(arrays['gate'][~good].mean()) if (~good).any() else None,wrong_allocation_mass_rate=100*float(arrays['update_mass'][~good].sum())/max(mass,1e-30),prediction_sha256=core.prediction_sha(ids,target,arrays['prediction']),trajectory_sha256=trajectory.hexdigest(),state_sha256=legacy.state_sha256(model.state),compatibility_sha256=compat_sha.hexdigest(),health_status='healthy',state_health=model.state.summary(),elapsed_sec=time.time()-started)
    result['temporal_accuracy'] = [100*float(x.mean()) for x in np.array_split(correct,10) if len(x)]
    result['net_correction_vs_same_state_base'] = int((arrays['prediction']==target).sum()-(arrays['base_prediction']==target).sum())
    result['wrong_to_correct'] = int(((arrays['base_prediction']!=target)&correct).sum())
    result['correct_to_wrong'] = int(((arrays['base_prediction']==target)&~correct).sum())
    result['changed_prediction_rate'] = 100*float((arrays['prediction']!=arrays['base_prediction']).mean())
    trace_hash=hashlib.sha256()
    for key in sorted(arrays):
        a=np.ascontiguousarray(arrays[key])
        trace_hash.update(key.encode());trace_hash.update(str(a.dtype).encode());trace_hash.update(core.canonical(list(a.shape)).encode());trace_hash.update(a.tobytes())
    result['analysis_trace_sha256']=trace_hash.hexdigest()
    if save_path is not None:
        core.atomic_npz(Path(save_path), **arrays)
        result['trace_path'] = str(save_path)
        result['trace_file_sha256'] = core.sha256_file(Path(save_path))
    return result
