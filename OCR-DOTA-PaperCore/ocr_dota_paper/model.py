"""Literal legacy DOTA state plus two OCR consumers; no target argument."""
import torch
import torch.nn.functional as F
from scripts.run_cross_benchmark_ocr_ablation import LegacyState, align_prob_map
from .compatibility import compute_ocr_compatibility
from .posterior import compute_ocr_posterior
from .responsibility import compute_clean_mass, compute_ocr_responsibility

DEFAULT_CONFIG = dict(tau_rank=.15, alpha=.25, lambda_max=.25, prediction_power=1., posterior_mode='gated_log_prior', delta=1., beta_resp=1., responsibility_mode='calibrated_clip')

class PaperDOTA:
    def __init__(self, base, config, dim, classes, text_prototypes, device):
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f'unknown paper configuration: {unknown}')
        self.config = {**DEFAULT_CONFIG, **config}
        self.base = base
        self.state = LegacyState(base, dim, classes, device)
        text = text_prototypes.float()
        if text.shape == (dim, classes):
            text = text.t()
        if text.shape != (classes, dim):
            raise ValueError('invalid text shape')
        self.text = F.normalize(text, dim=-1)

    def compute_ocr_evidence(self, z):
        gaussian = self.state.scores(z, use_prior=False)
        parts = compute_ocr_compatibility(F.normalize(z, dim=-1) @ self.text.t(), gaussian, self.config['tau_rank'])
        parts['gaussian_logits'] = gaussian
        parts['p_dota'] = torch.softmax(gaussian, -1)
        parts['clean_mass'] = compute_clean_mass(parts['p_dota'], parts['compatibility'])
        return parts

    def posterior(self, gaussian, parts):
        c = self.config
        return compute_ocr_posterior(gaussian, parts['compatibility'], parts['clean_mass'], c['alpha'], c['lambda_max'], c['prediction_power'], c['posterior_mode'])

    def compute_prediction(self, clip_logits, parts, num_views, enable_posterior=False):
        posterior = self.posterior(parts['gaussian_logits'], parts)
        gaussian = posterior['ocr_logits'] if enable_posterior else parts['gaussian_logits']
        weight = torch.clamp(float(self.base['rho']) * self.state.count.mean() / num_views, max=float(self.base['eta']))
        return clip_logits + weight * gaussian, posterior, weight

    def compute_responsibility(self, views, p_zs, parts, enable_responsibility=False):
        aligned = align_prob_map(p_zs, views.size(0))
        if not enable_responsibility:
            return aligned
        c = self.config
        p_ocr = None
        if c['responsibility_mode'] == 'posterior':
            p_ocr = self.posterior(self.state.scores(views, use_prior=False), parts)['p_ocr']
        return compute_ocr_responsibility(aligned, p_ocr, parts['compatibility'], parts['clean_mass'], c['delta'], c['beta_resp'], c['responsibility_mode'])

    def update(self, views, weights):
        self.state.fit(views, weights)
        self.state.refresh_inverse()
