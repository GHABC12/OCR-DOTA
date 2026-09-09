"""Stable probability arithmetic with exact identity shortcuts."""
import torch

def compute_ocr_posterior(gaussian_logits, compatibility, clean_mass, alpha=.25, lambda_max=.25, prediction_power=1., mode='gated_log_prior'):
    if alpha < 0 or not 0 <= lambda_max <= 1 or prediction_power < 0:
        raise ValueError('invalid posterior parameters')
    if mode not in ('gated_log_prior', 'probability_blend'):
        raise ValueError(mode)
    g = gaussian_logits
    log_p = torch.log_softmax(g, -1)
    p = torch.softmax(g, -1)
    prior = alpha * compatibility.log()
    gate = lambda_max * clean_mass.pow(prediction_power)
    if alpha == 0 or lambda_max == 0 or bool((compatibility == 1).all()):
        ocr = g
        p_ocr = p
    elif mode == 'gated_log_prior':
        ocr = g + gate * prior
        p_ocr = torch.softmax(ocr, -1)
    else:
        calibrated = torch.log_softmax(g + prior, -1)
        # logaddexp avoids epsilon changing the desired mixture distribution.
        log_mix = torch.logaddexp(torch.log1p(-gate) + log_p, gate.log() + calibrated)
        ocr = g + (log_mix - log_p)
        p_ocr = log_mix.exp()
    return dict(p_dota=p, ocr_log_prior=prior, lambda_pred=gate, ocr_logits=ocr, p_ocr=p_ocr)
