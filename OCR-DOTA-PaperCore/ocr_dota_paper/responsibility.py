import torch

def compute_clean_mass(p_dota, compatibility):
    if bool((compatibility == 1).all()):
        return torch.ones_like(p_dota[..., :1])
    return (p_dota * compatibility).sum(-1, keepdim=True).clamp(min=torch.finfo(p_dota.dtype).tiny, max=1.)

def compute_ocr_responsibility(p_zs, p_ocr, compatibility, clean_mass, delta=1., beta_resp=1., mode='calibrated_clip'):
    if delta < 0 or beta_resp < 0:
        raise ValueError('negative responsibility parameter')
    if mode == 'dota':
        return p_zs
    gate = clean_mass.pow(delta)
    if mode == 'gate_only':
        return gate * p_zs
    if mode == 'posterior':
        return gate * p_ocr
    if mode != 'calibrated_clip':
        raise ValueError(mode)
    if beta_resp == 0 or bool((compatibility == 1).all()):
        allocation = p_zs
    else:
        unnormalized = p_zs * compatibility.pow(beta_resp)
        allocation = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(torch.finfo(p_zs.dtype).tiny)
    return gate * allocation
