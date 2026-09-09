import unittest
import torch
from ocr_dota_paper.compatibility import compute_ocr_compatibility
from ocr_dota_paper.posterior import compute_ocr_posterior
from ocr_dota_paper.responsibility import compute_clean_mass,compute_ocr_responsibility

class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.g=torch.randn(4,9)
        self.p=torch.softmax(self.g,-1)
        self.c=torch.rand(1,9)*.9+.1
        self.m=compute_clean_mass(self.p,self.c)
    def test_compatibility_identity_and_range(self):
        parts=compute_ocr_compatibility(self.g,self.g)
        self.assertTrue(torch.equal(parts['compatibility'],torch.ones_like(self.g)))
        c=compute_ocr_compatibility(self.g,-self.g)['compatibility']
        self.assertTrue(bool(((c>0)&(c<=1)).all()))
    def test_clean_mass_range(self):
        self.assertTrue(bool(((self.m>0)&(self.m<=1)).all()))
    def test_posterior_off_degenerates(self):
        for mode in ('gated_log_prior','probability_blend'):
            r=compute_ocr_posterior(self.g,self.c,self.m,alpha=0,mode=mode)
            self.assertTrue(torch.equal(r['ocr_logits'],self.g));self.assertTrue(torch.equal(r['p_ocr'],self.p))
    def test_compatibility_one_posterior_degenerates(self):
        for mode in ('gated_log_prior','probability_blend'):
            r=compute_ocr_posterior(self.g,torch.ones_like(self.c),torch.ones_like(self.m),mode=mode)
            self.assertTrue(torch.equal(r['ocr_logits'],self.g))
    def test_responsibility_off_exact_dota(self):
        r=compute_ocr_responsibility(self.p,None,self.c,self.m,mode='dota')
        self.assertTrue(torch.equal(r,self.p))
    def test_responsibility_clean_limit(self):
        r=compute_ocr_responsibility(self.p,None,torch.ones_like(self.c),torch.ones_like(self.m))
        self.assertTrue(torch.equal(r,self.p))
        r=compute_ocr_responsibility(self.p,None,self.c,self.m,delta=0,beta_resp=0)
        self.assertTrue(torch.equal(r,self.p))
    def test_probability_blend_and_no_nan_inf(self):
        for mode in ('gated_log_prior','probability_blend'):
            r=compute_ocr_posterior(self.g*100,self.c,self.m,mode=mode)
            self.assertTrue(bool(torch.isfinite(r['ocr_logits']).all()))
            torch.testing.assert_close(torch.softmax(r['ocr_logits'],-1),r['p_ocr'],atol=2e-6,rtol=2e-5)
    def test_same_compatibility_used_by_prediction_and_update(self):
        from unittest.mock import patch
        from ocr_dota_paper.model import PaperDOTA
        base=dict(epsilon=.01,sigma=.01,eta=.1,rho=.01)
        model=PaperDOTA(base,{},3,4,torch.randn(4,3),'cpu')
        views=torch.randn(2,3)
        parts=model.compute_ocr_evidence(views.mean(0,keepdim=True))
        from ocr_dota_paper import model as module
        with patch.object(module,'compute_ocr_posterior',wraps=module.compute_ocr_posterior) as p:
            model.compute_prediction(torch.randn(1,4),parts,2,True)
            self.assertIs(p.call_args.args[1],parts['compatibility'])
        with patch.object(module,'compute_ocr_responsibility',wraps=module.compute_ocr_responsibility) as r:
            model.compute_responsibility(views,torch.softmax(torch.randn(2,4),-1),parts,True)
            self.assertIs(r.call_args.args[2],parts['compatibility'])

if __name__=='__main__':unittest.main()
