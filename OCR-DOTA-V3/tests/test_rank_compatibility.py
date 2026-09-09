from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

V3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_ROOT))

from ocr_dota_v3.rank_compatibility import (  # noqa: E402
    compute_rank_compatibility,
    prediction_rank_prior,
    sample_update_compatibility,
    use_rank_compatibility,
)


class RankCompatibilityTest(unittest.TestCase):
    def test_identical_orders_are_fully_compatible(self):
        stable = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        dynamic = torch.tensor([[40.0, 30.0, 20.0, 10.0]])
        comp = compute_rank_compatibility(stable, dynamic, tau_rank=0.2)
        torch.testing.assert_close(comp, torch.ones_like(comp))

    def test_reverse_order_is_less_compatible(self):
        stable = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        aligned = compute_rank_compatibility(stable, stable, tau_rank=0.2)
        reversed_comp = compute_rank_compatibility(stable, stable.flip(-1), tau_rank=0.2)
        self.assertLess(float(reversed_comp.mean()), float(aligned.mean()))
        self.assertLess(float(reversed_comp[0, 0]), 0.05)

    def test_prediction_prior_corrects_dynamic_order(self):
        base = torch.tensor([[0.0, 0.1, 0.2]])
        comp = torch.tensor([[1.0, 0.5, 0.1]])
        corrected, prior = prediction_rank_prior(base, comp, prediction_strength=1.0)
        self.assertEqual(int(corrected.argmax(-1)), 0)
        torch.testing.assert_close(corrected, base + prior)

    def test_same_compatibility_controls_update_gate(self):
        logits = torch.tensor([[3.0, 1.0, 0.0]])
        comp = torch.tensor([[0.2, 1.0, 1.0]])
        posterior = F.softmax(logits, dim=-1)
        expected = (posterior * comp).sum(-1, keepdim=True)
        actual = sample_update_compatibility(posterior, comp)
        torch.testing.assert_close(actual, expected)
        self.assertLess(float(actual), 0.5)

    def test_two_consumers_share_tensor(self):
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        comp = torch.tensor([[1.0, 0.7, 0.2]])
        rank_logits, _, gate, parts = use_rank_compatibility(
            logits,
            comp,
            prediction_strength=0.5,
            update_power=2.0,
            return_parts=True,
        )
        torch.testing.assert_close(parts["compatibility"], comp)
        torch.testing.assert_close(rank_logits, logits + 0.5 * torch.log(comp))
        expected_gate = ((F.softmax(logits, -1) * comp).sum(-1, keepdim=True)).pow(2.0)
        torch.testing.assert_close(gate, expected_gate)


if __name__ == "__main__":
    unittest.main()
