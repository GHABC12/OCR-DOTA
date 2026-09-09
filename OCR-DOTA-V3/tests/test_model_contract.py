from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

V3_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V3_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V3_ROOT))

from ocr_dota_v3.model import OCRDOTAV3  # noqa: E402


def config(prediction_strength: float):
    return {
        "rank": {
            "tau_rank": 0.2,
            "prediction_strength": prediction_strength,
            "update_power": 1.5,
        },
        "update": {
            "beta": 0.5,
            "residual_strength": 0.25,
            "init_count": 1.0,
            "init_mu": "text",
            "stat_decay": 1.0,
            "covariance_mode": "diag",
            "gaussian_mode": "shared_cov",
        },
    }


class ModelContractTest(unittest.TestCase):
    def setUp(self):
        self.text = torch.eye(3, dtype=torch.float32)
        self.base = {"epsilon": 1e-3, "sigma": 0.1}

    def make_model(self, strength: float):
        return OCRDOTAV3(self.base, config(strength), 3, 3, self.text, device="cpu")

    def test_removed_risks_are_rejected(self):
        bad = config(1.0)
        bad["update"]["eta_e"] = 0.1
        with self.assertRaisesRegex(ValueError, "removed risk keys"):
            OCRDOTAV3(self.base, bad, 3, 3, self.text, device="cpu")

    def test_prediction_strength_does_not_change_responsibility_or_trajectory(self):
        left, right = self.make_model(0.0), self.make_model(3.0)
        stream = [
            torch.tensor([[0.8, 0.1, 0.2]], dtype=torch.float32),
            torch.tensor([[0.2, 0.9, 0.1]], dtype=torch.float32),
            torch.tensor([[0.1, 0.2, 0.8]], dtype=torch.float32),
        ]
        for z in stream:
            a = left.posterior_and_responsibility(z, return_parts=True)
            b = right.posterior_and_responsibility(z, return_parts=True)
            torch.testing.assert_close(a["rank_compatibility"], b["rank_compatibility"])
            torch.testing.assert_close(a["omega"], b["omega"])
            left.fit_ocr(z, a["omega"])
            right.fit_ocr(z, b["omega"])
            left.update()
            right.update()
        for name in ("C", "S", "mu", "pi", "Sigma_diag"):
            torch.testing.assert_close(getattr(left, name), getattr(right, name))

    def test_prediction_prior_is_explicit(self):
        model = self.make_model(2.0)
        z = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
        parts = model.posterior_and_responsibility(z, return_parts=True)
        expected = parts["geometry_logits"] + parts["rank_log_prior"]
        torch.testing.assert_close(parts["rank_logits"], expected)

    def test_responsibility_mass_is_gated(self):
        model = self.make_model(1.0)
        z = torch.tensor([[0.4, 0.4, 0.8]], dtype=torch.float32)
        parts = model.posterior_and_responsibility(z)
        row_mass = parts["omega"].sum(-1, keepdim=True)
        torch.testing.assert_close(row_mass, parts["update_gate"])
        self.assertTrue(bool((parts["update_gate"] >= 0).all()))
        self.assertTrue(bool((parts["update_gate"] <= 1).all()))


if __name__ == "__main__":
    unittest.main()
