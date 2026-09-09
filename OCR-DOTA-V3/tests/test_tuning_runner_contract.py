import importlib.util
import sys
import unittest
from pathlib import Path

import torch


V3_ROOT = Path(__file__).resolve().parents[1]
RUNNER = V3_ROOT / "tune_nonimagenet21.py"
SPEC = importlib.util.spec_from_file_location("v3_tuning_runner", RUNNER)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class TuningRunnerContractTest(unittest.TestCase):
    def setUp(self):
        self.anchor = {
            "base": {"epsilon": 1e-4, "sigma": 0.002, "eta": 0.3, "rho": 0.02},
            "rank": {"tau_rank": 0.15, "prediction_strength": 1.0, "update_power": 1.0, "eps": 1e-12},
            "update": {
                "beta": 0.5, "residual_strength": 0.25, "stat_decay": 1.0,
                "init_count": 1.0, "init_mu": "constant", "prior_eps": 1e-6,
                "norm_eps": 1e-12, "gaussian_mode": "shared_cov", "covariance_mode": "full",
            },
            "update_rule": "mix08clip", "update_views": "selected",
        }

    def test_aliases(self):
        self.assertEqual(module.normalize_dataset("Aircraft"), "fgvc")
        self.assertEqual(module.normalize_dataset("Flood101"), "food101")
        self.assertEqual(module.normalize_dataset("SUN387"), "sun397")

    def test_initial_candidates_are_preregistered_a0_to_a7(self):
        rows = module.build_initial_candidates(self.anchor)
        observed = [(r["candidate_id"], r["config"]["rank"]["tau_rank"],
                     r["config"]["rank"]["prediction_strength"],
                     r["config"]["rank"]["update_power"]) for r in rows]
        self.assertEqual(observed, [
            ("A0", .15, 0., 0.), ("A1", .15, .5, 0.),
            ("A2", .15, 1., 0.), ("A3", .15, 2., 0.),
            ("A4", .075, .5, 1.), ("A5", .15, 1., 1.),
            ("A6", .30, 2., 1.), ("A7", .15, 1., 2.),
        ])
        self.assertEqual(len(module.generate_stage("dtd", "initial", {"config": self.anchor})), 8)

    def test_round_candidate_counts_and_count_direction(self):
        self.assertEqual(len(module.generate_stage("dtd", "round2", {"config": self.anchor})), 6)
        qd = module.generate_stage("domainnet_quickdraw", "round2", {"config": self.anchor})
        count = next(row for row in qd if "init_count" in row["label"])
        self.assertEqual(count["config"]["update"]["init_count"], .75)
        self.assertEqual(len(module.generate_stage("dtd", "round3", {"config": self.anchor})), 6)

    def test_shared_gate_is_applied_after_every_allocation_rule(self):
        prob = torch.tensor([[.7, .3]])
        geometry = torch.tensor([[.2, .8]])
        gate = torch.tensor([[.25]])
        for rule in module.UPDATE_RULES:
            allocation = module.allocation_by_rule(rule, prob, geometry)
            weights = gate * allocation
            self.assertTrue(torch.allclose(weights.sum(-1), gate.squeeze(-1)))

    def test_removed_eta_d_is_not_migrated(self):
        row = {
            "base": self.anchor["base"],
            "ocr": {**self.anchor["update"], "eta_d": 9.0, "update_views": "selected"},
            "update_rule": "mix08clip",
        }
        migrated = module.migrate_anchor("dtd", row)
        self.assertNotIn("eta_d", migrated["update"])
        self.assertEqual(migrated["update"]["residual_strength"], .25)


if __name__ == "__main__":
    unittest.main()
