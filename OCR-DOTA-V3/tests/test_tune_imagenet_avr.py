from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
RUNNER = V3_ROOT / "tune_imagenet_avr.py"
spec = importlib.util.spec_from_file_location("v3_avr_runner", RUNNER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def anchor():
    return {
        "base": {"epsilon": 1e-4, "sigma": .002, "eta": .5, "rho": .02},
        "rank": {"tau_rank": .15, "prediction_strength": 1., "update_power": 1., "eps": 1e-12},
        "update": {"beta": .4, "residual_strength": .25, "stat_decay": 1., "init_count": 2.,
                   "init_mu": "constant", "prior_eps": 1e-6, "norm_eps": 1e-12,
                   "gaussian_mode": "shared_cov", "covariance_mode": "full"},
        "update_rule": "mix08clip", "update_views": "selected",
    }


class AVRRunnerContractTest(unittest.TestCase):
    def test_registry_is_locked(self):
        self.assertEqual(module.DATASETS, ("A", "R", "V"))
        self.assertEqual(module.REGISTRY["A"]["sample_order_seed"], 152)
        self.assertIsNone(module.REGISTRY["R"]["sample_order_seed"])
        self.assertEqual(module.REGISTRY["V"]["sample_order_seed"], 20)
        for dataset in module.DATASETS:
            spec = module.REGISTRY[dataset]
            order = module.permutation_for(spec["num_samples"], spec["sample_order_seed"])
            self.assertEqual(module.int64_sha(order), spec["sample_order_sha256"])

    def test_initial_has_locked_A0_A7_plus_G0(self):
        rows = module.generate_stage("A", "initial", {"config": anchor()})
        self.assertEqual([row["candidate_id"] for row in rows], [f"A{i}" for i in range(8)] + ["G0"])
        g0 = rows[-1]["config"]
        self.assertEqual(g0["rank"]["prediction_strength"], 0.)
        self.assertEqual(g0["rank"]["update_power"], 0.)
        self.assertEqual(g0["update"]["residual_strength"], 0.)

    def test_stage_candidate_counts_and_values(self):
        current = {"config": anchor()}
        for dataset in module.DATASETS:
            stage2 = module.generate_stage(dataset, "stage2", current)
            stage3 = module.generate_stage(dataset, "stage3", current)
            self.assertEqual(len(stage2), 6)
            self.assertEqual(len(stage3), 4)
            self.assertEqual({row["config"]["update"]["residual_strength"] for row in stage2[:2]}, {0., .5})
            self.assertEqual({row["label"] for row in stage3}, {"S3_eta_x0.9", "S3_eta_x1.1", "S3_rho_x0.9", "S3_rho_x1.1"})

    def test_full_order_is_generated_before_smoke_prefix(self):
        full = module.permutation_for(100, 152)
        prefix = full[:32]
        wrong = module.permutation_for(32, 152)
        self.assertFalse(bool((prefix == wrong).all()))

    def test_identity_locks_budget_precision_and_order(self):
        result = module.build_identity(
            datasets=("A",), stages=("initial",), global_seed=1, max_samples=32,
            time_budget_hours=6., verification_reserve_hours=1.,
            cache_records={"A": {"sample_order_seed": 152, "permutation_sha256": "x"}},
            anchors={"A": anchor()}, provenance={"A": {"identity": "x"}},
            original_bases={"A": anchor()["base"]}, source_shas={"s": "x"}, code_shas={"c": "x"},
        )
        identity = result["identity"]
        self.assertEqual(identity["precision"], "fp32")
        self.assertEqual(identity["worker_count"], 1)
        self.assertEqual(identity["time_budget_hours"], 6.)
        self.assertFalse(identity["formal"])


if __name__ == "__main__":
    unittest.main()
