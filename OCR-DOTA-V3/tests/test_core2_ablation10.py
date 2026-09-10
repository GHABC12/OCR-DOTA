"""Contract tests for the isolated core2 ablation sidecar."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_core2_ablation10", ROOT / "run_core2_ablation10.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def _anchor():
    return {
        "base": {"epsilon": 0.002, "sigma": 0.002, "eta": 0.375, "rho": 0.1},
        "rank": {"tau_rank": 0.15, "prediction_strength": 0.0, "update_power": 0.0, "eps": 1e-12},
        "update": {"beta": 0.1, "residual_strength": 0.25, "stat_decay": 0.99,
                    "init_count": 0.1, "init_mu": "constant", "prior_eps": 1e-6,
                    "norm_eps": 1e-12, "gaussian_mode": "shared_cov", "covariance_mode": "full"},
        "update_rule": "mix08clip", "update_views": "selected",
    }


class Core2ContractTests(unittest.TestCase):
    def test_dataset_aliases(self):
        self.assertEqual(MOD.normalize_dataset("Aircraft"), "fgvc")
        self.assertEqual(MOD.normalize_dataset("Cars"), "stanford_cars")
        self.assertEqual(MOD.normalize_dataset("Flower102"), "oxford_flowers")
        self.assertEqual(MOD.normalize_dataset("Flood101"), "food101")
        self.assertEqual(MOD.normalize_dataset("SUN387"), "sun397")


    def test_baseline_rank_disabled(self):
        cfg = MOD.make_config(_anchor(), prediction_strength=0.0, power=0.0, tau=0.15)
        self.assertEqual(cfg["rank"]["prediction_strength"], 0.0)
        self.assertEqual(cfg["rank"]["update_power"], 0.0)
        self.assertEqual(cfg["rank"]["tau_rank"], 0.15)


    def test_prediction_only_has_update_power_zero(self):
        cfg = MOD.make_config(_anchor(), prediction_strength=0.3, power=0.0)
        self.assertAlmostEqual(cfg["rank"]["prediction_strength"], 0.3)
        self.assertEqual(cfg["rank"]["update_power"], 0.0)


    def test_update_only_has_prediction_strength_zero(self):
        cfg = MOD.make_config(_anchor(), prediction_strength=0.0, power=1.0, tau=0.3)
        self.assertEqual(cfg["rank"]["prediction_strength"], 0.0)
        self.assertEqual(cfg["rank"]["update_power"], 1.0)
        self.assertEqual(cfg["rank"]["tau_rank"], 0.3)


    def test_corrected_regressed_identity(self):
        base = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
        candidate = np.array([1, 1, 0, 0, 2], dtype=np.uint8)
        target = np.array([1, 1, 2, 0, 0], dtype=np.uint8)
        d = MOD._diagnostics(base, candidate, target, np.linspace(0.1, 1.0, 5, dtype=np.float32))
        self.assertEqual(d["corrected"], 1)
        self.assertEqual(d["regressed"], 1)
        self.assertEqual(d["net_correction"], 0)
        self.assertEqual(d["mcnemar_b"], 1)
        self.assertEqual(d["mcnemar_c"], 1)


    def test_last50_metric_definition(self):
        target = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        pred = np.array([0, 0, 2, 0, 4], dtype=np.uint8)
        start = len(target) // 2
        self.assertEqual(start, 2)
        correct, accuracy = MOD._accuracy(pred, target, start)
        self.assertEqual(correct, 2)
        self.assertAlmostEqual(accuracy, 66.6666667, places=5)


    def test_candidate_state_cold_start_fingerprint(self):
        a = _anchor()
        c1 = MOD.make_config(a, prediction_strength=0.3, power=0.0)
        c2 = MOD.make_config(a, prediction_strength=0.3, power=1.0)
        self.assertNotEqual(MOD.candidate_fingerprint("dtd", c1, "posterior"), MOD.candidate_fingerprint("dtd", c2, "update"))
        self.assertEqual(MOD.candidate_fingerprint("dtd", c1, "posterior"), MOD.candidate_fingerprint("dtd", c1, "posterior"))


    def test_loocv_does_not_use_heldout_score(self):
        # Selection in summarize_posterior only receives rows from the other nine
        # streams; this fixture checks that a held-out row is not required to form
        # the selected candidate.
        rows = []
        for dataset in MOD.DATASETS:
            for i, strength in enumerate(MOD.POSTERIOR_STRENGTHS):
                rows.append({"phase": "posterior", "status": "ok", "dataset": dataset,
                             "candidate_id": f"P{i}", "config": {"rank": {"prediction_strength": strength}},
                             "accuracy": 50.0 + (1.0 if i == 1 and dataset != "dtd" else 0.0),
                             "base_accuracy": 50.0,
                             "correct": int(500 + (10 if i == 1 and dataset != "dtd" else 0)),
                             "num_samples": 100, "delta_correct": 10 if i == 1 and dataset != "dtd" else 0,
                             "delta_pp": 10.0 if i == 1 and dataset != "dtd" else 0.0})
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            summary = MOD.summarize_posterior(Path(td), rows, MOD.DATASETS)
        loo = {r["heldout_dataset"]: r for r in summary["loo_rows"]}
        self.assertEqual(loo["dtd"]["selected_candidate_from_other9"], "P1")
