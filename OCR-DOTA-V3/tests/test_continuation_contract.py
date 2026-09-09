import importlib.util
import sys
import unittest
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_ROOT))
SPEC = importlib.util.spec_from_file_location("v3_continuation", V3_ROOT / "tune_nonimagenet21_continuation.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class ContinuationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = module.validate_parent(module.SOURCE)
        cls.parent = parent
        cls.jobs, cls.registry = module.build_jobs(parent)

    def test_exact_target_order_and_candidate_count(self):
        self.assertEqual(len(module.TARGETS), 8)
        self.assertEqual(sum(len(v["candidates"]) for v in self.jobs["datasets"].values()), 38)

    def test_all_candidates_are_new_and_unique(self):
        self.assertEqual(sum(self.registry["reused"].values()), 0)
        keys = []
        for dataset, block in self.jobs["datasets"].items():
            keys.extend((dataset, row["fingerprint"]) for row in block["candidates"])
        self.assertEqual(len(keys), len(set(keys)))

    def test_anchor_and_candidate_configs_are_valid(self):
        for dataset, block in self.jobs["datasets"].items():
            self.assertEqual(block["anchor_fingerprint"], self.parent["winners"][dataset]["fingerprint"])
            for row in block["candidates"]:
                module.core.validate_config(row["config"])


if __name__ == "__main__":
    unittest.main()
