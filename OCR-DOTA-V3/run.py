#!/usr/bin/env python3
"""Entry point for OCR-DOTA-V3 contract checks."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.print_contract:
        print(json.dumps({
            "prediction": "rank_logits = base_logits + strength * log(rank_compatibility)",
            "update": "omega = E_p[rank_compatibility]^update_power * p_geometry",
            "removed": ["semantic_leakage", "independent_order_violation"],
            "retained_geometry": "residual_magnitude",
        }, indent=2))
    if args.self_test:
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if not args.print_contract:
        parser.error("choose --self-test or --print-contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
