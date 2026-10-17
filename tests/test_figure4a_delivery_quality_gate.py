from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_figure4a_delivery_quality_gate as quality_gate


class Figure4aDeliveryQualityGateTests(unittest.TestCase):
    def test_curated_handoff_surface_has_no_stale_or_garbled_claims(self) -> None:
        rows = quality_gate.build_rows()
        failures = rows[rows["status"] != "pass"]

        self.assertTrue(failures.empty, failures.to_string(index=False))

    def test_primary_gap_gate_uses_latest_candidate(self) -> None:
        metrics = quality_gate.final_metrics()

        self.assertGreaterEqual(metrics["gap"], 0.020)
        self.assertGreater(metrics["paper_share"], 79.0)


if __name__ == "__main__":
    unittest.main()
