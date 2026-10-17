from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_inverse_reconstruction_v2_metric_curve_audit as inverse_v2


class InverseReconstructionV2MetricCurveAuditTests(unittest.TestCase):
    def test_build_metric_long_reads_gzip_test_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original_dir = inverse_v2.ROW_LEVEL_DIR
            inverse_v2.ROW_LEVEL_DIR = Path(tmp)
            try:
                rows = pd.DataFrame(
                    [
                        {
                            "selected_or_test": "test",
                            "method": inverse_v2.MAINLINE_DU,
                            "seed": 0,
                            "cycle": 0,
                            "y_true": 5.0,
                            "y_pred_clipped": 4.0,
                            "err_clipped": -1.0,
                            "err_raw": -1.0,
                            "reward": 1.0,
                            "item_popularity": 10.0,
                        },
                        {
                            "selected_or_test": "selected",
                            "method": inverse_v2.MAINLINE_DU,
                            "seed": 0,
                            "cycle": 0,
                            "y_true": 1.0,
                            "y_pred_clipped": 5.0,
                            "err_clipped": 4.0,
                            "err_raw": 4.0,
                            "reward": 0.0,
                            "item_popularity": 1.0,
                        },
                    ]
                )
                rows.to_csv(Path(tmp) / "du__seed_00.csv.gz", index=False, compression="gzip")

                metric_long = inverse_v2.build_metric_long()
            finally:
                inverse_v2.ROW_LEVEL_DIR = original_dir

        standard = metric_long[metric_long["metric"] == "standard_rmse"].iloc[0]
        self.assertEqual(standard["rows"], 1)
        self.assertEqual(standard["method"], inverse_v2.MAINLINE_DU)
        self.assertAlmostEqual(float(standard["value"]), 1.0)

    def test_completeness_marks_partial_and_missing_methods(self) -> None:
        metric_long = pd.DataFrame(
            [
                {
                    "method": inverse_v2.MAINLINE_DU,
                    "seed": 0,
                    "cycle": 0,
                    "metric": "standard_rmse",
                    "value": 1.0,
                    "rows": 2,
                }
            ]
        )

        completion = inverse_v2.completeness(metric_long)

        mainline = completion[completion["method"] == inverse_v2.MAINLINE_DU].iloc[0]
        linear = completion[completion["method"] == "linucb_concat"].iloc[0]
        self.assertEqual(mainline["status"], "partial")
        self.assertEqual(linear["status"], "missing")
        self.assertIn("20", str(mainline["missing_cycles"]))

    def test_align_metric_promotes_exact_synthetic_curve(self) -> None:
        cycles = list(range(21))
        curves = pd.DataFrame(
            [
                {
                    "method": inverse_v2.MAINLINE_DU,
                    "metric": "standard_rmse",
                    "cycle": cycle,
                    "local_value": 0.8 - cycle * 0.01,
                    "seed_count": 20,
                }
                for cycle in cycles
            ]
            + [
                {
                    "method": "linucb_concat",
                    "metric": "standard_rmse",
                    "cycle": cycle,
                    "local_value": 0.9 - cycle * 0.005,
                    "seed_count": 20,
                }
                for cycle in cycles
            ]
        )
        targets = pd.DataFrame(
            [
                {
                    "figure": "Figure 4a",
                    "method": "DU-Net",
                    "cycle": cycle,
                    "paper_value": 0.8 - cycle * 0.01,
                }
                for cycle in cycles
            ]
            + [
                {
                    "figure": "Figure 4a",
                    "method": "Linear UCB",
                    "cycle": cycle,
                    "paper_value": 0.9 - cycle * 0.005,
                }
                for cycle in cycles
            ]
        )

        alignment = inverse_v2.align_metric(curves, targets)

        promoted = alignment[alignment["paper_wrmse_reconstructed_candidate"] == "yes"]
        self.assertFalse(promoted.empty)
        self.assertEqual(promoted.iloc[0]["metric"], "standard_rmse")


if __name__ == "__main__":
    unittest.main()
