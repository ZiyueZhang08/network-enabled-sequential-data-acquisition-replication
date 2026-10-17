from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_inverse_reconstruction_v2_joint_leaderboard as joint
import run_inverse_reconstruction_v2_linucb_candidate_confirm as linucb
import run_inverse_reconstruction_v2_mask_screening as masks


class InverseReconstructionV2MaskLinucbJointTests(unittest.TestCase):
    def test_mask_score_uses_requested_curve_errors(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "variant": method,
                    "seed": 0,
                    "cycle": cycle,
                    "standard_mask_rmse": value,
                    "reward_tail_proxy_rmse": value,
                    "cumulative_reward": reward,
                }
                for method, value, reward in [
                    (masks.DU_METHOD, 1.0, 10.0),
                    ("linucb_concat", 1.2, 12.0),
                ]
                for cycle in [0, 1]
            ]
        )
        targets = pd.DataFrame(
            [
                {"figure": "Figure 4a", "method": "DU-Net", "cycle": cycle, "paper_value": 1.0}
                for cycle in [0, 1]
            ]
            + [
                {"figure": "Figure 4a", "method": "Linear UCB", "cycle": cycle, "paper_value": 1.2}
                for cycle in [0, 1]
            ]
            + [
                {"figure": "Figure 4b", "method": "DU-Net", "cycle": cycle, "paper_value": 10.0}
                for cycle in [0, 1]
            ]
            + [
                {"figure": "Figure 4b", "method": "Linear UCB", "cycle": cycle, "paper_value": 12.0}
                for cycle in [0, 1]
            ]
        )

        row = masks.score_one_candidate("mask", "linucb_concat", metrics, targets, protocol_penalty=0.0)

        self.assertAlmostEqual(row["figure4a_du_curve_error"], 0.0)
        self.assertAlmostEqual(row["figure4a_linear_curve_error"], 0.0)
        self.assertAlmostEqual(row["figure4b_du_reward_error"], 0.0)
        self.assertAlmostEqual(row["figure4b_linear_reward_error"], 0.0)

    def test_linucb_candidate_label_is_stable(self) -> None:
        self.assertEqual(linucb.candidate_label(0.1, 10.0), "linucb_mf_alpha_0p1_reg_10")

    def test_joint_claim_status_flags_unpromoted_metric(self) -> None:
        row = {
            "metric_status": "metric_gate_failed",
            "mask_id": "u1_fallback",
            "mask_source": "current_u1_fallback",
            "linear_source": "base_20seed",
            "evidence_source": "u1_fallback_metrics",
            "du_net": joint.TAU1,
            "du_seed_count": 20,
            "linear_seed_count": 20,
            "figure4a_du_curve_error": 0.01,
            "figure4a_linear_curve_error": 0.01,
            "figure4b_du_curve_error_norm": 0.01,
            "figure4b_linear_curve_error_norm": 0.01,
            "total_error": 0.01,
        }

        self.assertIn("metric_gate_not_promoted", joint.claim_status(row))

    def test_joint_metric_mapping_carries_reward_tail(self) -> None:
        self.assertEqual(joint.metric_to_column("reward_tail_proxy"), "reward_tail_proxy_rmse")


if __name__ == "__main__":
    unittest.main()
