from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_inverse_reconstruction_leaderboard as inverse


class InverseReconstructionLeaderboardTests(unittest.TestCase):
    def test_builder_outputs_candidate_surface_without_exact_claim(self) -> None:
        outputs = inverse.build_all(mask_count=50, mask_seed=20260712, force_masks=False)

        metric_top20 = outputs["metric_top20"]
        metric_endpoint_top20 = outputs["metric_endpoint_family_top20"]
        linucb = outputs["linucb_candidate_ranking"]
        masks = outputs["mask_candidates_50"]
        leaderboard = outputs["joint_candidate_leaderboard"]
        stop_rules = outputs["stop_rules"]

        self.assertFalse(metric_top20.empty)
        self.assertFalse(metric_endpoint_top20.empty)
        self.assertIn("curve_search_status", metric_endpoint_top20.columns)
        self.assertFalse(linucb.empty)
        self.assertFalse(leaderboard.empty)
        self.assertEqual(len(masks), 50)
        self.assertTrue((masks["initial_size"] == inverse.INITIAL_SIZE).all())
        self.assertTrue((masks["test_size"] == inverse.TEST_SIZE).all())
        self.assertTrue((masks["users"] == inverse.REQUIRED_MASK_USERS).all())
        self.assertTrue((masks["items"] == inverse.REQUIRED_MASK_ITEMS).all())

        exact = stop_rules[stop_rules["stop_rule"] == "exact_replication"].iloc[0]
        self.assertEqual(exact["status"], "not_claimable")
        self.assertTrue(inverse.report_path().exists())


if __name__ == "__main__":
    unittest.main()
