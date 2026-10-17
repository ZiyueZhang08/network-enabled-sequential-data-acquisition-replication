from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_source_trigger_intake_check as intake


class SourceTriggerIntakeCheckTests(unittest.TestCase):
    def test_intake_sequence_refreshes_gate_before_reports(self) -> None:
        sequence = intake.BUILDER_SEQUENCE

        self.assertLess(
            sequence.index("build_local_source_artifact_inventory_audit"),
            sequence.index("build_paper_faithful_trigger_status"),
        )
        self.assertLess(
            sequence.index("build_paper_faithful_trigger_status"),
            sequence.index("build_source_trigger_monitor_snapshot"),
        )
        self.assertLess(
            sequence.index("build_author_source_request_email"),
            sequence.index("build_author_source_request_send_checklist"),
        )
        self.assertLess(
            sequence.index("build_author_source_request_send_checklist"),
            sequence.index("build_author_source_response_intake_template"),
        )
        self.assertLess(
            sequence.index("build_author_source_response_intake_template"),
            sequence.index("build_source_trigger_monitor_snapshot"),
        )
        self.assertLess(
            sequence.index("build_source_trigger_monitor_snapshot"),
            sequence.index("build_reproduction_delivery_package"),
        )
        self.assertLess(
            sequence.index("build_reproduction_delivery_package"),
            sequence.index("build_reproduction_artifact_integrity_check"),
        )

    def test_intake_sequence_does_not_run_proxy_model_experiments(self) -> None:
        forbidden_prefixes = ("run_", "plot_")
        forbidden_names = {
            "build_figure4a_reward_weight_metric_20seeds",
            "build_figure4a_rating_tail_20seed_confirmation",
            "run_main",
            "run_movielens",
            "run_movielens_grid",
        }

        self.assertFalse(any(name.startswith(forbidden_prefixes) for name in intake.BUILDER_SEQUENCE))
        self.assertTrue(forbidden_names.isdisjoint(set(intake.BUILDER_SEQUENCE)))


if __name__ == "__main__":
    unittest.main()
