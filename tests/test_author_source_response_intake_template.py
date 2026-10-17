from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_author_source_response_intake_template as intake


class AuthorSourceResponseIntakeTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_paths = {
            "SEND_CHECKLIST": intake.SEND_CHECKLIST,
            "NEXT_TRIGGER": intake.NEXT_TRIGGER,
            "SOURCE_PACKET": intake.SOURCE_PACKET,
            "OUT_CSV": intake.OUT_CSV,
            "OUT_REPORT": intake.OUT_REPORT,
            "ROOT": intake.ROOT,
        }

    def tearDown(self) -> None:
        for name, value in self._old_paths.items():
            setattr(intake, name, value)

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _row(self, rows: list[dict[str, str]], item: str) -> dict[str, str]:
        for row in rows:
            if row["response_item"] == item:
                return row
        raise AssertionError(f"missing response_item={item}")

    def _install_fixtures(self, root: Path) -> None:
        paths = {
            "SEND_CHECKLIST": root / "author_source_request_send_checklist.csv",
            "NEXT_TRIGGER": root / "figure4a_next_trigger_protocol.csv",
            "SOURCE_PACKET": root / "paper_source_request_packet.csv",
            "OUT_CSV": root / "author_source_response_intake_template.csv",
            "OUT_REPORT": root / "author_source_response_intake_template.md",
        }
        for name, path in paths.items():
            setattr(intake, name, path)
        intake.ROOT = root

        self._write_csv(
            intake.SEND_CHECKLIST,
            [
                {
                    "step": "waiting_state",
                    "status": "waiting_for_external_response",
                    "evidence": "No material yet.",
                    "next_action": "Do not run another local proxy sweep while waiting.",
                }
            ],
        )
        self._write_csv(
            intake.NEXT_TRIGGER,
            [
                {
                    "trigger": "Exact Figure 4a w-RMSE definition recovered",
                    "minimum_input": "metric",
                    "first_action": "metric action",
                    "validation": "Re-evaluate saved predictions or run seeds 0,1,2 only",
                    "promote_to_20seed_if": "metric gate passes",
                },
                {
                    "trigger": "Author MovieLens split or initial/candidate masks recovered",
                    "minimum_input": "split",
                    "first_action": "split action",
                    "validation": "Run seeds 0,1,2",
                    "promote_to_20seed_if": "split gate passes",
                },
                {
                    "trigger": "DU-Net entropy/DLS implementation recovered",
                    "minimum_input": "backend",
                    "first_action": "backend action",
                    "validation": "Run seeds 0,1,2",
                    "promote_to_20seed_if": "backend gate passes",
                },
                {
                    "trigger": "Per-cycle lambda/sigma_t/p_t tuning protocol recovered",
                    "minimum_input": "tuning",
                    "first_action": "tuning action",
                    "validation": "Run 1 seed first, then seeds 0,1,2",
                    "promote_to_20seed_if": "tuning gate passes",
                },
                {
                    "trigger": "Paper Linear UCB protocol recovered",
                    "minimum_input": "linear",
                    "first_action": "linear action",
                    "validation": "Run seeds 0,1,2",
                    "promote_to_20seed_if": "linear gate passes",
                },
            ],
        )
        self._write_csv(
            intake.SOURCE_PACKET,
            [
                {
                    "request_id": "exact_figure4a_metric",
                    "needed_artifact": "Exact metric.",
                },
                {
                    "request_id": "exact_split_and_preprocessing",
                    "needed_artifact": "Exact split.",
                },
                {
                    "request_id": "du_net_backend_config",
                    "needed_artifact": "Backend config.",
                },
                {
                    "request_id": "linear_ucb_adapter",
                    "needed_artifact": "Linear UCB adapter.",
                },
            ],
        )

    def test_response_template_keeps_waiting_state_and_small_rerun_gate(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            self._install_fixtures(Path(root_text))
            rows = intake.build_rows()

            self.assertEqual(
                self._row(rows, "intake_decision")["status"],
                "waiting_for_external_response",
            )
            self.assertIn(
                "seeds 0,1,2",
                self._row(rows, "intake_decision")["first_validation"],
            )
            self.assertEqual(
                self._row(rows, "non_actionable_reply")["first_validation"],
                "No model rerun.",
            )
            self.assertIn(
                "Exact metric",
                self._row(rows, "exact_metric_reply")["minimum_evidence"],
            )

    def test_main_writes_template_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            self._install_fixtures(Path(root_text))
            intake.main()

            self.assertTrue(intake.OUT_CSV.exists())
            self.assertTrue(intake.OUT_REPORT.exists())
            report = intake.OUT_REPORT.read_text(encoding="utf-8")
            self.assertIn("waiting_for_external_response", report)
            self.assertIn("seeds 0,1,2", report)


if __name__ == "__main__":
    unittest.main()
