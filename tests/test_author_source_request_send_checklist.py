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

import build_author_source_request_send_checklist as checklist


class AuthorSourceRequestSendChecklistTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_paths = {
            "AUTHOR_EMAIL": checklist.AUTHOR_EMAIL,
            "BUNDLE_META": checklist.BUNDLE_META,
            "OUT_CSV": checklist.OUT_CSV,
            "OUT_REPORT": checklist.OUT_REPORT,
            "ROOT": checklist.ROOT,
        }

    def tearDown(self) -> None:
        for name, value in self._old_paths.items():
            setattr(checklist, name, value)

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _row(self, rows: list[dict[str, str]], step: str) -> dict[str, str]:
        for row in rows:
            if row["step"] == step:
                return row
        raise AssertionError(f"missing step={step}")

    def _install_fixtures(self, root: Path) -> None:
        paths = {
            "AUTHOR_EMAIL": root / "author_source_request_email.csv",
            "BUNDLE_META": root / "figure4a_submission_bundle_latest.csv",
            "OUT_CSV": root / "author_source_request_send_checklist.csv",
            "OUT_REPORT": root / "author_source_request_send_checklist.md",
        }
        for name, path in paths.items():
            setattr(checklist, name, path)
        checklist.ROOT = root

        self._write_csv(
            checklist.AUTHOR_EMAIL,
            [
                {
                    "item": "email_subject",
                    "status": "ready",
                    "content": "Request for DU-Net Figure 4a MovieLens reproduction details",
                },
                {
                    "item": "p0_request_count",
                    "status": "ready",
                    "content": "4",
                },
                {
                    "item": "p1_request_count",
                    "status": "ready",
                    "content": "1",
                },
                {
                    "item": "execution_decision",
                    "status": "ready_to_send",
                    "content": "Use this email to request the source-backed trigger.",
                },
            ],
        )
        self._write_csv(
            checklist.BUNDLE_META,
            [
                {"item": "zip_path", "value": "outputs\\figure4a_submission_bundle_20260708_006.zip"},
                {"item": "zip_sha256", "value": "abc123"},
                {"item": "submission_status", "value": "ready_for_caveated_submission"},
                {"item": "primary_gap", "value": "0.0207"},
            ],
        )

    def test_checklist_keeps_send_step_external_and_preserves_rerun_gate(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            self._install_fixtures(Path(root_text))
            rows = checklist.build_rows()

            self.assertEqual(self._row(rows, "email_body")["status"], "ready")
            self.assertIn("p0=4; p1=1", self._row(rows, "email_body")["evidence"])
            self.assertEqual(
                self._row(rows, "recipient_logging")["status"],
                "external_action_required",
            )
            self.assertEqual(
                self._row(rows, "waiting_state")["status"],
                "waiting_for_external_response",
            )
            self.assertIn(
                "seeds 0,1,2",
                self._row(rows, "response_intake")["next_action"],
            )

    def test_main_writes_report_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            self._install_fixtures(Path(root_text))
            checklist.main()

            self.assertTrue(checklist.OUT_CSV.exists())
            self.assertTrue(checklist.OUT_REPORT.exists())
            report = checklist.OUT_REPORT.read_text(encoding="utf-8")
            self.assertIn("ready_for_caveated_submission", report)
            self.assertIn("seeds 0,1,2", report)


if __name__ == "__main__":
    unittest.main()
