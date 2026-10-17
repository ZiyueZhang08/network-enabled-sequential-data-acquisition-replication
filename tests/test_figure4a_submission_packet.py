from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_figure4a_submission_packet as submission_packet


AUTHOR_REQUEST_CSV = ROOT / "outputs" / "tables" / "author_source_request_email.csv"
AUTHOR_REQUEST_REPORT = ROOT / "reports" / "author_source_request_email.md"


def read_author_request_rows() -> dict[str, dict[str, str]]:
    with AUTHOR_REQUEST_CSV.open(newline="", encoding="utf-8-sig") as file:
        return {row["item"]: row for row in csv.DictReader(file)}


class Figure4aSubmissionPacketTests(unittest.TestCase):
    def test_metric_rows_lock_frozen_submission_numbers(self) -> None:
        rows = submission_packet.metric_rows(pd.read_csv(submission_packet.FINAL_SUMMARY))
        by_item = {row["item"]: row for row in rows}

        self.assertEqual(by_item["primary_gap"]["value"], "0.0207")
        self.assertEqual(by_item["paper_gap_share"]["value"], "79.5%")
        self.assertEqual(by_item["myopic_gap"]["value"], "0.0210")
        self.assertEqual(by_item["conventional_rmse_caveat"]["value"], "-0.0077")
        self.assertEqual(
            by_item["source_request"]["value"],
            "reports\\author_source_request_email.md",
        )

    def test_report_preserves_caveated_submission_guardrails(self) -> None:
        text = submission_packet.OUT_REPORT.read_text(encoding="utf-8-sig")

        self.assertIn("ready_for_caveated_submission", text)
        self.assertIn("strong partial reproduction", text)
        self.assertIn("not be submitted\nas an exact DU-Net Figure 4a replication", text)
        self.assertIn("Exact Figure 4a replication achieved", text)
        self.assertIn("Do not run another broad local tuning sweep", text)
        self.assertIn("seeds `0,1,2`", text)

    def test_author_source_request_csv_is_ready_to_send(self) -> None:
        by_item = read_author_request_rows()

        self.assertEqual(by_item["email_subject"]["status"], "ready")
        self.assertIn("DU-Net Figure 4a MovieLens", by_item["email_subject"]["content"])
        self.assertEqual(by_item["p0_request_count"]["content"], "4")
        self.assertEqual(by_item["p1_request_count"]["content"], "1")
        self.assertEqual(by_item["execution_decision"]["status"], "ready_to_send")
        self.assertIn(
            "do not run another local proxy sweep",
            by_item["execution_decision"]["content"],
        )

    def test_author_source_request_body_preserves_reopen_triggers(self) -> None:
        by_item = read_author_request_rows()
        body = by_item["email_body"]["content"]
        report = AUTHOR_REQUEST_REPORT.read_text(encoding="utf-8")

        for expected in [
            "Network-Enabled Sequential Data Acquisition",
            "Figure 4a",
            "DU-Net-style 0.9811",
            "MF-centered Linear UCB proxy 1.0018",
            "Linear-minus-DU gap 0.0207",
            "79.5%",
            "Myopic 1.0021",
            "conventional RMSE still reverses the ordering",
            "Exact Figure 4a w-RMSE definition",
            "MovieLens split and initialization protocol",
            "DU-Net observational entropy and DLS implementation",
            "Per-cycle tuning protocol for lambda, sigma_t, p_t",
            "Linear UCB feature/update protocol",
            "seeds 0,1,2",
        ]:
            self.assertIn(expected, body)

        self.assertIn("Status: `ready_to_send`", report)
        self.assertIn("do not run another local proxy sweep", report)


if __name__ == "__main__":
    unittest.main()
