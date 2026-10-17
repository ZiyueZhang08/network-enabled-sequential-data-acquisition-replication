from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import build_local_source_artifact_inventory_audit as inventory
import build_paper_faithful_trigger_status as trigger_status


class TriggerGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_inventory_root = inventory.ROOT

    def tearDown(self) -> None:
        inventory.ROOT = self._old_inventory_root

    def _with_temp_root(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory()

    def _row(self, rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
        for row in rows:
            if row[key] == value:
                return row
        raise AssertionError(f"missing row {key}={value}")

    def test_known_local_artifacts_do_not_unlock_trigger_gate(self) -> None:
        with self._with_temp_root() as root_text:
            root = Path(root_text)
            (root / "configs").mkdir(parents=True)
            (root / "configs" / "local.json").write_text("{}", encoding="utf-8")
            (root / "data" / "raw" / "gcmc" / "flixster").mkdir(parents=True)
            (root / "data" / "raw" / "gcmc" / "flixster" / "training_test_dataset.mat").write_bytes(b"known")
            (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
            (root / "data" / "raw" / "ml-100k.zip").write_bytes(b"known")
            (root / "Network-Enabled Sequential Data Acquisition (1).pdf").write_bytes(b"known")

            inventory.ROOT = root
            inventory_rows = inventory.build_rows()
            inventory_decision = self._row(inventory_rows, "artifact", "execution_decision")
            self.assertEqual(inventory_decision["classification"], "no_new_actionable_source_artifact")
            self.assertFalse(
                any(
                    row["actionable_trigger_candidate"] == "yes"
                    for row in inventory_rows
                    if row["artifact"] != "execution_decision"
                )
            )

            trigger_rows = trigger_status.build_rows()
            self.assertEqual(
                self._row(trigger_rows, "trigger", "local_source_inventory")["status"],
                "clear",
            )
            self.assertEqual(
                self._row(trigger_rows, "trigger", "rerun_decision")["status"],
                "blocked_by_missing_source",
            )

    def test_unknown_pdf_archive_and_source_require_inspection(self) -> None:
        with self._with_temp_root() as root_text:
            root = Path(root_text)
            (root / "supplement.pdf").write_bytes(b"candidate")
            (root / "author_source.zip").write_bytes(b"candidate")
            (root / "analysis.ipynb").write_text("{}", encoding="utf-8")

            inventory.ROOT = root
            inventory_rows = inventory.build_rows()
            classifications = {
                row["artifact"]: row["classification"]
                for row in inventory_rows
                if row["artifact"] != "execution_decision"
            }
            self.assertEqual(classifications["supplement.pdf"], "unknown_pdf_candidate")
            self.assertEqual(classifications["author_source.zip"], "unknown_archive_candidate")
            self.assertEqual(classifications["analysis.ipynb"], "source_like_candidate")
            self.assertEqual(
                self._row(inventory_rows, "artifact", "execution_decision")["classification"],
                "inspect_actionable_candidates",
            )

            trigger_rows = trigger_status.build_rows()
            self.assertEqual(
                self._row(trigger_rows, "trigger", "author_code_or_supplement")["status"],
                "inspection_required",
            )
            self.assertEqual(
                self._row(trigger_rows, "trigger", "local_source_inventory")["status"],
                "inspection_required",
            )
            self.assertEqual(
                self._row(trigger_rows, "trigger", "rerun_decision")["status"],
                "inspection_required",
            )

    def test_unknown_split_or_config_candidates_mark_split_gate_for_inspection(self) -> None:
        with self._with_temp_root() as root_text:
            root = Path(root_text)
            (root / "figure4a_split.mat").write_bytes(b"candidate")
            (root / "author_config.yaml").write_text("seed: 1", encoding="utf-8")

            inventory.ROOT = root
            trigger_rows = trigger_status.build_rows()
            self.assertEqual(
                self._row(trigger_rows, "trigger", "exact_split_or_config_files")["status"],
                "inspection_required",
            )
            self.assertEqual(
                self._row(trigger_rows, "trigger", "rerun_decision")["status"],
                "inspection_required",
            )


if __name__ == "__main__":
    unittest.main()
