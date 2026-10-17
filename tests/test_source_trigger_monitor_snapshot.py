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

import build_source_trigger_monitor_snapshot as monitor


class SourceTriggerMonitorSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_paths = {
            "PUBLIC_RESCAN": monitor.PUBLIC_RESCAN,
            "LOCAL_INVENTORY": monitor.LOCAL_INVENTORY,
            "TRIGGER_STATUS": monitor.TRIGGER_STATUS,
            "INSPECTION_PROTOCOL": monitor.INSPECTION_PROTOCOL,
            "PLAYBOOK": monitor.PLAYBOOK,
            "AUTHOR_EMAIL": monitor.AUTHOR_EMAIL,
            "USER_GOAL": monitor.USER_GOAL,
            "QUEUE": monitor.QUEUE,
            "OUT_CSV": monitor.OUT_CSV,
            "OUT_REPORT": monitor.OUT_REPORT,
            "ROOT": monitor.ROOT,
        }

    def tearDown(self) -> None:
        for name, value in self._old_paths.items():
            setattr(monitor, name, value)

    def _row(self, rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
        for row in rows:
            if row[key] == value:
                return row
        raise AssertionError(f"missing row {key}={value}")

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _install_fixtures(self, root: Path, *, inspection_required: bool = False) -> None:
        public_status = "inspection_required" if inspection_required else "blocked_by_missing_source"
        local_status = "inspection_required" if inspection_required else "clear"
        rerun_status = "inspection_required" if inspection_required else "blocked_by_missing_source"
        candidate_rows = [
            {
                "artifact": "supplement.pdf",
                "classification": "unknown_pdf_candidate",
                "size_bytes": "12",
                "last_write_time": "2026-07-07T00:00:00",
                "actionable_trigger_candidate": "yes",
                "interpretation": "Unclassified PDF; inspect before keeping the rerun gate closed.",
            },
            {
                "artifact": "author_source.zip",
                "classification": "unknown_archive_candidate",
                "size_bytes": "12",
                "last_write_time": "2026-07-07T00:00:00",
                "actionable_trigger_candidate": "yes",
                "interpretation": "Archive could contain supplement/source/config material.",
            },
        ] if inspection_required else []

        paths = {
            "PUBLIC_RESCAN": root / "public_source_rescan_audit.csv",
            "LOCAL_INVENTORY": root / "local_source_artifact_inventory_audit.csv",
            "TRIGGER_STATUS": root / "paper_faithful_trigger_status.csv",
            "INSPECTION_PROTOCOL": root / "source_artifact_inspection_protocol.csv",
            "PLAYBOOK": root / "paper_faithful_rerun_playbook.csv",
            "AUTHOR_EMAIL": root / "author_source_request_email.csv",
            "USER_GOAL": root / "figure4a_user_goal_comparison_brief.csv",
            "QUEUE": root / "next_experiment_queue.csv",
            "OUT_CSV": root / "source_trigger_monitor_snapshot.csv",
            "OUT_REPORT": root / "source_trigger_monitor_snapshot.md",
        }
        for name, path in paths.items():
            setattr(monitor, name, path)
        monitor.ROOT = root

        self._write_csv(
            monitor.PUBLIC_RESCAN,
            [
                {
                    "check_id": "execution_decision",
                    "source_or_query": "combined public rescan",
                    "result": public_status,
                    "evidence": "No source-backed public artifact was recovered.",
                    "decision": "Do not run another proxy sweep.",
                }
            ],
        )
        self._write_csv(
            monitor.LOCAL_INVENTORY,
            candidate_rows
            + [
                {
                    "artifact": "execution_decision",
                    "classification": "inspect_actionable_candidates" if inspection_required else "no_new_actionable_source_artifact",
                    "size_bytes": "",
                    "last_write_time": "",
                    "actionable_trigger_candidate": "yes" if inspection_required else "no",
                    "interpretation": (
                        "2 unclassified local artifact(s) require inspection before the gate can stay closed."
                        if inspection_required
                        else "Only known paper/data/Monti/local-config artifacts are present; no new source-backed rerun trigger."
                    ),
                }
            ],
        )
        self._write_csv(
            monitor.TRIGGER_STATUS,
            [
                {
                    "trigger": "local_source_inventory",
                    "status": local_status,
                    "evidence": "Inventory status.",
                    "decision": "Inspect actionable local artifacts." if inspection_required else "Inventory is clear.",
                },
                {
                    "trigger": "public_web_recovery",
                    "status": "not_triggered",
                    "evidence": "No public DU-Net source recovered.",
                    "decision": "Treat external source recovery as unavailable.",
                },
                {
                    "trigger": "rerun_decision",
                    "status": rerun_status,
                    "evidence": "Candidate artifacts require inspection." if inspection_required else "All rerun triggers are currently absent.",
                    "decision": "Inspect before rerun." if inspection_required else "No further local proxy sweep.",
                },
            ],
        )
        self._write_csv(
            monitor.INSPECTION_PROTOCOL,
            [
                {
                    "candidate_class": "execution_decision",
                    "inspection_status": "inspection_required" if inspection_required else "ready_but_idle",
                    "inspection_action": "Inspect candidate artifacts before rerun." if inspection_required else "Keep protocol ready; no candidate is present.",
                    "unlocks": "minimal_3seed_diagnostic_after_source_mapping",
                    "acceptance_rule": "Proceed only after acceptance rules pass.",
                    "current_state": rerun_status,
                }
            ],
        )
        self._write_csv(
            monitor.PLAYBOOK,
            [
                {
                    "step": "7_minimal_3seed_diagnostic",
                    "action": "Run MovieLens Figure 4a seeds 0-2 only.",
                    "pass_rule": "",
                },
                {
                    "step": "11_promote_or_freeze",
                    "action": "Promote or freeze.",
                    "pass_rule": "Promote only if a primary blocker improves.",
                },
            ],
        )
        self._write_csv(
            monitor.AUTHOR_EMAIL,
            [
                {
                    "item": "execution_decision",
                    "status": "ready_to_send",
                    "content": "Use this email to request the source-backed trigger.",
                }
            ],
        )
        self._write_csv(
            monitor.USER_GOAL,
            [
                {
                    "question": "user_goal_verdict",
                    "distance_to_goal": "absolute scale not close; reward-weighted gap recovers part of paper gap",
                }
            ],
        )
        self._write_csv(
            monitor.QUEUE,
            [
                {
                    "priority": "71",
                    "phase": "long-term",
                    "status": "done",
                    "task": "Build source trigger monitor snapshot",
                    "command": "python experiments\\build_source_trigger_monitor_snapshot.py",
                    "expected_artifact": "reports/source_trigger_monitor_snapshot.md",
                    "reason": "Monitor current gate.",
                },
                {
                    "priority": "72",
                    "phase": "long-term",
                    "status": "next",
                    "task": "Wait for primary source or approved dependency trigger",
                    "command": "TBD",
                    "expected_artifact": "TBD",
                    "reason": "Resume only if source-backed material arrives.",
                },
            ],
        )

    def test_blocked_monitor_prevents_model_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            self._install_fixtures(Path(root_text))
            rows = monitor.build_rows()

            self.assertEqual(
                self._row(rows, "monitor_item", "rerun_gate")["status"],
                "blocked_by_missing_source",
            )
            self.assertEqual(
                self._row(rows, "monitor_item", "next_executable_run")["status"],
                "blocked",
            )
            self.assertIn(
                "0 actionable local candidate",
                self._row(rows, "monitor_item", "local_source_inventory")["evidence"],
            )
            self.assertEqual(
                self._row(rows, "monitor_item", "queue_state")["status"],
                "last_done=71; next=72",
            )

            monitor.main()
            self.assertTrue(monitor.OUT_CSV.exists())
            self.assertTrue(monitor.OUT_REPORT.exists())

    def test_candidate_monitor_requires_inspection_before_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            self._install_fixtures(Path(root_text), inspection_required=True)
            rows = monitor.build_rows()

            self.assertEqual(
                self._row(rows, "monitor_item", "rerun_gate")["status"],
                "inspection_required",
            )
            self.assertEqual(
                self._row(rows, "monitor_item", "next_executable_run")["status"],
                "inspect_or_run",
            )
            self.assertIn(
                "2 actionable local candidate",
                self._row(rows, "monitor_item", "local_source_inventory")["evidence"],
            )
            self.assertEqual(
                self._row(rows, "monitor_item", "inspection_protocol")["status"],
                "inspection_required",
            )


if __name__ == "__main__":
    unittest.main()
