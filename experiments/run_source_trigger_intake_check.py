"""Run the source-trigger intake check from one stable entry point.

This runner is intentionally not a model experiment. It refreshes the source
and gate artifacts that decide whether a paper-faithful Figure 4a diagnostic is
authorized.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

TRIGGER_STATUS = TABLE_DIR / "paper_faithful_trigger_status.csv"
MONITOR = TABLE_DIR / "source_trigger_monitor_snapshot.csv"
INTEGRITY = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
QUEUE = TABLE_DIR / "next_experiment_queue.csv"

OUT_CSV = TABLE_DIR / "source_trigger_intake_check.csv"
OUT_REPORT = REPORT_DIR / "source_trigger_intake_check.md"


CORE_BUILDERS = [
    "build_public_source_rescan_audit",
    "build_local_source_artifact_inventory_audit",
    "build_paper_faithful_trigger_status",
    "build_source_artifact_inspection_protocol",
    "build_author_source_request_email",
    "build_author_source_request_send_checklist",
    "build_author_source_response_intake_template",
    "build_source_trigger_monitor_snapshot",
]

REPORT_BUILDERS = [
    "build_reproduction_delivery_package",
    "build_reproduction_artifact_integrity_check",
    "build_reproduction_claim_evidence_index",
    "build_reproduction_final_user_brief_zh",
]

BUILDER_SEQUENCE = CORE_BUILDERS + REPORT_BUILDERS + REPORT_BUILDERS


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def latest_queue(rows: list[dict[str, str]], status: str) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("status") == status]
    return matches[-1] if matches else None


def import_builder(module_name: str) -> ModuleType:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"{module_name} has no main()")
    return module


def run_builder(module_name: str) -> None:
    module = import_builder(module_name)
    module.main()


def final_status_rows(executed: list[str]) -> list[dict[str, str]]:
    trigger = read_rows(TRIGGER_STATUS)
    monitor = read_rows(MONITOR)
    integrity = read_rows(INTEGRITY)
    queue = read_rows(QUEUE)

    rerun = row_by(trigger, "trigger", "rerun_decision")
    next_run = row_by(monitor, "monitor_item", "next_executable_run")
    queue_state = row_by(monitor, "monitor_item", "queue_state")
    last_done = latest_queue(queue, "done")
    next_item = latest_queue(queue, "next")
    pass_count = sum(1 for row in integrity if row.get("status") == "pass")
    fail_count = sum(1 for row in integrity if row.get("status") == "fail")

    return [
        {
            "item": "executed_builders",
            "status": "completed",
            "evidence": "; ".join(executed),
            "decision": "All intake builders completed in the stable source-trigger order.",
        },
        {
            "item": "rerun_gate",
            "status": rerun["status"],
            "evidence": rerun["evidence"],
            "decision": rerun["decision"],
        },
        {
            "item": "next_executable_run",
            "status": next_run["status"],
            "evidence": next_run["evidence"],
            "decision": next_run["next_action"],
        },
        {
            "item": "queue_state",
            "status": queue_state["status"],
            "evidence": f"last_done={last_done['priority'] if last_done else 'n/a'}; next={next_item['priority'] if next_item else 'n/a'}",
            "decision": next_item["reason"] if next_item else "No queued next item.",
        },
        {
            "item": "artifact_integrity",
            "status": "pass" if fail_count == 0 else "fail",
            "evidence": f"{pass_count} pass; {fail_count} fail",
            "decision": "The handoff artifacts are internally consistent." if fail_count == 0 else "Fix artifact failures before handoff.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| item | status | evidence | decision |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['item']} | {row['status']} | {row['evidence']} | {row['decision']} |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    rerun = row_by(rows, "item", "rerun_gate")
    next_run = row_by(rows, "item", "next_executable_run")
    report = f"""# Source Trigger Intake Check

Generated by `experiments/run_source_trigger_intake_check.py`.

## Current Progress

This is the one-command intake refresh for future source materials. It refreshes
the public/local source scans, paper-faithful gate, inspection protocol,
source-trigger monitor, final reporting artifacts, and artifact integrity.

## Intake Summary

{markdown_table(rows)}

## Decision

Rerun gate: `{rerun['status']}`.

Next executable run: `{next_run['status']}`.

If the rerun gate remains `blocked_by_missing_source`, do not run a model
experiment. If it changes to `inspection_required`, inspect the candidate
artifact first and launch the minimal 3-seed diagnostic only after the
inspection acceptance rules pass.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    executed: list[str] = []
    for module_name in BUILDER_SEQUENCE:
        run_builder(module_name)
        executed.append(module_name)
    rows = final_status_rows(executed)
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
