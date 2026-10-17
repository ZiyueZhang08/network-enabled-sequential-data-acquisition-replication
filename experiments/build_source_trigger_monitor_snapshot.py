"""Build a single source-trigger monitor snapshot for the blocked rerun gate.

The reproduction is intentionally frozen until a source-backed trigger appears.
This snapshot consolidates the public rescan, local inventory, gate status,
inspection protocol, author request, and next-run playbook into one handoff
artifact.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PUBLIC_RESCAN = TABLE_DIR / "public_source_rescan_audit.csv"
LOCAL_INVENTORY = TABLE_DIR / "local_source_artifact_inventory_audit.csv"
TRIGGER_STATUS = TABLE_DIR / "paper_faithful_trigger_status.csv"
INSPECTION_PROTOCOL = TABLE_DIR / "source_artifact_inspection_protocol.csv"
PLAYBOOK = TABLE_DIR / "paper_faithful_rerun_playbook.csv"
AUTHOR_EMAIL = TABLE_DIR / "author_source_request_email.csv"
USER_GOAL = TABLE_DIR / "figure4a_user_goal_comparison_brief.csv"
QUEUE = TABLE_DIR / "next_experiment_queue.csv"

OUT_CSV = TABLE_DIR / "source_trigger_monitor_snapshot.csv"
OUT_REPORT = REPORT_DIR / "source_trigger_monitor_snapshot.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value} in {rows[:1]}")


def latest_queue(rows: list[dict[str, str]], status: str) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("status") == status]
    return matches[-1] if matches else None


def actionable_inventory_count(rows: list[dict[str, str]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("artifact") != "execution_decision"
        and row.get("actionable_trigger_candidate") == "yes"
    )


def build_rows() -> list[dict[str, str]]:
    public = read_rows(PUBLIC_RESCAN)
    local = read_rows(LOCAL_INVENTORY)
    trigger = read_rows(TRIGGER_STATUS)
    inspection = read_rows(INSPECTION_PROTOCOL)
    playbook = read_rows(PLAYBOOK)
    author = read_rows(AUTHOR_EMAIL)
    user_goal = read_rows(USER_GOAL)
    queue = read_rows(QUEUE)

    public_decision = row_by(public, "check_id", "execution_decision")
    local_decision = row_by(local, "artifact", "execution_decision")
    rerun = row_by(trigger, "trigger", "rerun_decision")
    local_source = row_by(trigger, "trigger", "local_source_inventory")
    public_web = row_by(trigger, "trigger", "public_web_recovery")
    inspection_decision = row_by(inspection, "candidate_class", "execution_decision")
    diagnostic = row_by(playbook, "step", "7_minimal_3seed_diagnostic")
    promotion = row_by(playbook, "step", "11_promote_or_freeze")
    author_decision = row_by(author, "item", "execution_decision")
    goal_verdict = row_by(user_goal, "question", "user_goal_verdict")
    last_done = latest_queue(queue, "done")
    next_item = latest_queue(queue, "next")

    candidate_count = actionable_inventory_count(local)
    run_status = "blocked" if rerun["status"] == "blocked_by_missing_source" else "inspect_or_run"
    return [
        {
            "monitor_item": "scientific_status",
            "status": "strong_partial_reproduction",
            "evidence": goal_verdict["distance_to_goal"],
            "next_action": "Report Linear-UCB-consistent partial result; do not claim exact Figure 4a.",
        },
        {
            "monitor_item": "public_source_rescan",
            "status": public_decision["result"],
            "evidence": public_decision["evidence"],
            "next_action": public_decision["decision"],
        },
        {
            "monitor_item": "local_source_inventory",
            "status": local_source["status"],
            "evidence": f"{candidate_count} actionable local candidate(s); {local_decision['interpretation']}",
            "next_action": local_source["decision"],
        },
        {
            "monitor_item": "public_web_recovery",
            "status": public_web["status"],
            "evidence": public_web["evidence"],
            "next_action": public_web["decision"],
        },
        {
            "monitor_item": "rerun_gate",
            "status": rerun["status"],
            "evidence": rerun["evidence"],
            "next_action": rerun["decision"],
        },
        {
            "monitor_item": "inspection_protocol",
            "status": inspection_decision["inspection_status"],
            "evidence": inspection_decision["current_state"],
            "next_action": inspection_decision["inspection_action"],
        },
        {
            "monitor_item": "author_request",
            "status": author_decision["status"],
            "evidence": author_decision["content"],
            "next_action": "Use the prepared email/request packet to obtain source-backed trigger materials.",
        },
        {
            "monitor_item": "next_executable_run",
            "status": run_status,
            "evidence": diagnostic["action"],
            "next_action": "Run only after the rerun gate changes from blocked to inspection-cleared.",
        },
        {
            "monitor_item": "promotion_rule",
            "status": "defined",
            "evidence": promotion["pass_rule"],
            "next_action": "Promote to 20 seeds only if the 3-seed diagnostic improves or explains the active blocker.",
        },
        {
            "monitor_item": "queue_state",
            "status": f"last_done={last_done['priority'] if last_done else 'n/a'}; next={next_item['priority'] if next_item else 'n/a'}",
            "evidence": f"last={last_done['task'] if last_done else 'n/a'}; next={next_item['task'] if next_item else 'n/a'}",
            "next_action": next_item["reason"] if next_item else "No queued next item.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| monitor item | status | evidence | next action |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['monitor_item']} | {row['status']} | {row['evidence']} | {row['next_action']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    rerun = row_by(rows, "monitor_item", "rerun_gate")
    queue = row_by(rows, "monitor_item", "queue_state")
    report = f"""# Source Trigger Monitor Snapshot

Generated by `experiments/build_source_trigger_monitor_snapshot.py`.

## Current Progress

The reproduction remains a strong partial reproduction. The monitor asks one
question: is there now enough source-backed evidence to leave the blocked state
and run the minimal MovieLens Figure 4a diagnostic?

## Monitor Table

{markdown_table(rows)}

## Decision

Rerun gate: `{rerun['status']}`.

Queue: `{queue['status']}`.

No model rerun is authorized while the gate remains
`blocked_by_missing_source`. If a future PDF/archive/source/config/matrix
candidate appears, inspect it with the source artifact inspection protocol
before launching the 3-seed diagnostic.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
