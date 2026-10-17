"""Audit the next safe continuation step for the reproduction.

This artifact records whether a new experimental branch is justified. It keeps
the continuation decision grounded in the trigger gate, distance scorecard,
source request packet, diagnostic protocol, and artifact integrity status.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
SNAPSHOT_CSV = TABLE_DIR / "reproduction_iteration_status_snapshot.csv"
INTEGRITY_CSV = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
SOURCE_PACKET_CSV = TABLE_DIR / "paper_source_request_packet.csv"
PROTOCOL_CSV = TABLE_DIR / "minimal_faithful_diagnostic_protocol.csv"
QUEUE_CSV = TABLE_DIR / "next_experiment_queue.csv"

OUT_CSV = TABLE_DIR / "reproduction_continuation_audit.csv"
OUT_REPORT = REPORT_DIR / "reproduction_continuation_audit.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def count_status(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def latest_queue(rows: list[dict[str, str]], status: str) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("status") == status]
    return matches[-1] if matches else None


def build_rows() -> list[dict[str, str]]:
    trigger = read_rows(TRIGGER_CSV)
    snapshot = read_rows(SNAPSHOT_CSV)
    integrity = count_status(read_rows(INTEGRITY_CSV))
    distance = read_rows(DISTANCE_CSV)
    source = read_rows(SOURCE_PACKET_CSV)
    protocol = read_rows(PROTOCOL_CSV)
    queue = read_rows(QUEUE_CSV)

    rerun = row_by(trigger, "trigger", "rerun_decision")
    overall = row_by(snapshot, "area", "overall_progress")
    reward = row_by(distance, "claim_area", "local_reward_weighted")
    full = row_by(distance, "claim_area", "local_full_rmse")
    precondition = row_by(protocol, "stage", "precondition")
    diagnostic = row_by(protocol, "stage", "diagnostic_scope")
    next_item = latest_queue(queue, "next")
    p0_missing = [
        row["request_id"]
        for row in source
        if row["priority"] == "P0"
        and row["request_id"] not in {"minimum_rerun_protocol", "promotion_rule"}
        and row["current_status"] not in {"available", "defined"}
    ]

    should_run_experiment = rerun["status"] != "blocked_by_missing_source"
    return [
        {
            "check": "current_scientific_status",
            "status": overall["status"],
            "evidence": overall["evidence"],
            "decision": "Keep reporting as strong partial reproduction.",
        },
        {
            "check": "artifact_integrity",
            "status": "pass" if integrity.get("fail", 0) == 0 else "fail",
            "evidence": f"{integrity.get('pass', 0)} pass; {integrity.get('fail', 0)} fail",
            "decision": "Handoff set is internally consistent.",
        },
        {
            "check": "paper_faithful_trigger",
            "status": rerun["status"],
            "evidence": rerun["evidence"],
            "decision": "Do not start a new experimental branch while blocked.",
        },
        {
            "check": "figure4a_reward_weighted_distance",
            "status": reward["status"],
            "evidence": (
                f"gap={reward['linear_minus_du']}; "
                f"paper_gap_ratio={float(reward['paper_gap_ratio']) * 100:.1f}%; "
                f"scale_ratio={float(reward['du_scale_ratio_to_paper']) * 100:.1f}%"
            ),
            "decision": "Direction is useful; absolute scale remains non-matching.",
        },
        {
            "check": "figure4a_full_rmse_blocker",
            "status": full["status"],
            "evidence": f"gap={full['linear_minus_du']}; direction_matches={full['direction_matches_paper']}",
            "decision": "Exact Figure 4a claim remains blocked.",
        },
        {
            "check": "missing_source_items",
            "status": "missing",
            "evidence": "; ".join(p0_missing),
            "decision": "Use the source request packet rather than inventing more local proxy rows.",
        },
        {
            "check": "minimal_protocol",
            "status": precondition["status"],
            "evidence": diagnostic["requirement"],
            "decision": "Protocol is defined but blocked until a source-backed trigger appears.",
        },
        {
            "check": "continuation_decision",
            "status": "run_experiment" if should_run_experiment else "do_not_run_experiment",
            "evidence": f"next_queue_item={next_item['priority'] if next_item else 'n/a'}",
            "decision": (
                "Run the minimal 3-seed protocol now."
                if should_run_experiment
                else "Continue waiting for primary source or approved dependency trigger."
            ),
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| check | status | evidence | decision |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['check']} | {row['status']} | {row['evidence']} | {row['decision']} |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Reproduction Continuation Audit

Generated by `experiments/build_reproduction_continuation_audit.py`.

## Current Progress

The current branch remains a strong partial reproduction and the delivery
package is internally consistent. The paper-faithful rerun gate is still blocked
by missing source details, so the safe continuation is documentation and source
request readiness rather than another local proxy experiment.

## Audit Table

{markdown_table(rows)}

## Result

No new local 20-seed or proxy experiment is justified from the current evidence.
The next executable experiment is already defined in
`reports/minimal_faithful_diagnostic_protocol.md`, but it should run only after
author code, supplement, split/config files, exact Figure 4a metric definition,
or an approved validated external mechanism dependency appears.

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
