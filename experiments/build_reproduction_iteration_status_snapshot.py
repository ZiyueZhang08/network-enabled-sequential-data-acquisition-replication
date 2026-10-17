"""Build the latest reproduction iteration status snapshot.

This report is intentionally read-only with respect to experiment results. It
summarizes the frozen Figure 4a comparison, trigger gate, artifact integrity,
and next queue action for a concise handoff after a continuation request.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FIGURE4A_CSV = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
INTEGRITY_CSV = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
QUEUE_CSV = TABLE_DIR / "next_experiment_queue.csv"

OUT_CSV = TABLE_DIR / "reproduction_iteration_status_snapshot.csv"
OUT_REPORT = REPORT_DIR / "reproduction_iteration_status_snapshot.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row where {key}={value} in input table")


def count_status(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def latest_queue_rows(rows: list[dict[str, str]]) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    done_rows = [row for row in rows if row.get("status") == "done"]
    next_rows = [row for row in rows if row.get("status") == "next"]
    return (done_rows[-1] if done_rows else None, next_rows[-1] if next_rows else None)


def build_snapshot() -> list[dict[str, str]]:
    figure_rows = read_rows(FIGURE4A_CSV)
    trigger_rows = read_rows(TRIGGER_CSV)
    integrity_rows = read_rows(INTEGRITY_CSV)
    queue_rows = read_rows(QUEUE_CSV)

    paper = row_by(figure_rows, "claim_area", "paper_target")
    weighted = row_by(figure_rows, "claim_area", "local_reward_weighted")
    full = row_by(figure_rows, "claim_area", "local_full_rmse")
    high = row_by(figure_rows, "claim_area", "local_high_rating_region")
    sigmoid = row_by(figure_rows, "claim_area", "sigmoid_reward_weighted")
    rerun = row_by(trigger_rows, "trigger", "rerun_decision")
    integrity_counts = count_status(integrity_rows)
    last_done, next_row = latest_queue_rows(queue_rows)

    return [
        {
            "area": "overall_progress",
            "status": "strong_partial_reproduction",
            "evidence": (
                "Figure 4a direction is recovered only in local reward-weighted "
                "and high-rating regions; exact scale/full-RMSE ordering is not recovered."
            ),
            "decision": "Report as strong partial reproduction, not paper-exact DU-Net replication.",
        },
        {
            "area": "paper_target",
            "status": paper["status"],
            "evidence": f"DU-Net {paper['du_net']} vs Linear UCB {paper['linear_ucb']}; gap {paper['linear_minus_du']}",
            "decision": "Use as visual target from the local PDF extraction.",
        },
        {
            "area": "local_reward_weighted",
            "status": weighted["status"],
            "evidence": f"DU-Net {weighted['du_net']} vs Linear UCB {weighted['linear_ucb']}; gap {weighted['linear_minus_du']}",
            "decision": "Direction is consistent with Figure 4a/Linear-UCB comparison.",
        },
        {
            "area": "local_full_rmse",
            "status": full["status"],
            "evidence": f"DU-Net {full['du_net']} vs Linear UCB {full['linear_ucb']}; gap {full['linear_minus_du']}",
            "decision": "Do not claim exact Figure 4a reproduction.",
        },
        {
            "area": "local_high_rating_region",
            "status": high["status"],
            "evidence": f"DU-Net {high['du_net']} vs Linear UCB {high['linear_ucb']}; gap {high['linear_minus_du']}",
            "decision": "Use as mechanism support for reward-region consistency.",
        },
        {
            "area": "sigmoid_reward_weighted",
            "status": sigmoid["status"],
            "evidence": f"DU-Net {sigmoid['du_net']} vs Linear UCB {sigmoid['linear_ucb']}; gap {sigmoid['linear_minus_du']}",
            "decision": "Treat as near-tie, not a stable DU-Net win.",
        },
        {
            "area": "artifact_integrity",
            "status": "pass" if integrity_counts.get("fail", 0) == 0 else "fail",
            "evidence": f"{integrity_counts.get('pass', 0)} pass; {integrity_counts.get('fail', 0)} fail",
            "decision": "Delivery package remains internally consistent.",
        },
        {
            "area": "paper_faithful_rerun_gate",
            "status": rerun["status"],
            "evidence": rerun["evidence"],
            "decision": rerun["decision"],
        },
        {
            "area": "queue_progress",
            "status": "advanced" if last_done and next_row else "unknown",
            "evidence": (
                f"last_done={last_done['priority'] if last_done else 'n/a'}; "
                f"next={next_row['priority'] if next_row else 'n/a'}"
            ),
            "decision": next_row["reason"] if next_row else "Inspect queue before continuing.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| area | status | evidence | decision |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['area']} | {row['status']} | {row['evidence']} | {row['decision']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# Reproduction Iteration Status Snapshot

Generated by `experiments/build_reproduction_iteration_status_snapshot.py` at
{generated}.

## Status

The reproduction remains a strong partial reproduction. The current evidence is
consistent with the paper's DU-Net-over-Linear-UCB relationship only under the
local reward-weighted Figure 4a reading and in high-rating regions. It is not an
exact paper reproduction because the absolute w-RMSE scale differs and full-test
RMSE still favors Linear UCB.

## Snapshot Table

{markdown_table(rows)}

## Analysis

No new paper-faithful trigger is active. The correct next action is therefore
not another local proxy sweep. Future execution should resume only when author
code, supplement, exact split/config files, precise Figure 4a w-RMSE definition,
or an approved validated mechanism dependency becomes available.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_snapshot()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
