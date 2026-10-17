"""Close the current execution gate for Figure 4a reproduction.

This artifact records the current run/stop decision after all available local
source, split, metric, and comparator audits. It is intentionally conservative:
it prevents another proxy sweep from being mistaken for progress toward an
exact DU-Net Figure 4a reproduction.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FIGURE4A_CSV = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
REMAINING_GAP_CSV = TABLE_DIR / "remaining_gap_priority_audit.csv"
PROTOCOL_CSV = TABLE_DIR / "minimal_faithful_diagnostic_protocol.csv"
GCMC_CSV = TABLE_DIR / "gcmc_monti_split_provenance_audit.csv"
QUEUE_CSV = TABLE_DIR / "next_experiment_queue.csv"

OUT_CSV = TABLE_DIR / "execution_gate_closure_audit.csv"
OUT_REPORT = REPORT_DIR / "execution_gate_closure_audit.md"


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


def p0_open_items(rows: list[dict[str, str]]) -> list[str]:
    return [
        row["gap_id"]
        for row in rows
        if row["priority"] == "P0" and row["current_status"] not in {"available", "defined"}
    ]


def build_rows() -> list[dict[str, str]]:
    figure = read_rows(FIGURE4A_CSV)
    distance = read_rows(DISTANCE_CSV)
    trigger = read_rows(TRIGGER_CSV)
    gaps = read_rows(REMAINING_GAP_CSV)
    protocol = read_rows(PROTOCOL_CSV)
    gcmc = read_rows(GCMC_CSV)
    queue = read_rows(QUEUE_CSV)

    paper = row_by(figure, "claim_area", "paper_target")
    weighted = row_by(figure, "claim_area", "local_reward_weighted")
    full = row_by(figure, "claim_area", "local_full_rmse")
    high = row_by(figure, "claim_area", "local_high_rating_region")
    weighted_distance = row_by(distance, "claim_area", "local_reward_weighted")
    rerun = row_by(trigger, "trigger", "rerun_decision")
    protocol_precondition = row_by(protocol, "stage", "precondition")
    diagnostic_scope = row_by(protocol, "stage", "diagnostic_scope")
    promotion = row_by(protocol, "stage", "promotion_decision")
    gcmc_decision = row_by(gcmc, "check_id", "execution_decision")
    next_item = latest_queue(queue, "next")
    p0_open = p0_open_items(gaps)

    return [
        {
            "gate_id": "user_goal_status",
            "status": "partially_satisfied",
            "evidence": (
                f"paper DU={paper['du_net']} Linear={paper['linear_ucb']}; "
                f"local reward-weighted DU={weighted['du_net']} Linear={weighted['linear_ucb']}; "
                f"full-RMSE gap={full['linear_minus_du']}"
            ),
            "decision": "Report strong partial reproduction and Linear-UCB consistency, not exact Figure 4a.",
            "unlock_condition": "Exact metric/split/backend evidence changes the full-RMSE or scale interpretation.",
        },
        {
            "gate_id": "best_local_support",
            "status": "direction_matches_scale_gap_remains",
            "evidence": (
                f"reward-weighted gap={weighted['linear_minus_du']}; "
                f"paper_gap_ratio={float(weighted_distance['paper_gap_ratio']) * 100:.1f}%; "
                f"du_scale_ratio={float(weighted_distance['du_scale_ratio_to_paper']) * 100:.1f}%"
            ),
            "decision": "Use reward-weighted/high-rating evidence as the reportable local consistency result.",
            "unlock_condition": "Source-backed metric explains why this is the paper's w-RMSE or replaces it.",
        },
        {
            "gate_id": "mechanism_support",
            "status": "supported_locally",
            "evidence": f"high-rating DU={high['du_net']} Linear={high['linear_ucb']}; gap={high['linear_minus_du']}",
            "decision": "Keep the reward-region mechanism claim as support only.",
            "unlock_condition": "Recovered DU-Net backend confirms the same mechanism under exact paper settings.",
        },
        {
            "gate_id": "source_gate",
            "status": rerun["status"],
            "evidence": f"{rerun['evidence']}; open_P0={'; '.join(p0_open)}",
            "decision": "Do not start another local proxy experiment.",
            "unlock_condition": "Author code, supplement, exact metric, exact split/config, or backend config appears.",
        },
        {
            "gate_id": "split_preprocessing_gate",
            "status": gcmc_decision["status"],
            "evidence": gcmc_decision["evidence"],
            "decision": "GCMC/Monti provenance is useful but does not unlock MovieLens Figure 4a.",
            "unlock_condition": "MovieLens-specific split/preprocessing artifact or seed mapping is recovered.",
        },
        {
            "gate_id": "invalid_next_action",
            "status": "blocked",
            "evidence": "A new 20-seed proxy sweep would reuse guessed metric/split/backend assumptions.",
            "decision": "Freeze broad proxy sweeps because they would not increase paper faithfulness.",
            "unlock_condition": "A source-backed 3-seed diagnostic first improves a primary blocker.",
        },
        {
            "gate_id": "valid_next_action",
            "status": protocol_precondition["status"],
            "evidence": (
                f"next_queue_item={next_item['priority'] if next_item else 'n/a'}; "
                f"diagnostic={diagnostic_scope['requirement']}"
            ),
            "decision": "After a trigger, run only the minimal MovieLens Figure 4a 3-seed diagnostic.",
            "unlock_condition": promotion["pass_rule"],
        },
        {
            "gate_id": "execution_decision",
            "status": "gate_closed_wait_for_trigger",
            "evidence": "All local audits are integrated; no P0 executable-detail trigger is present.",
            "decision": "Stop local experiment expansion and wait for primary source or approved dependency trigger.",
            "unlock_condition": "Any P0 trigger becomes available or the user explicitly approves a validated external mechanism dependency.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| gate | status | evidence | decision | unlock condition |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gate_id']} | {row['status']} | {row['evidence']} | "
            f"{row['decision']} | {row['unlock_condition']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = row_by(rows, "gate_id", "execution_decision")
    report = f"""# Execution Gate Closure Audit

Generated by `experiments/build_execution_gate_closure_audit.py`.

## Current Progress

The current Figure 4a reproduction is a strong partial reproduction. The local
DU-Net-style result is directionally consistent with Linear UCB under the
reward-weighted and high-rating evaluations, but it does not match the paper's
absolute w-RMSE scale or full-test ordering.

## Gate Table

{markdown_table(rows)}

## Analysis

The user's fallback criterion has been handled locally: when exact DU-Net is not
matched, the local result is compared against Linear UCB. That comparison is
directionally consistent only in the reward-weighted/high-rating region, while
full RMSE still reverses. Recent PDF, URL, public-source, MovieLens split,
official-fold, and GCMC/Monti provenance audits did not produce a source-backed
trigger that would make another broad proxy sweep scientifically stronger.

## Decision

Status: `{decision['status']}`.

{decision['decision']} The next executable experiment remains the minimal
MovieLens Figure 4a 3-seed diagnostic, and it should run only after an unlock
condition in the table is satisfied.

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
