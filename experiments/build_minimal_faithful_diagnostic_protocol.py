"""Build the minimal protocol for a future paper-faithful diagnostic.

This artifact is intentionally not an experiment run. It records the smallest
diagnostic to execute once source-backed Figure 4a details become available.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
SOURCE_PACKET_CSV = TABLE_DIR / "paper_source_request_packet.csv"

OUT_CSV = TABLE_DIR / "minimal_faithful_diagnostic_protocol.csv"
OUT_REPORT = REPORT_DIR / "minimal_faithful_diagnostic_protocol.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def build_rows() -> list[dict[str, str]]:
    trigger_rows = read_rows(TRIGGER_CSV)
    distance_rows = read_rows(DISTANCE_CSV)
    source_rows = read_rows(SOURCE_PACKET_CSV)

    rerun = row_by(trigger_rows, "trigger", "rerun_decision")
    reward = row_by(distance_rows, "claim_area", "local_reward_weighted")
    full = row_by(distance_rows, "claim_area", "local_full_rmse")
    high = row_by(distance_rows, "claim_area", "local_high_rating_region")
    metric = row_by(source_rows, "request_id", "exact_figure4a_metric")
    split = row_by(source_rows, "request_id", "exact_split_and_preprocessing")
    backend = row_by(source_rows, "request_id", "du_net_backend_config")

    return [
        {
            "stage": "precondition",
            "status": "blocked" if rerun["status"] == "blocked_by_missing_source" else "ready",
            "requirement": "At least one P0 source-backed trigger is available.",
            "evidence": rerun["evidence"],
            "pass_rule": "Do not run diagnostic while rerun_decision is blocked_by_missing_source.",
        },
        {
            "stage": "source_validation",
            "status": "pending",
            "requirement": "Validate exact Figure 4a metric, split/preprocessing, and DU-Net backend details.",
            "evidence": (
                f"metric={metric['current_status']}; split={split['current_status']}; "
                f"backend={backend['current_status']}"
            ),
            "pass_rule": "Recovered material must map to code/config changes without guessed masks or guessed normalization.",
        },
        {
            "stage": "diagnostic_scope",
            "status": "defined",
            "requirement": "Run MovieLens Figure 4a only, seeds 0-2, comparing DU-Net-style, MF-centered Linear UCB, and Myopic.",
            "evidence": "Smallest diagnostic isolates the target Figure 4a gap before any 20-seed promotion.",
            "pass_rule": "All three policies must use the same recovered split, metric, backend settings, and cycle schedule.",
        },
        {
            "stage": "primary_metric_threshold",
            "status": "defined",
            "requirement": "Improve or explain the current reward-weighted DU-Net-vs-Linear-UCB gap.",
            "evidence": (
                f"current gap={reward['linear_minus_du']}; "
                f"paper_gap_ratio={float(reward['paper_gap_ratio']) * 100:.1f}%; "
                f"scale_ratio={float(reward['du_scale_ratio_to_paper']) * 100:.1f}%"
            ),
            "pass_rule": "Promotable if recovered mechanism increases the positive gap or explains why the paper scale differs.",
        },
        {
            "stage": "blocker_metric_threshold",
            "status": "defined",
            "requirement": "Address the current full-test RMSE reversal.",
            "evidence": f"current full-RMSE gap={full['linear_minus_du']}; direction_matches={full['direction_matches_paper']}",
            "pass_rule": "Promotable if full-RMSE ordering becomes DU-Net-favored, near-tied with source-backed explanation, or explicitly not the paper metric.",
        },
        {
            "stage": "mechanism_preservation",
            "status": "defined",
            "requirement": "Preserve the high-rating region support for DU-Net.",
            "evidence": f"current high-rating gap={high['linear_minus_du']}; paper_gap_ratio={float(high['paper_gap_ratio']) * 100:.1f}%",
            "pass_rule": "Recovered mechanism should not erase the high-rating/reward-region support unless it replaces the metric interpretation.",
        },
        {
            "stage": "promotion_decision",
            "status": "defined",
            "requirement": "Decide whether to promote from 3 seeds to 20 seeds.",
            "evidence": "A 20-seed rerun is expensive and should only follow a source-backed positive diagnostic.",
            "pass_rule": "Promote only if at least one primary blocker improves and no new contradiction is introduced.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| stage | status | requirement | evidence | pass rule |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['status']} | {row['requirement']} | "
            f"{row['evidence']} | {row['pass_rule']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Minimal Paper-Faithful Diagnostic Protocol

Generated by `experiments/build_minimal_faithful_diagnostic_protocol.py`.

## Current Progress

The reproduction remains a strong partial reproduction and is still blocked for
paper-faithful rerun. This protocol defines the smallest diagnostic to execute
after a real source-backed trigger appears.

## Protocol

{markdown_table(rows)}

## Execution Summary

1. Validate recovered source/metric/split/backend material first.
2. Run only MovieLens Figure 4a, seeds 0-2.
3. Compare DU-Net-style, MF-centered Linear UCB, and Myopic under the same
   recovered metric and split.
4. Promote to 20 seeds only if the diagnostic improves the current
   reward-weighted gap or resolves the full-RMSE/absolute-scale mismatch.

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
