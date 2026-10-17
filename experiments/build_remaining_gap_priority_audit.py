"""Prioritize the remaining gaps for exact Figure 4a reproduction."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SOURCE_PACKET = TABLE_DIR / "paper_source_request_packet.csv"
CLAIM_INDEX = TABLE_DIR / "reproduction_claim_evidence_index.csv"
METRIC_FORMULA = TABLE_DIR / "figure4a_metric_formula_visual_audit.csv"
PUBLIC_RESCAN = TABLE_DIR / "public_source_rescan_audit.csv"
OFFICIAL_FOLDS = TABLE_DIR / "movielens_100k_svdpp_official_folds_sensitivity_fold_gaps.csv"

OUT_CSV = TABLE_DIR / "remaining_gap_priority_audit.csv"
OUT_REPORT = REPORT_DIR / "remaining_gap_priority_audit.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def build_rows() -> list[dict[str, str]]:
    source = read_rows(SOURCE_PACKET)
    claims = read_rows(CLAIM_INDEX)
    metric = read_rows(METRIC_FORMULA)
    public = read_rows(PUBLIC_RESCAN)
    folds = read_rows(OFFICIAL_FOLDS)

    source_gap = row_by(claims, "claim_id", "source_gap")
    exact_metric = row_by(source, "request_id", "exact_figure4a_metric")
    split = row_by(source, "request_id", "exact_split_and_preprocessing")
    backend = row_by(source, "request_id", "du_net_backend_config")
    linear = row_by(source, "request_id", "linear_ucb_adapter")
    author = row_by(source, "request_id", "author_code_or_supplement")
    dependency = row_by(source, "request_id", "validated_external_dependency_approval")
    metric_standard = row_by(metric, "metric_interpretation", "standard_rmse_indicator_mask")
    metric_literal = row_by(metric, "metric_interpretation", "page28_literal_cardinality_outside_sqrt")
    public_decision = row_by(public, "check_id", "execution_decision")
    fold_w = [float(row["linear_gap_favoring_du"]) for row in folds if row["metric"] == "w_rmse"]
    fold_mean = sum(fold_w) / len(fold_w)

    return [
        {
            "gap_id": "author_code_or_supplement",
            "priority": "P0",
            "current_status": author["current_status"],
            "latest_evidence": public_decision["evidence"],
            "local_resolution_possible": "no",
            "impact_if_resolved": "Could replace proxy DU-Net/backend assumptions and define exact rerun path.",
            "next_action": "Request or receive official code/supplement/reproduction package.",
        },
        {
            "gap_id": "exact_figure4a_metric",
            "priority": "P0",
            "current_status": exact_metric["current_status"],
            "latest_evidence": (
                f"page literal DU={metric_literal['du_net']} scale={metric_literal['scale_vs_paper_du']}x; "
                f"standard-RMSE gap={metric_standard['linear_minus_du']}"
            ),
            "local_resolution_possible": "no",
            "impact_if_resolved": "Would determine whether the paper scale/order is a metric-definition issue.",
            "next_action": "Obtain exact weighting, normalization, mask, and aggregation rule.",
        },
        {
            "gap_id": "exact_split_and_preprocessing",
            "priority": "P0",
            "current_status": split["current_status"],
            "latest_evidence": f"Official u1-u5 proxy mean w-RMSE gap={fold_mean:.6f}; split-only fix is not robust.",
            "local_resolution_possible": "no",
            "impact_if_resolved": "Would replace guessed/random masks and candidate sets with paper-faithful data.",
            "next_action": "Obtain split files, initial observed set, candidate mask, item filtering, and seed protocol.",
        },
        {
            "gap_id": "du_net_backend_config",
            "priority": "P0",
            "current_status": backend["current_status"],
            "latest_evidence": source_gap["evidence"],
            "local_resolution_possible": "no",
            "impact_if_resolved": "Would test whether missing DU-Net architecture/DLS/entropy/SVD++ settings explain the gap.",
            "next_action": "Obtain backend config or approved validated mechanism implementation.",
        },
        {
            "gap_id": "linear_ucb_adapter",
            "priority": "P1",
            "current_status": linear["current_status"],
            "latest_evidence": "Public contextualbandits confirms comparator family only, not the exact paper adapter.",
            "local_resolution_possible": "partial",
            "impact_if_resolved": "Would make the second-best comparator closer to the paper's Linear UCB row.",
            "next_action": "Obtain features, reward transform, regularization, alpha schedule, and update protocol.",
        },
        {
            "gap_id": "validated_external_dependency_approval",
            "priority": "P1",
            "current_status": dependency["current_status"],
            "latest_evidence": "No approval is recorded; dependency-free proxies remain frozen.",
            "local_resolution_possible": "requires_approval",
            "impact_if_resolved": "Would allow a bounded mechanism diagnostic when author source remains unavailable.",
            "next_action": "If explicitly approved, run smallest 3-seed diagnostic before any 20-seed run.",
        },
        {
            "gap_id": "execution_decision",
            "priority": "decision",
            "current_status": "blocked_by_missing_source",
            "latest_evidence": "All P0 executable-details gaps remain unresolved.",
            "local_resolution_possible": "no",
            "impact_if_resolved": "Defines whether more local execution is scientifically justified.",
            "next_action": "Do not run another proxy sweep; wait for source/metric/split/backend trigger.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| gap | priority | status | local resolution | evidence | next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gap_id']} | {row['priority']} | {row['current_status']} | "
            f"{row['local_resolution_possible']} | {row['latest_evidence']} | {row['next_action']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    p0_open = [
        row["gap_id"]
        for row in rows
        if row["priority"] == "P0" and row["current_status"] not in {"available", "defined"}
    ]
    report = f"""# Remaining Gap Priority Audit

Generated by `experiments/build_remaining_gap_priority_audit.py`.

## Current Progress

The reproduction remains a strong partial reproduction. Recent PDF formula,
method-detail, official-fold, and public-source audits narrowed the uncertainty,
but they did not convert the local proxy into an exact Figure 4a reproduction.

## Open P0 Gaps

{", ".join(p0_open)}

## Priority Table

{markdown_table(rows)}

## Decision

The highest-value next step is not another local proxy sweep. It is recovery of
source-backed executable details. If any P0 gap is resolved, run only the
smallest MovieLens Figure 4a 3-seed faithful diagnostic first; promote to 20
seeds only if that diagnostic improves the reward-weighted gap or resolves the
full-RMSE/scale mismatch.

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
