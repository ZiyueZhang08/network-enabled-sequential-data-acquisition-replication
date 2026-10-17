"""Audit the Figure 4a metric formula recovered from the rendered PDF pages."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_DELIVERY = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
PROTOCOL_ALIGNMENT = TABLE_DIR / "movielens_protocol_alignment_audit.csv"

OUT_CSV = TABLE_DIR / "figure4a_metric_formula_visual_audit.csv"
OUT_REPORT = REPORT_DIR / "figure4a_metric_formula_visual_audit.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def f(value: str) -> float:
    return float(value)


def build_rows() -> list[dict[str, str]]:
    delivery = read_rows(FINAL_DELIVERY)
    protocol = read_rows(PROTOCOL_ALIGNMENT)
    test_size = int(row_by(protocol, "check", "official_u1_split")["local_value"].split("test=")[1].split(";")[0])

    paper = row_by(delivery, "claim_area", "paper_target")
    full = row_by(delivery, "claim_area", "local_full_rmse")
    reward = row_by(delivery, "claim_area", "local_reward_weighted")
    sigmoid = row_by(delivery, "claim_area", "sigmoid_reward_weighted")

    full_du = f(full["du_net"])
    full_linear = f(full["linear_ucb"])
    literal_du = full_du / math.sqrt(test_size)
    literal_linear = full_linear / math.sqrt(test_size)

    return [
        {
            "metric_interpretation": "paper_visual_target",
            "formula_or_source": "Figure 4a visual estimate",
            "du_net": paper["du_net"],
            "linear_ucb": paper["linear_ucb"],
            "linear_minus_du": paper["linear_minus_du"],
            "scale_vs_paper_du": "1.000000",
            "direction_matches_paper": "True",
            "interpretation": "Reference target only; values are visual estimates from Figure 4a.",
        },
        {
            "metric_interpretation": "page28_literal_cardinality_outside_sqrt",
            "formula_or_source": "1/|Omega_test| * sqrt(||Omega_test o (Y-Yhat)||_2^2)",
            "du_net": f"{literal_du:.6f}",
            "linear_ucb": f"{literal_linear:.6f}",
            "linear_minus_du": f"{literal_linear - literal_du:.6f}",
            "scale_vs_paper_du": f"{literal_du / f(paper['du_net']):.6f}",
            "direction_matches_paper": str(literal_linear - literal_du > 0),
            "interpretation": (
                "Literal reading with |Omega_test|=20000 produces near-zero scale, so it cannot explain the Figure 4a y-axis."
            ),
        },
        {
            "metric_interpretation": "standard_rmse_indicator_mask",
            "formula_or_source": "sqrt(mean((Y-Yhat)^2)) on test entries",
            "du_net": full["du_net"],
            "linear_ucb": full["linear_ucb"],
            "linear_minus_du": full["linear_minus_du"],
            "scale_vs_paper_du": f"{f(full['du_net']) / f(paper['du_net']):.6f}",
            "direction_matches_paper": "False",
            "interpretation": (
                "This is the most plausible executable reading of the page-28 formula, but it reverses DU-Net vs Linear UCB locally."
            ),
        },
        {
            "metric_interpretation": "local_reward_weighted_proxy",
            "formula_or_source": "sqrt(sum(local_reward_weights * error^2) / sum(local_reward_weights))",
            "du_net": reward["du_net"],
            "linear_ucb": reward["linear_ucb"],
            "linear_minus_du": reward["linear_minus_du"],
            "scale_vs_paper_du": f"{f(reward['du_net']) / f(paper['du_net']):.6f}",
            "direction_matches_paper": "True",
            "interpretation": (
                "Best local directional match, but it is not the page-28 formula and remains far below the paper scale."
            ),
        },
        {
            "metric_interpretation": "sigmoid_reward_weighted_proxy",
            "formula_or_source": "sqrt(sum(sigmoid(Y-3) * error^2) / sum(sigmoid(Y-3)))",
            "du_net": sigmoid["du_net"],
            "linear_ucb": sigmoid["linear_ucb"],
            "linear_minus_du": sigmoid["linear_minus_du"],
            "scale_vs_paper_du": f"{f(sigmoid['du_net']) / f(paper['du_net']):.6f}",
            "direction_matches_paper": "False",
            "interpretation": "Paper-like reward weighting is a near tie and does not recover robust DU-Net superiority.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| interpretation | DU-Net | Linear UCB | gap | scale vs paper DU | direction | note |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric_interpretation']} | {row['du_net']} | {row['linear_ucb']} | "
            f"{row['linear_minus_du']} | {row['scale_vs_paper_du']} | "
            f"{row['direction_matches_paper']} | {row['interpretation']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Figure 4a Metric Formula Visual Audit

Generated by `experiments/build_figure4a_metric_formula_visual_audit.py`.

## Current Progress

The rendered PDF pages clarify more than the earlier text-only audit: page 27
defines cumulative reward as a sigmoid-transformed sum, and page 28 provides a
w-RMSE expression for the test set. However, the expression still does not
recover the Figure 4a result when mapped to the local metrics.

Rendered evidence:

- `tmp/pdfs/figure4_metric_formula_page-27.png`
- `tmp/pdfs/figure4_metric_formula_page-28.png`

## Metric Interpretation Table

{markdown_table(rows)}

## Analysis

The page-28 formula is useful source evidence, but it does not clear the exact
metric blocker. A literal reading with cardinality outside the square root gives
near-zero values; the plausible standard-RMSE reading matches the local full
RMSE column, where Linear UCB is slightly better than DU-Net; and the local
reward-weighted metric is directionally consistent but is not the displayed
formula and remains far from the paper scale.

## Decision

Do not promote the reproduction to exact Figure 4a. The next executable step is
still gated on source-backed metric/split/backend details or an approved
validated mechanism dependency.

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
