"""Build a quantitative distance scorecard for Figure 4a reproduction.

The scorecard compares frozen local DU-Net/Linear-UCB rows with the paper
Figure 4a visual target. It does not run new experiments.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SOURCE_CSV = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
OUT_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
OUT_REPORT = REPORT_DIR / "figure4a_distance_scorecard.md"


def read_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], claim_area: str) -> dict[str, str]:
    for row in rows:
        if row["claim_area"] == claim_area:
            return row
    raise KeyError(f"Missing claim_area={claim_area}")


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def build_scorecard() -> list[dict[str, str]]:
    rows = read_rows()
    paper = row_by(rows, "paper_target")
    paper_du = as_float(paper, "du_net")
    paper_linear = as_float(paper, "linear_ucb")
    paper_gap = as_float(paper, "linear_minus_du")
    paper_relative_improvement = paper_gap / paper_linear

    score_rows: list[dict[str, str]] = []
    for area in (
        "local_reward_weighted",
        "local_full_rmse",
        "local_high_rating_region",
        "local_low_rating_region",
        "sigmoid_reward_weighted",
    ):
        row = row_by(rows, area)
        du = as_float(row, "du_net")
        linear = as_float(row, "linear_ucb")
        gap = as_float(row, "linear_minus_du")
        sign_matches = gap > 0
        gap_ratio = gap / paper_gap if paper_gap else 0.0
        relative_improvement = gap / linear if linear else 0.0
        relative_improvement_ratio = (
            relative_improvement / paper_relative_improvement
            if paper_relative_improvement
            else 0.0
        )
        scale_ratio = du / paper_du if paper_du else 0.0
        scale_gap = du - paper_du

        if area == "local_reward_weighted":
            conclusion = (
                "Best reportable Figure 4a comparison: direction matches and "
                "relative DU-Net advantage is meaningfully close, but scale does not match."
            )
        elif area == "local_high_rating_region":
            conclusion = "Mechanism support: high-rating region matches the paper direction."
        elif area == "local_full_rmse":
            conclusion = "Exact-reproduction blocker: full-test RMSE reverses the paper ordering."
        elif area == "local_low_rating_region":
            conclusion = "Mechanism limit: low-rating region favors Linear UCB."
        else:
            conclusion = "Metric ambiguity check: effectively tied, not a stable DU-Net win."

        score_rows.append(
            {
                "claim_area": area,
                "status": row["status"],
                "du_net": f"{du:.6f}",
                "linear_ucb": f"{linear:.6f}",
                "linear_minus_du": f"{gap:.6f}",
                "direction_matches_paper": str(sign_matches),
                "paper_gap_ratio": f"{gap_ratio:.6f}",
                "du_scale_ratio_to_paper": f"{scale_ratio:.6f}",
                "du_absolute_gap_to_paper": f"{scale_gap:.6f}",
                "relative_improvement": f"{relative_improvement:.6f}",
                "relative_improvement_vs_paper": f"{relative_improvement_ratio:.6f}",
                "conclusion": conclusion,
            }
        )
    return score_rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| claim area | direction | gap vs paper gap | DU scale vs paper | relative improvement vs paper | conclusion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {claim_area} | {direction} | {gap_ratio} | {scale_ratio} | {relative_ratio} | {conclusion} |".format(
                claim_area=row["claim_area"],
                direction=row["direction_matches_paper"],
                gap_ratio=pct(float(row["paper_gap_ratio"])),
                scale_ratio=pct(float(row["du_scale_ratio_to_paper"])),
                relative_ratio=pct(float(row["relative_improvement_vs_paper"])),
                conclusion=row["conclusion"],
            )
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    reward = row_by(rows, "local_reward_weighted")
    full = row_by(rows, "local_full_rmse")
    high = row_by(rows, "local_high_rating_region")
    report = f"""# Figure 4a Reproduction Distance Scorecard

Generated by `experiments/build_figure4a_distance_scorecard.py`.

## Current Progress

The current Figure 4a result is a strong partial reproduction. The local
reward-weighted comparison recovers the paper's DU-Net-over-Linear-UCB direction,
but the absolute w-RMSE scale remains far from the paper visual endpoint and the
full-test RMSE ordering is reversed.

## Distance Summary

- Paper visual target: DU-Net `1.788`, Linear UCB `1.814`, Linear-minus-DU gap
  `0.026`.
- Local reward-weighted gap is `{float(reward['paper_gap_ratio']) * 100:.1f}%`
  of the paper gap and its relative DU-Net advantage is
  `{float(reward['relative_improvement_vs_paper']) * 100:.1f}%` of the paper's
  relative advantage.
- Local reward-weighted DU-Net scale is only
  `{float(reward['du_scale_ratio_to_paper']) * 100:.1f}%` of the paper DU-Net
  endpoint, so the absolute scale is not close.
- Full-test RMSE has a negative Linear-minus-DU gap
  `{float(full['linear_minus_du']):.4f}`, so it does not reproduce the paper
  ordering.
- High-rating regions remain supportive: the high-rating gap is
  `{float(high['paper_gap_ratio']) * 100:.1f}%` of the paper gap.

## Scorecard

{markdown_table(rows)}

## Decision

The result is close enough to report as directionally consistent against Linear
UCB under the local reward-weighted/high-rating interpretation, but not close
enough to claim Figure 4a reproduction. The next step remains gated on author
code, supplement, split/config files, exact w-RMSE definition, or an approved
validated mechanism dependency.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_scorecard()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
