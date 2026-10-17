"""Build the MovieLens Figure 4a sigmoid-reward metric diagnostic report."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SUMMARY_PATH = TABLE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_3seeds_final_summary.csv"
METRICS_PATH = TABLE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_3seeds_metrics.csv"
OUT_CSV = TABLE_DIR / "figure4a_reward_weight_metric_diagnostic.csv"
OUT_REPORT = REPORT_DIR / "figure4a_reward_weight_metric_diagnostic.md"

PAPER_VISUAL = {
    "du_net": 1.788,
    "mf_linear_ucb_a002": 1.814,
    "myopic": 1.823,
}

DISPLAY_NAME = {
    "du_net": "DU-Net",
    "mf_linear_ucb_a002": "MF-centered Linear UCB",
    "myopic": "Myopic",
}


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_rows() -> list[dict[str, object]]:
    summary = pd.read_csv(SUMMARY_PATH)
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        variant = str(row.variant)
        sigmoid_value = float(row.sigmoid_reward_w_rmse)
        local_value = float(row.w_rmse)
        rmse_value = float(row.rmse)
        visual = PAPER_VISUAL[variant]
        rows.append(
            {
                "variant": variant,
                "method": DISPLAY_NAME[variant],
                "cumulative_reward": float(row.cumulative_reward),
                "cumulative_reward_ci95": float(row.cumulative_reward_ci95),
                "local_reward_weighted_w_rmse": local_value,
                "local_reward_weighted_w_rmse_ci95": float(row.w_rmse_ci95),
                "sigmoid_reward_w_rmse": sigmoid_value,
                "sigmoid_reward_w_rmse_ci95": float(row.sigmoid_reward_w_rmse_ci95),
                "conventional_rmse": rmse_value,
                "conventional_rmse_ci95": float(row.rmse_ci95),
                "paper_visual_w_rmse": visual,
                "sigmoid_ratio_to_paper_visual": sigmoid_value / visual,
                "local_weighted_ratio_to_paper_visual": local_value / visual,
                "conventional_rmse_ratio_to_paper_visual": rmse_value / visual,
            }
        )
    return sorted(rows, key=lambda row: float(row["sigmoid_reward_w_rmse"]))


def seed_winners() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_PATH)
    final = metrics[metrics["cycle"] == metrics["cycle"].max()].copy()
    final["rank"] = final.groupby("seed")["sigmoid_reward_w_rmse"].rank(method="first")
    winners = final[final["rank"] == 1].copy()
    return winners[["seed", "variant", "sigmoid_reward_w_rmse", "w_rmse", "rmse", "cumulative_reward"]].sort_values("seed")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "method",
        "reward",
        "sigmoid w-RMSE",
        "local w-RMSE",
        "RMSE",
        "paper visual",
        "sigmoid/visual",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    fmt(row["cumulative_reward"], 2),
                    fmt(row["sigmoid_reward_w_rmse"]),
                    fmt(row["local_reward_weighted_w_rmse"]),
                    fmt(row["conventional_rmse"]),
                    fmt(row["paper_visual_w_rmse"]),
                    fmt(row["sigmoid_ratio_to_paper_visual"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def winner_table(winners: pd.DataFrame) -> str:
    lines = [
        "| seed | winner | sigmoid w-RMSE | local w-RMSE | RMSE | reward |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in winners.itertuples(index=False):
        lines.append(
            f"| {int(row.seed)} | {DISPLAY_NAME[str(row.variant)]} | "
            f"{float(row.sigmoid_reward_w_rmse):.4f} | {float(row.w_rmse):.4f} | "
            f"{float(row.rmse):.4f} | {float(row.cumulative_reward):.2f} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], winners: pd.DataFrame, path: Path) -> None:
    du = next(row for row in rows if row["variant"] == "du_net")
    linear = next(row for row in rows if row["variant"] == "mf_linear_ucb_a002")
    myopic = next(row for row in rows if row["variant"] == "myopic")
    gap_vs_linear = float(linear["sigmoid_reward_w_rmse"]) - float(du["sigmoid_reward_w_rmse"])
    gap_vs_myopic = float(myopic["sigmoid_reward_w_rmse"]) - float(du["sigmoid_reward_w_rmse"])
    du_seed_wins = int((winners["variant"] == "du_net").sum())
    report = f"""# Figure 4a Reward-Weighted Metric Diagnostic

Generated by `experiments/build_figure4a_reward_weight_metric_diagnostic.py`.

## Summary

This diagnostic adds `sigmoid_reward_w_rmse`, using the paper's reward transform
as a test-set weight: `sigmoid(rating - 3)`. It reruns the smallest MovieLens
3-seed Figure 4a comparison across DU-Net, MF-centered Linear UCB, and Myopic.

The metric weakly recovers the paper's DU-Net-over-Linear-UCB direction:
DU-Net is lower than MF-centered Linear UCB by {fmt(gap_vs_linear, 6)} and
lower than Myopic by {fmt(gap_vs_myopic, 6)}. However, the difference is tiny
relative to the 95% confidence intervals, so this is a weak diagnostic signal,
not a final claim.

DU-Net wins {du_seed_wins}/3 seeds by `sigmoid_reward_w_rmse`.

## Final-Cycle Table

{markdown_table(rows)}

## Per-Seed Winners

{winner_table(winners)}

## Interpretation

The metric-weighting hypothesis is partly supported: using the paper's
`sigmoid(rating - 3)` reward transform restores the desired ordering in the
3-seed mean. But it does not solve the absolute scale problem. DU-Net's
sigmoid-weighted value is {fmt(du["sigmoid_ratio_to_paper_visual"], 3)}x of the
paper Figure 4a visual endpoint.

The current evidence is therefore:

- local reward-weighted metric: DU-Net clearly lower than MF-centered Linear UCB;
- conventional RMSE reading: MF-centered Linear UCB slightly lower than DU-Net;
- sigmoid reward-weighted metric: DU-Net barely lower than MF-centered Linear UCB.

## Decision

Queue item 25 is complete. The next step should be a 20-seed confirmation of
`sigmoid_reward_w_rmse` on MovieLens Figure 4a. If DU-Net's advantage remains
tiny or disappears, metric choice alone is not enough and the next bottleneck is
matrix-completion / DU-Net mechanism fidelity.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    winners = seed_winners()
    write_csv(rows, OUT_CSV)
    write_report(rows, winners, OUT_REPORT)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
