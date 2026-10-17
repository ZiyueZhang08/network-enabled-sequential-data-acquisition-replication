"""Build the 20-seed MovieLens Figure 4a sigmoid-reward metric confirmation."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SUMMARY_PATH = TABLE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_final_summary.csv"
METRICS_PATH = TABLE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_metrics.csv"
OUT_CSV = TABLE_DIR / "figure4a_reward_weight_metric_20seeds.csv"
OUT_SEED_CSV = TABLE_DIR / "figure4a_reward_weight_metric_20seeds_paired_seed_gaps.csv"
OUT_REPORT = REPORT_DIR / "figure4a_reward_weight_metric_20seeds.md"

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


def f(value: object) -> float:
    return float(value)


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summary_rows() -> list[dict[str, object]]:
    summary = pd.read_csv(SUMMARY_PATH)
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        variant = str(row.variant)
        visual = PAPER_VISUAL[variant]
        rows.append(
            {
                "variant": variant,
                "method": DISPLAY_NAME[variant],
                "cumulative_reward": f(row.cumulative_reward),
                "cumulative_reward_ci95": f(row.cumulative_reward_ci95),
                "local_reward_weighted_w_rmse": f(row.w_rmse),
                "local_reward_weighted_w_rmse_ci95": f(row.w_rmse_ci95),
                "sigmoid_reward_w_rmse": f(row.sigmoid_reward_w_rmse),
                "sigmoid_reward_w_rmse_ci95": f(row.sigmoid_reward_w_rmse_ci95),
                "conventional_rmse": f(row.rmse),
                "conventional_rmse_ci95": f(row.rmse_ci95),
                "paper_visual_w_rmse": visual,
                "sigmoid_ratio_to_paper_visual": f(row.sigmoid_reward_w_rmse) / visual,
                "local_weighted_ratio_to_paper_visual": f(row.w_rmse) / visual,
                "conventional_rmse_ratio_to_paper_visual": f(row.rmse) / visual,
            }
        )
    return sorted(rows, key=lambda item: f(item["sigmoid_reward_w_rmse"]))


def paired_seed_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(METRICS_PATH)
    final = metrics[metrics["cycle"] == metrics["cycle"].max()].copy()
    rows: list[dict[str, object]] = []
    for seed, group in final.groupby("seed", sort=True):
        by_variant = {str(row.variant): row for row in group.itertuples(index=False)}
        du = by_variant["du_net"]
        linear = by_variant["mf_linear_ucb_a002"]
        myopic = by_variant["myopic"]
        rows.append(
            {
                "seed": int(seed),
                "sigmoid_gap_linear_minus_du": f(linear.sigmoid_reward_w_rmse) - f(du.sigmoid_reward_w_rmse),
                "local_weighted_gap_linear_minus_du": f(linear.w_rmse) - f(du.w_rmse),
                "conventional_rmse_gap_linear_minus_du": f(linear.rmse) - f(du.rmse),
                "sigmoid_gap_myopic_minus_du": f(myopic.sigmoid_reward_w_rmse) - f(du.sigmoid_reward_w_rmse),
                "du_net_sigmoid_reward_w_rmse": f(du.sigmoid_reward_w_rmse),
                "linear_sigmoid_reward_w_rmse": f(linear.sigmoid_reward_w_rmse),
                "myopic_sigmoid_reward_w_rmse": f(myopic.sigmoid_reward_w_rmse),
            }
        )
    return rows


def aggregate_gap(rows: list[dict[str, object]], key: str) -> tuple[float, float, int]:
    values = [f(row[key]) for row in rows]
    wins = sum(1 for value in values if value > 0)
    return statistics.fmean(values), ci95(values), wins


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


def write_report(rows: list[dict[str, object]], seed_rows: list[dict[str, object]], path: Path) -> None:
    du = next(row for row in rows if row["variant"] == "du_net")
    linear = next(row for row in rows if row["variant"] == "mf_linear_ucb_a002")
    sigmoid_gap, sigmoid_gap_ci, sigmoid_wins = aggregate_gap(seed_rows, "sigmoid_gap_linear_minus_du")
    local_gap, local_gap_ci, local_wins = aggregate_gap(seed_rows, "local_weighted_gap_linear_minus_du")
    rmse_gap, rmse_gap_ci, rmse_wins = aggregate_gap(seed_rows, "conventional_rmse_gap_linear_minus_du")

    report = f"""# Figure 4a Reward-Weighted Metric 20-Seed Confirmation

Generated by `experiments/build_figure4a_reward_weight_metric_20seeds.py`.

## Summary

The 3-seed sigmoid reward-weighted diagnostic did **not** stabilize at 20
seeds. Under `sigmoid_reward_w_rmse`, MF-centered Linear UCB is slightly lower
than DU-Net:

- DU-Net: {fmt(du["sigmoid_reward_w_rmse"])} +/- {fmt(du["sigmoid_reward_w_rmse_ci95"])}
- MF-centered Linear UCB: {fmt(linear["sigmoid_reward_w_rmse"])} +/- {fmt(linear["sigmoid_reward_w_rmse_ci95"])}
- paired gap `Linear - DU`: {fmt(sigmoid_gap, 6)} +/- {fmt(sigmoid_gap_ci, 6)}

Positive paired gaps mean DU-Net is lower/better. The 20-seed paired gap is
slightly negative, so the sigmoid weighting does not support a final
DU-Net-over-Linear-UCB Figure 4a claim.

## Final-Cycle Table

{markdown_table(rows)}

## Paired Direction Checks

| metric | paired gap Linear - DU | 95% CI | DU-Net seed wins |
|---|---:|---:|---:|
| sigmoid reward-weighted w-RMSE | {fmt(sigmoid_gap, 6)} | {fmt(sigmoid_gap_ci, 6)} | {sigmoid_wins}/20 |
| local reward-weighted w-RMSE | {fmt(local_gap, 6)} | {fmt(local_gap_ci, 6)} | {local_wins}/20 |
| conventional RMSE | {fmt(rmse_gap, 6)} | {fmt(rmse_gap_ci, 6)} | {rmse_wins}/20 |

## Interpretation

This confirms that metric choice alone is not enough. The current local
DU-Net proxy is ahead of MF-centered Linear UCB under the original local
reward-weighted metric, but not under conventional RMSE and not under the
paper-reward sigmoid-weighted metric at 20 seeds. The absolute scale remains
far from the paper Figure 4a visual endpoint: DU-Net's sigmoid-weighted value is
{fmt(du["sigmoid_ratio_to_paper_visual"], 3)}x of the paper visual estimate.

## Decision

Queue item 26 is complete. The next step should shift from metric variants to
fidelity of the model/mechanism itself: audit MovieLens preprocessing and the
matrix-completion backend, especially rating scaling/clipping and whether the
paper's noisy matrix-completion setup is materially different from the current
SVD++ proxy.

CSV artifacts:

- `{OUT_CSV.relative_to(ROOT)}`
- `{OUT_SEED_CSV.relative_to(ROOT)}`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = summary_rows()
    seed_rows = paired_seed_rows()
    write_csv(rows, OUT_CSV)
    write_csv(seed_rows, OUT_SEED_CSV)
    write_report(rows, seed_rows, OUT_REPORT)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SEED_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
