"""Recompute latest Figure 4a metric variants from saved final predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PREDICTIONS = TABLE_DIR / "movielens_100k_official_u1_best_candidate_rating_bins_20seeds_predictions.csv"
OUT_CSV = TABLE_DIR / "figure4a_latest_metric_reconciliation.csv"
OUT_GAPS = TABLE_DIR / "figure4a_latest_metric_reconciliation_gaps.csv"
OUT_REPORT = REPORT_DIR / "figure4a_latest_metric_reconciliation.md"

DU_VARIANT = "graph_late_lambda_half_start16"
COMPARATORS = ("mf_linear_ucb_a002", "myopic")
VARIANT_LABELS = {
    DU_VARIANT: "DU-Net best candidate",
    "mf_linear_ucb_a002": "MF-centered Linear UCB proxy",
    "myopic": "Myopic",
}


def fmt(value: object, digits: int = 6) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_values(frame: pd.DataFrame) -> pd.Series:
    truth = frame["rating"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=float)
    err = truth - pred
    linear_weight = np.maximum(truth - 3.0, 0.0) + 0.1
    positive_weight = np.maximum(truth - 3.0, 0.0)
    sigmoid_weight = 1.0 / (1.0 + np.exp(-(truth - 3.0)))
    high_mask = truth >= 4.0
    low_mask = truth <= 3.0
    rating5_mask = truth == 5.0
    return pd.Series(
        {
            "rows": len(frame),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "sum_norm_weighted_w_rmse": float(np.sqrt(np.sum(linear_weight * err**2) / np.sum(linear_weight))),
            "linear_reward_norm_w_rmse": float(np.sqrt(np.sum((linear_weight * err) ** 2) / len(truth))),
            "positive_reward_norm_w_rmse": float(np.sqrt(np.sum((positive_weight * err) ** 2) / len(truth))),
            "sigmoid_reward_w_rmse": float(np.sqrt(np.sum(sigmoid_weight * err**2) / np.sum(sigmoid_weight))),
            "high_rating_rmse_ge4": float(np.sqrt(np.mean(err[high_mask] ** 2))),
            "low_rating_rmse_le3": float(np.sqrt(np.mean(err[low_mask] ** 2))),
            "rating5_rmse": float(np.sqrt(np.mean(err[rating5_mask] ** 2))),
        }
    )


def metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = predictions.groupby("variant", sort=False).apply(metric_values, include_groups=False).reset_index()
    summary.insert(1, "label", summary["variant"].map(VARIANT_LABELS).fillna(summary["variant"]))
    return summary


def metric_gaps(summary: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rmse",
        "sum_norm_weighted_w_rmse",
        "linear_reward_norm_w_rmse",
        "positive_reward_norm_w_rmse",
        "sigmoid_reward_w_rmse",
        "high_rating_rmse_ge4",
        "low_rating_rmse_le3",
        "rating5_rmse",
    ]
    by_variant = summary.set_index("variant")
    by_seed = seed_summary.set_index(["variant", "seed"])
    du = by_variant.loc[DU_VARIANT]
    rows = []
    for comparator in COMPARATORS:
        comp = by_variant.loc[comparator]
        for metric in metrics:
            comp_seed = by_seed.loc[comparator][metric]
            du_seed = by_seed.loc[DU_VARIANT][metric]
            paired = comp_seed - du_seed
            rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "comparator_minus_du": float(comp[metric] - du[metric]),
                    "positive_favors_du": bool(comp[metric] - du[metric] > 0.0),
                    "du_seed_wins": int((paired > 0.0).sum()),
                    "seed_count": int(paired.notna().sum()),
                    "paired_gap_ci95": float(1.96 * paired.std(ddof=1) / np.sqrt(paired.notna().sum())),
                }
            )
    return pd.DataFrame(rows)


def markdown_metrics(summary: pd.DataFrame) -> str:
    lines = [
        "| variant | label | RMSE | sum-norm w-RMSE | linear reward-norm | positive reward-norm | sigmoid w-RMSE | high>=4 RMSE | low<=3 RMSE | rating-5 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.label} | {fmt(row.rmse)} | "
            f"{fmt(row.sum_norm_weighted_w_rmse)} | {fmt(row.linear_reward_norm_w_rmse)} | "
            f"{fmt(row.positive_reward_norm_w_rmse)} | {fmt(row.sigmoid_reward_w_rmse)} | "
            f"{fmt(row.high_rating_rmse_ge4)} | {fmt(row.low_rating_rmse_le3)} | {fmt(row.rating5_rmse)} |"
        )
    return "\n".join(lines)


def markdown_gaps(gaps: pd.DataFrame) -> str:
    lines = [
        "| comparator | metric | comparator - DU | DU wins seeds | interpretation |",
        "|---|---|---:|---:|---|",
    ]
    for row in gaps.itertuples(index=False):
        direction = "favors DU" if row.positive_favors_du else "favors comparator"
        lines.append(
            f"| {row.comparator} | {row.metric} | {fmt(row.comparator_minus_du)} | "
            f"{row.du_seed_wins}/{row.seed_count} | {direction} |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, gaps: pd.DataFrame) -> None:
    linear_primary = gaps[
        (gaps["comparator"] == "mf_linear_ucb_a002") & (gaps["metric"] == "linear_reward_norm_w_rmse")
    ].iloc[0]
    linear_sigmoid = gaps[
        (gaps["comparator"] == "mf_linear_ucb_a002") & (gaps["metric"] == "sigmoid_reward_w_rmse")
    ].iloc[0]
    linear_rmse = gaps[(gaps["comparator"] == "mf_linear_ucb_a002") & (gaps["metric"] == "rmse")].iloc[0]
    report = f"""# Figure 4a Latest Metric Reconciliation

Generated by `experiments/build_figure4a_latest_metric_reconciliation.py`.

This report recomputes final-cycle metrics from the saved 20-seed test-row
predictions for the frozen best candidate `{DU_VARIANT}`, MF-centered Linear
UCB proxy, and Myopic.

## Summary

The latest best candidate is strong only under reward-tail-weighted norms. Its
primary linear reward-norm gap against the MF-centered Linear UCB proxy is
`{fmt(linear_primary.comparator_minus_du)}` with `{linear_primary.du_seed_wins}/{linear_primary.seed_count}`
seed wins. Under sigmoid weighting, the comparison is essentially a tie and
slightly favors the comparator (`{fmt(linear_sigmoid.comparator_minus_du)}`).
Conventional RMSE still favors the comparator (`{fmt(linear_rmse.comparator_minus_du)}`).

## Metric Table

{markdown_metrics(summary)}

## Paired Direction Checks

Positive gaps mean the comparator has higher error than DU-Net, so positive
values favor DU-Net.

{markdown_gaps(gaps)}

## Decision

Keep `linear_reward_norm_w_rmse` as the transparent primary local proxy because
it matches the high-reward mechanism behind the current Figure 4a candidate.
Report `sigmoid_reward_w_rmse` and conventional `rmse` as mandatory caveats.
Do not call the result an exact Figure 4a reproduction unless source-backed
metric, split, backend, and Linear UCB details become available.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions["variant"].isin((DU_VARIANT, *COMPARATORS))].copy()
    summary = metric_summary(predictions)
    seed_summary = (
        predictions.groupby(["variant", "seed"], sort=False)
        .apply(metric_values, include_groups=False)
        .reset_index()
    )
    gaps = metric_gaps(summary, seed_summary)
    summary.to_csv(OUT_CSV, index=False)
    gaps.to_csv(OUT_GAPS, index=False)
    write_report(summary, gaps)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_GAPS}")
    print(f"Wrote {OUT_REPORT}")
    print(summary.to_string(index=False))
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
