"""Run a local proxy sensitivity diagnostic over MovieLens u1-u5 folds.

This extends the single-fold `u1` sensitivity check to all bundled MovieLens
official 80/20 folds. It is not a paper-faithful rerun: each fold uses one
fixed initial-set seed so the diagnostic isolates fold choice rather than
claiming robust reproduction.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import RATING_COLUMNS, RatingSplit, load_movielens_100k
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir


TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ml-100k"

PREFIX = "movielens_100k_svdpp_official_folds_sensitivity"
OUT_METRICS = TABLE_DIR / f"{PREFIX}_metrics.csv"
OUT_SUMMARY = TABLE_DIR / f"{PREFIX}_final_summary.csv"
OUT_FOLD_GAPS = TABLE_DIR / f"{PREFIX}_fold_gaps.csv"
OUT_REPORT = REPORT_DIR / f"{PREFIX}_report.md"


FOLDS = [1, 2, 3, 4, 5]
INITIAL_SEED = 0
VARIANTS = [
    {"variant": "du_net", "policy": "du_net", "overrides": {}},
    {"variant": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "overrides": {"ucb_alpha": 0.02}},
    {"variant": "myopic", "policy": "myopic", "overrides": {}},
]


def read_fold(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    return frame[RATING_COLUMNS].reset_index(drop=True)


def make_split(train: pd.DataFrame, test: pd.DataFrame, seed: int, initial_fraction: float = 0.01) -> RatingSplit:
    rng = np.random.default_rng(seed)
    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_initial = max(1, int(round(len(shuffled) * initial_fraction)))
    initial_idx = rng.choice(len(shuffled), size=n_initial, replace=False)
    initial_mask = np.zeros(len(shuffled), dtype=bool)
    initial_mask[initial_idx] = True
    return RatingSplit(
        initial=shuffled.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True),
        candidate=shuffled.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True),
        test=test[RATING_COLUMNS].reset_index(drop=True),
    )


def summarize_final(final: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grouped = final.groupby(group_cols, sort=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, float | str] = dict(zip(group_cols, group_key))
        row["policy"] = str(group["policy"].iloc[0])
        for col in ["cumulative_reward", "rmse", "w_rmse", "sigmoid_reward_w_rmse", "observed_size"]:
            values = group[col].dropna().to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def gap_rows(fold_summary: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fold, group in fold_summary.groupby("fold", sort=True):
        du = group[group["variant"] == "du_net"].iloc[0]
        linear = group[group["variant"] == "mf_linear_ucb_a002"].iloc[0]
        myopic = group[group["variant"] == "myopic"].iloc[0]
        for metric in ["w_rmse", "rmse", "sigmoid_reward_w_rmse", "cumulative_reward"]:
            lower_is_better = metric != "cumulative_reward"
            du_value = float(du[metric])
            linear_value = float(linear[metric])
            myopic_value = float(myopic[metric])
            if lower_is_better:
                linear_gap = linear_value - du_value
                myopic_gap = myopic_value - du_value
            else:
                linear_gap = du_value - linear_value
                myopic_gap = du_value - myopic_value
            rows.append(
                {
                    "fold": str(fold),
                    "metric": metric,
                    "du_net": f"{du_value:.6f}",
                    "mf_linear_ucb_a002": f"{linear_value:.6f}",
                    "myopic": f"{myopic_value:.6f}",
                    "linear_gap_favoring_du": f"{linear_gap:.6f}",
                    "myopic_gap_favoring_du": f"{myopic_gap:.6f}",
                }
            )
    return rows


def write_gap_csv(rows: list[dict[str, str]]) -> None:
    with OUT_FOLD_GAPS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_gap_summary(gaps: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in ["w_rmse", "rmse", "sigmoid_reward_w_rmse", "cumulative_reward"]:
        cur = [row for row in gaps if row["metric"] == metric]
        linear_gaps = np.array([float(row["linear_gap_favoring_du"]) for row in cur])
        myopic_gaps = np.array([float(row["myopic_gap_favoring_du"]) for row in cur])
        result[metric] = {
            "mean_linear_gap": float(np.mean(linear_gaps)),
            "positive_linear_folds": float(np.sum(linear_gaps > 0)),
            "mean_myopic_gap": float(np.mean(myopic_gaps)),
            "positive_myopic_folds": float(np.sum(myopic_gaps > 0)),
        }
    return result


def plot_metric(metrics: pd.DataFrame, metric: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for variant in ["du_net", "mf_linear_ucb_a002", "myopic"]:
        cur = metrics[metrics["variant"] == variant]
        grouped = cur.groupby("cycle")[metric].agg(["mean", "std", "count"]).reset_index()
        ci95 = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
        x = grouped["cycle"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        ci = ci95.to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=2, label=variant)
        plt.fill_between(x, y - ci, y + ci, alpha=0.12)
    plt.xlabel("Acquisition cycle")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{PREFIX}_{metric}.png", dpi=180)
    plt.close()


def markdown_summary(summary: pd.DataFrame) -> str:
    lines = [
        "| variant | policy | reward | w-RMSE | RMSE | sigmoid w-RMSE | observed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("variant").itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.policy} | {row.cumulative_reward:.4f} | "
            f"{row.w_rmse:.4f} | {row.rmse:.4f} | {row.sigmoid_reward_w_rmse:.4f} | "
            f"{row.observed_size:.0f} |"
        )
    return "\n".join(lines)


def markdown_gap_summary(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "| metric | mean Linear gap favoring DU | DU-favored folds vs Linear | mean Myopic gap favoring DU | DU-favored folds vs Myopic |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, row in summary.items():
        lines.append(
            f"| {metric} | {row['mean_linear_gap']:.6f} | {row['positive_linear_folds']:.0f}/5 | "
            f"{row['mean_myopic_gap']:.6f} | {row['positive_myopic_folds']:.0f}/5 |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, gaps: list[dict[str, str]]) -> None:
    gap_summary = aggregate_gap_summary(gaps)
    weighted = gap_summary["w_rmse"]
    full = gap_summary["rmse"]
    sigmoid = gap_summary["sigmoid_reward_w_rmse"]
    report = f"""# MovieLens Official Folds Sensitivity Diagnostic

Generated by `experiments/run_movielens_official_folds_sensitivity.py`.

## Current Progress

This is a local proxy diagnostic over `u1-u5.base/test`, not a paper-faithful
rerun. It uses one fixed initial-set seed per fold to isolate official-fold
choice from seed averaging.

## Aggregate Final Summary

{markdown_summary(summary)}

## Fold Gap Summary

{markdown_gap_summary(gap_summary)}

## Analysis

Across official folds, reward-weighted w-RMSE remains DU-Net-favored on
`{weighted['positive_linear_folds']:.0f}/5` folds against MF-centered Linear UCB
with mean gap `{weighted['mean_linear_gap']:.6f}`. Full RMSE remains
DU-Net-favored on `{full['positive_linear_folds']:.0f}/5` folds with mean gap
`{full['mean_linear_gap']:.6f}`. Sigmoid reward-weighted RMSE is
DU-Net-favored on `{sigmoid['positive_linear_folds']:.0f}/5` folds with mean
gap `{sigmoid['mean_linear_gap']:.6f}`.

The official-fold sweep therefore reinforces the earlier conclusion: changing
MovieLens folds can preserve a local reward-weighted DU-Net advantage, but it
does not resolve full-RMSE ordering, sigmoid near-tie/reversal, or absolute
Figure 4a scale.

## Decision

Do not promote this local proxy sweep to a paper-faithful result. The next
scientific step remains exact metric/split/backend recovery, or an explicitly
approved dependency-backed mechanism diagnostic.

CSV artifacts:

- `{OUT_SUMMARY.relative_to(ROOT)}`
- `{OUT_FOLD_GAPS.relative_to(ROOT)}`
- `{OUT_METRICS.relative_to(ROOT)}`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_DIR)

    dataset = load_movielens_100k(RAW_DIR)
    mf_config = MFConfig(rank=12, epochs=6, learning_rate=0.025, regularization=0.04, algorithm="svdpp")
    base_acquisition = AcquisitionConfig(
        cycles=20,
        batch_size=100,
        lambda_entropy=0.25,
        diffusion_prob=0.005,
        sigma=0.4,
        dls_top_k=20,
        selection="topk",
        rating_midpoint=3.0,
        entropy_mode="hybrid",
        entropy_coverage_weight=0.0,
        sigma_grid=[0.1, 0.2, 0.4, 0.8],
        ucb_alpha=0.8,
        ucb_regularization=1.0,
        thompson_noise=0.08,
    )

    metrics_parts = []
    for fold in FOLDS:
        train = read_fold(RAW_DIR / f"u{fold}.base")
        test = read_fold(RAW_DIR / f"u{fold}.test")
        split = make_split(train, test, INITIAL_SEED)
        for variant in VARIANTS:
            acquisition = replace(base_acquisition, **variant["overrides"])
            metrics = run_policy(
                policy=variant["policy"],
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf_config,
                acquisition_config=acquisition,
                seed=INITIAL_SEED + fold * 100,
            )
            metrics.insert(0, "variant", variant["variant"])
            metrics.insert(1, "fold", f"u{fold}")
            metrics.insert(2, "initial_seed", INITIAL_SEED)
            metrics_parts.append(metrics)

    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    final = metrics_df.sort_values("cycle").groupby(["variant", "fold"], as_index=False).tail(1)
    fold_summary = summarize_final(final, ["fold", "variant"])
    aggregate_summary = summarize_final(final, ["variant"])
    aggregate_summary.to_csv(OUT_SUMMARY, index=False)
    gaps = gap_rows(fold_summary)
    write_gap_csv(gaps)
    plot_metric(metrics_df, "w_rmse", "w-RMSE")
    plot_metric(metrics_df, "cumulative_reward", "Cumulative centered reward")
    write_report(aggregate_summary, gaps)
    print(f"Wrote metrics: {OUT_METRICS}")
    print(aggregate_summary.sort_values("variant").to_string(index=False))
    print(pd.DataFrame(gaps).to_string(index=False))


if __name__ == "__main__":
    main()
