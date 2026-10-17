"""Run a small official-fold sensitivity diagnostic for MovieLens Figure 4a.

This is intentionally a local proxy diagnostic, not a paper-faithful rerun. It
uses the bundled MovieLens `u1.base/u1.test` split to test whether moving from
the current seeded random split protocol to an official MovieLens fold changes
the DU-Net vs MF-centered Linear UCB conclusion.
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

PREFIX = "movielens_100k_svdpp_official_u1_sensitivity_3seeds"
OUT_METRICS = TABLE_DIR / f"{PREFIX}_metrics.csv"
OUT_SUMMARY = TABLE_DIR / f"{PREFIX}_final_summary.csv"
OUT_GAPS = TABLE_DIR / f"{PREFIX}_gaps.csv"
OUT_REPORT = REPORT_DIR / f"{PREFIX}_report.md"


SEEDS = [0, 1, 2]
VARIANTS = [
    {"variant": "du_net", "policy": "du_net", "overrides": {}},
    {"variant": "myopic", "policy": "myopic", "overrides": {}},
    {"variant": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "overrides": {"ucb_alpha": 0.02}},
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


def summarize_final(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, group in final.groupby("variant", sort=False):
        row: dict[str, float | str] = {
            "variant": variant,
            "policy": str(group["policy"].iloc[0]),
        }
        for col in ["cumulative_reward", "rmse", "w_rmse", "sigmoid_reward_w_rmse", "observed_size"]:
            values = group[col].dropna().to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_gaps(summary: pd.DataFrame) -> list[dict[str, str]]:
    du = summary[summary["variant"] == "du_net"].iloc[0]
    linear = summary[summary["variant"] == "mf_linear_ucb_a002"].iloc[0]
    myopic = summary[summary["variant"] == "myopic"].iloc[0]
    rows = []
    for metric in ["w_rmse", "rmse", "sigmoid_reward_w_rmse", "cumulative_reward"]:
        lower_is_better = metric != "cumulative_reward"
        du_value = float(du[metric])
        linear_value = float(linear[metric])
        myopic_value = float(myopic[metric])
        if lower_is_better:
            linear_minus_du = linear_value - du_value
            myopic_minus_du = myopic_value - du_value
            interpretation = "DU-Net better if comparator minus DU-Net is positive."
        else:
            linear_minus_du = du_value - linear_value
            myopic_minus_du = du_value - myopic_value
            interpretation = "DU-Net better if DU-Net minus comparator is positive."
        rows.append(
            {
                "metric": metric,
                "du_net": f"{du_value:.6f}",
                "mf_linear_ucb_a002": f"{linear_value:.6f}",
                "myopic": f"{myopic_value:.6f}",
                "linear_gap_favoring_du": f"{linear_minus_du:.6f}",
                "myopic_gap_favoring_du": f"{myopic_minus_du:.6f}",
                "interpretation": interpretation,
            }
        )
    return rows


def write_gaps(rows: list[dict[str, str]]) -> None:
    with OUT_GAPS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def markdown_gaps(gaps: list[dict[str, str]]) -> str:
    lines = [
        "| metric | DU-Net | MF Linear UCB | Myopic | Linear gap favoring DU | Myopic gap favoring DU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gaps:
        lines.append(
            f"| {row['metric']} | {row['du_net']} | {row['mf_linear_ucb_a002']} | {row['myopic']} | "
            f"{row['linear_gap_favoring_du']} | {row['myopic_gap_favoring_du']} |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, gaps: list[dict[str, str]]) -> None:
    weighted_gap = float(next(row for row in gaps if row["metric"] == "w_rmse")["linear_gap_favoring_du"])
    rmse_gap = float(next(row for row in gaps if row["metric"] == "rmse")["linear_gap_favoring_du"])
    report = f"""# MovieLens Official u1 Fold Sensitivity Diagnostic

Generated by `experiments/run_movielens_official_fold_sensitivity.py`.

## Current Progress

This is a local proxy diagnostic, not a paper-faithful rerun. It uses
`u1.base/u1.test` with seeds 0-2 for initial-set sampling and compares DU-Net,
MF-centered Linear UCB (`alpha=0.02`), and Myopic under the existing local
SVD++/DU-Net-style backend.

## Final Summary

{markdown_summary(summary)}

## Gaps

{markdown_gaps(gaps)}

## Analysis

The official u1 fold does not explain the Figure 4a distance by itself.
The local reward-weighted ordering is {'DU-Net-favored' if weighted_gap > 0 else 'not DU-Net-favored'}
against MF-centered Linear UCB, with a gap of `{weighted_gap:.6f}`. The full
RMSE ordering is {'DU-Net-favored' if rmse_gap > 0 else 'not DU-Net-favored'},
with a gap of `{rmse_gap:.6f}`. This remains a sensitivity check around the
current proxy implementation, not evidence of exact paper reproduction.

## Decision

Do not promote this to a full 20-seed rerun unless a source-backed trigger
appears or an explicit decision is made to run a local-only sensitivity suite.
The main next step remains exact metric/split/backend recovery.

CSV artifacts:

- `{OUT_SUMMARY.relative_to(ROOT)}`
- `{OUT_GAPS.relative_to(ROOT)}`
- `{OUT_METRICS.relative_to(ROOT)}`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_DIR)

    dataset = load_movielens_100k(RAW_DIR)
    train = read_fold(RAW_DIR / "u1.base")
    test = read_fold(RAW_DIR / "u1.test")
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
    for seed in SEEDS:
        split = make_split(train, test, seed)
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
                seed=seed,
            )
            metrics.insert(0, "variant", variant["variant"])
            metrics.insert(1, "seed", seed)
            metrics.insert(2, "fold", "u1")
            metrics_parts.append(metrics)

    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    final = metrics_df.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    summary.to_csv(OUT_SUMMARY, index=False)
    gaps = build_gaps(summary)
    write_gaps(gaps)
    plot_metric(metrics_df, "w_rmse", "w-RMSE")
    plot_metric(metrics_df, "cumulative_reward", "Cumulative centered reward")
    write_report(summary, gaps)
    print(f"Wrote metrics: {OUT_METRICS}")
    print(summary.sort_values("variant").to_string(index=False))
    print(pd.DataFrame(gaps).to_string(index=False))


if __name__ == "__main__":
    main()
