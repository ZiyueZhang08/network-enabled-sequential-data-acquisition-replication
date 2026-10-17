"""Audit MovieLens preprocessing and backend fidelity for Figure 4a."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset_from_config, load_movielens_100k, split_dataset
from src.matrix_factorization import MFConfig, MatrixFactorization
from src.utils import load_json


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"
METRICS_PATH = TABLE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_metrics.csv"
SUMMARY_PATH = TABLE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_final_summary.csv"

OUT_AUDIT_CSV = TABLE_DIR / "figure4a_model_backend_fidelity_audit.csv"
OUT_DRIFT_CSV = TABLE_DIR / "figure4a_model_backend_metric_drift.csv"
OUT_PROBE_CSV = TABLE_DIR / "figure4a_backend_probe_initial_predictions.csv"
OUT_REPORT = REPORT_DIR / "figure4a_model_backend_fidelity_audit.md"

PAPER = {
    "users": 943,
    "items": 1681,
    "initial_density": 0.00086,
    "test_density": 0.0126,
    "du_net_figure4a_visual": 1.788,
    "linear_ucb_figure4a_visual": 1.814,
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


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def current_dataset_stats() -> list[dict[str, object]]:
    dataset = load_movielens_100k(ROOT / "data" / "raw" / "ml-100k")
    total_entries = dataset.num_users * dataset.num_items
    paper_entries = PAPER["users"] * PAPER["items"]
    config = load_json(CONFIG_PATH)
    split = split_dataset(dataset, seed=0, **config["split"])
    item_counts = dataset.ratings.groupby("item_id").size().sort_values()
    low_count_items = ",".join(str(int(item)) for item in item_counts.head(5).index)
    rows = [
        {
            "category": "preprocessing",
            "check": "users",
            "paper_value": PAPER["users"],
            "current_value": dataset.num_users,
            "current_to_paper": dataset.num_users / PAPER["users"],
            "diagnosis": "Current loader matches Table 1 user count.",
        },
        {
            "category": "preprocessing",
            "check": "items",
            "paper_value": PAPER["items"],
            "current_value": dataset.num_items,
            "current_to_paper": dataset.num_items / PAPER["items"],
            "diagnosis": f"Current loader keeps one extra MovieLens item; five sparsest remapped item ids are {low_count_items}.",
        },
        {
            "category": "preprocessing",
            "check": "initial_density_seed0",
            "paper_value": PAPER["initial_density"],
            "current_value": len(split.initial) / total_entries,
            "current_to_paper": (len(split.initial) / total_entries) / PAPER["initial_density"],
            "diagnosis": "Current default initial density is smaller than the paper Table 1 density.",
        },
        {
            "category": "preprocessing",
            "check": "test_density_seed0",
            "paper_value": PAPER["test_density"],
            "current_value": len(split.test) / total_entries,
            "current_to_paper": (len(split.test) / total_entries) / PAPER["test_density"],
            "diagnosis": "Current random 20% test density matches the paper closely.",
        },
        {
            "category": "preprocessing",
            "check": "matrix_size_entries",
            "paper_value": paper_entries,
            "current_value": total_entries,
            "current_to_paper": total_entries / paper_entries,
            "diagnosis": "The one-item mismatch changes total matrix size by only about 0.06%.",
        },
    ]
    return rows


def split_distribution_rows() -> list[dict[str, object]]:
    config = load_json(CONFIG_PATH)
    dataset = load_dataset_from_config(config)
    rows = []
    for seed in config["seeds"]:
        split = split_dataset(dataset, seed=int(seed), **config["split"])
        rows.append(
            {
                "seed": int(seed),
                "initial_count": len(split.initial),
                "candidate_count": len(split.candidate),
                "test_count": len(split.test),
                "initial_mean_rating": float(split.initial["rating"].mean()),
                "candidate_mean_rating": float(split.candidate["rating"].mean()),
                "test_mean_rating": float(split.test["rating"].mean()),
            }
        )
    out = []
    for key in ["initial_mean_rating", "candidate_mean_rating", "test_mean_rating"]:
        values = [f(row[key]) for row in rows]
        out.append(
            {
                "category": "rating_distribution",
                "check": key,
                "paper_value": "",
                "current_value": statistics.fmean(values),
                "current_to_paper": "",
                "diagnosis": f"20-seed mean {key.replace('_', ' ')}; CI95={ci95(values):.4f}.",
            }
        )
    return out


def metric_drift_rows() -> list[dict[str, object]]:
    metrics = pd.read_csv(METRICS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    rows: list[dict[str, object]] = []
    initial_count = 800
    acquired_count = 2000
    midpoint = 3.0
    for row in summary.itertuples(index=False):
        variant = str(row.variant)
        final = metrics[(metrics["variant"] == variant) & (metrics["cycle"] == metrics["cycle"].max())]
        start = metrics[(metrics["variant"] == variant) & (metrics["cycle"] == 0)]
        acquired_mean = midpoint + f(row.cumulative_reward) / acquired_count
        rows.append(
            {
                "variant": variant,
                "reward": f(row.cumulative_reward),
                "avg_acquired_rating": acquired_mean,
                "cycle0_local_w_rmse": float(start["w_rmse"].mean()),
                "cycle20_local_w_rmse": f(row.w_rmse),
                "cycle0_sigmoid_w_rmse": float(start["sigmoid_reward_w_rmse"].mean()),
                "cycle20_sigmoid_w_rmse": f(row.sigmoid_reward_w_rmse),
                "cycle0_conventional_rmse": float(start["rmse"].mean()),
                "cycle20_conventional_rmse": f(row.rmse),
                "local_w_rmse_change": f(row.w_rmse) - float(start["w_rmse"].mean()),
                "sigmoid_w_rmse_change": f(row.sigmoid_reward_w_rmse) - float(start["sigmoid_reward_w_rmse"].mean()),
                "conventional_rmse_change": f(row.rmse) - float(start["rmse"].mean()),
                "final_seed_count": int(final["seed"].nunique()),
            }
        )
    return sorted(rows, key=lambda row: str(row["variant"]))


def raw_predict(model: MatrixFactorization, users: np.ndarray, items: np.ndarray) -> np.ndarray:
    factors = model.user_representations()
    return (
        model.global_mean
        + model.user_bias[users]
        + model.item_bias[items]
        + np.sum(factors[users] * model.item_factors[items], axis=1)
    )


def backend_probe_rows() -> list[dict[str, object]]:
    config = load_json(CONFIG_PATH)
    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    rows = []
    for seed in [0, 1, 2]:
        split = split_dataset(dataset, seed=seed, **config["split"])
        model = MatrixFactorization(dataset.num_users, dataset.num_items, mf_config, seed=seed * 1009).fit(split.initial)
        users = split.test["user_id"].to_numpy(dtype=int)
        items = split.test["item_id"].to_numpy(dtype=int)
        truth = split.test["rating"].to_numpy(dtype=float)
        raw = raw_predict(model, users, items)
        clipped = np.clip(raw, 1.0, 5.0)
        rows.append(
            {
                "seed": seed,
                "train_count": len(split.initial),
                "test_count": len(split.test),
                "raw_prediction_min": float(raw.min()),
                "raw_prediction_max": float(raw.max()),
                "raw_prediction_mean": float(raw.mean()),
                "pct_predictions_clipped_low": float(np.mean(raw < 1.0)),
                "pct_predictions_clipped_high": float(np.mean(raw > 5.0)),
                "raw_rmse": float(np.sqrt(np.mean((truth - raw) ** 2))),
                "clipped_rmse": float(np.sqrt(np.mean((truth - clipped) ** 2))),
            }
        )
    return rows


def add_probe_summary(audit_rows: list[dict[str, object]], probe_rows: list[dict[str, object]]) -> None:
    for key in ["pct_predictions_clipped_low", "pct_predictions_clipped_high", "raw_rmse", "clipped_rmse"]:
        values = [f(row[key]) for row in probe_rows]
        audit_rows.append(
            {
                "category": "backend_probe_initial_only",
                "check": key,
                "paper_value": "",
                "current_value": statistics.fmean(values),
                "current_to_paper": "",
                "diagnosis": "Initial-only SVD++ probe across seeds 0-2; not a final policy metric.",
            }
        )


def markdown_audit_table(rows: list[dict[str, object]]) -> str:
    lines = ["| category | check | paper | current | current/paper | diagnosis |", "|---|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["category"]),
                    str(row["check"]),
                    fmt(row["paper_value"]),
                    fmt(row["current_value"]),
                    fmt(row["current_to_paper"]),
                    str(row["diagnosis"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def markdown_drift_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| variant | acquired avg rating | local w-RMSE change | sigmoid w-RMSE change | RMSE change | final local | final sigmoid | final RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['avg_acquired_rating'])} | "
            f"{fmt(row['local_w_rmse_change'])} | {fmt(row['sigmoid_w_rmse_change'])} | "
            f"{fmt(row['conventional_rmse_change'])} | {fmt(row['cycle20_local_w_rmse'])} | "
            f"{fmt(row['cycle20_sigmoid_w_rmse'])} | {fmt(row['cycle20_conventional_rmse'])} |"
        )
    return "\n".join(lines)


def write_report(
    audit_rows: list[dict[str, object]],
    drift_rows: list[dict[str, object]],
    probe_rows: list[dict[str, object]],
    path: Path,
) -> None:
    du = next(row for row in drift_rows if row["variant"] == "du_net")
    linear = next(row for row in drift_rows if row["variant"] == "mf_linear_ucb_a002")
    avg_clip_high = statistics.fmean(f(row["pct_predictions_clipped_high"]) for row in probe_rows)
    avg_clip_low = statistics.fmean(f(row["pct_predictions_clipped_low"]) for row in probe_rows)
    report = f"""# Figure 4a Model/Backend Fidelity Audit

Generated by `experiments/build_figure4a_model_backend_fidelity_audit.py`.

## Summary

Metric variants alone did not reproduce the paper's MovieLens Figure 4a
DU-Net-over-Linear-UCB ordering. This audit points to model/backend fidelity and
selection-induced rating-distribution shift as the next bottleneck.

Main findings:

- MovieLens preprocessing is close but not exact: current data has 943 users and
  1682 items, while the paper Table 1 reports 943 users and 1681 items.
- Test density is essentially aligned with the paper, but default initial density
  remains smaller than Table 1. A prior Table-1 initial-density diagnostic did
  not close the scale gap.
- All high-reward policies acquire very high-rating observations. DU-Net's
  average acquired rating is {fmt(du["avg_acquired_rating"])}; MF-centered
  Linear UCB's is {fmt(linear["avg_acquired_rating"])}. This improves
  reward-weighted metrics but can worsen conventional RMSE.
- The current backend is an in-house clipped SVD++ proxy. The paper describes a
  noisy matrix-completion model and says SVDpp is used to estimate user/item
  representations, but exact preprocessing, hyperparameter tuning, and prediction
  treatment are not fully specified locally.
- In an initial-only backend probe, clipping is negligible:
  {fmt(avg_clip_low * 100, 2)}% low-clipped and {fmt(avg_clip_high * 100, 2)}%
  high-clipped predictions on average across seeds 0-2.

## Fidelity Audit Table

{markdown_audit_table(audit_rows)}

## Metric Drift Across Acquisition

{markdown_drift_table(drift_rows)}

## Interpretation

The critical pattern is metric divergence. Local reward-weighted w-RMSE improves
substantially through acquisition, while conventional RMSE worsens. That means
the current acquisition policies are learning high-reward regions well but are
not improving global test-set accuracy in the way the paper Figure 4a appears to
show. Because sigmoid reward weighting also failed to stabilize the paper
ordering at 20 seeds, the next useful work is backend fidelity rather than more
metric variants.

## Decision

Queue item 27 is complete. The next execution branch should be a small
MovieLens backend-fidelity screen that tests paper-plausible backend choices:

1. reproduce the paper item count by dropping one sparse MovieLens item or
   confirming the paper count is a typo;
2. compare clipped versus unclipped prediction evaluation in a controlled
   diagnostic;
3. run a small rank/epoch/regularization SVD++ backend screen and check whether
   conventional RMSE and sigmoid weighted RMSE move toward the paper ordering.

CSV artifacts:

- `{OUT_AUDIT_CSV.relative_to(ROOT)}`
- `{OUT_DRIFT_CSV.relative_to(ROOT)}`
- `{OUT_PROBE_CSV.relative_to(ROOT)}`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows = current_dataset_stats() + split_distribution_rows()
    drift_rows = metric_drift_rows()
    probe_rows = backend_probe_rows()
    add_probe_summary(audit_rows, probe_rows)
    write_csv(audit_rows, OUT_AUDIT_CSV)
    write_csv(drift_rows, OUT_DRIFT_CSV)
    write_csv(probe_rows, OUT_PROBE_CSV)
    write_report(audit_rows, drift_rows, probe_rows, OUT_REPORT)
    print(f"Wrote {OUT_AUDIT_CSV}")
    print(f"Wrote {OUT_DRIFT_CSV}")
    print(f"Wrote {OUT_PROBE_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
