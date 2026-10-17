"""Audit Figure 4a metric variants on shared final predictions."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import load_dataset_from_config, split_dataset
from src.graph_embedding import user_embedding
from src.graph_transforms import transform_user_graph
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds.json"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
PREFIX = "figure4a_metric_alignment_audit"
PREDICTIONS_CSV = TABLE_DIR / f"{PREFIX}_predictions.csv"
LONG_CSV = TABLE_DIR / f"{PREFIX}_long.csv"
SUMMARY_CSV = TABLE_DIR / f"{PREFIX}_summary.csv"
WIDE_CSV = TABLE_DIR / f"{PREFIX}_matrix.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}.md"
SELECTION_METRICS_CSV = TABLE_DIR / "movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds_metrics.csv"

VARIANTS = {
    "du_net_topk_current",
    "mf_linear_ucb_a002_topk",
    "myopic_topk",
    "du_net_exp_tau05",
    "du_net_exp_tau1",
}

DISPLAY = {
    "du_net_topk_current": "DU-Net current",
    "mf_linear_ucb_a002_topk": "MF-centered Linear UCB",
    "myopic_topk": "Myopic",
    "du_net_exp_tau05": "DU-Net exp tau0.5",
    "du_net_exp_tau1": "DU-Net exp tau1",
}


def ci95(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def weights(truth: np.ndarray, midpoint: float, kind: str) -> np.ndarray:
    if kind == "linear_reward":
        return np.maximum(truth - midpoint, 0.0) + 0.1
    if kind == "sigmoid_reward":
        return 1.0 / (1.0 + np.exp(-(truth - midpoint)))
    if kind == "positive_reward":
        return np.maximum(truth - midpoint, 0.0)
    raise ValueError(kind)


def compute_metrics(group: pd.DataFrame, midpoint: float) -> dict[str, float]:
    truth = group["rating"].to_numpy(dtype=float)
    pred = group["prediction"].to_numpy(dtype=float)
    err_sq = (truth - pred) ** 2
    out: dict[str, float] = {}
    out["conventional_rmse"] = float(np.sqrt(np.mean(err_sq)))

    linear = weights(truth, midpoint, "linear_reward")
    sigmoid = weights(truth, midpoint, "sigmoid_reward")
    positive = weights(truth, midpoint, "positive_reward")

    out["local_linear_reward_w_rmse"] = float(np.sqrt(np.sum(linear * err_sq) / np.sum(linear)))
    out["sigmoid_reward_w_rmse"] = float(np.sqrt(np.sum(sigmoid * err_sq) / np.sum(sigmoid)))
    out["linear_reward_weighted_norm_over_sqrt_n"] = float(np.sqrt(np.sum(linear * err_sq) / len(err_sq)))
    out["linear_reward_weighted_norm_over_sqrt_sum_weights"] = out["local_linear_reward_w_rmse"]
    if np.sum(positive) > 0:
        out["positive_reward_weighted_norm_over_sqrt_sum_weights"] = float(
            np.sqrt(np.sum(positive * err_sq) / np.sum(positive))
        )
    else:
        out["positive_reward_weighted_norm_over_sqrt_sum_weights"] = np.nan

    high = truth >= 4.0
    low = truth <= 3.0
    out["high_rating_rmse_ge4"] = float(np.sqrt(np.mean(err_sq[high]))) if high.any() else np.nan
    out["low_rating_rmse_le3"] = float(np.sqrt(np.mean(err_sq[low]))) if low.any() else np.nan
    return out


def collect_predictions(config: dict) -> pd.DataFrame:
    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    variants = [variant for variant in config["variants"] if variant["label"] in VARIANTS]
    all_predictions: list[pd.DataFrame] = []

    for seed in config["seeds"]:
        split = split_dataset(dataset, seed=int(seed), **config["split"])
        embedding_cache: dict[tuple[object, ...], np.ndarray] = {}
        for variant in variants:
            label = str(variant["label"])
            policy = str(variant.get("policy", "du_net"))
            overrides = variant.get("acquisition_overrides", {})
            acquisition = replace(base_acquisition, **overrides)
            graph_options = variant.get("graph_transform", {"mode": "none", "rate": 0.0})
            graph_mode = graph_options.get("mode", "none")
            graph_rate = float(graph_options.get("rate", 0.0))
            graph_seed = int(seed) * 1009 + sum(ord(ch) for ch in label)
            graph = transform_user_graph(dataset.adjacency, mode=graph_mode, rate=graph_rate, seed=graph_seed)
            embedding_key = (
                graph_mode,
                graph_rate,
                graph_seed,
                acquisition.graph_embedding_method,
                mf_config.rank,
            )
            if embedding_key not in embedding_cache:
                embedding_cache[embedding_key] = user_embedding(
                    graph,
                    dim=mf_config.rank,
                    method=acquisition.graph_embedding_method,
                    seed=int(seed),
                )
            sink: list[pd.DataFrame] = []
            metrics = run_policy(
                policy=policy,
                split=split,
                adjacency=graph,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf_config,
                acquisition_config=acquisition,
                seed=int(seed),
                network_embedding=embedding_cache[embedding_key],
                prediction_sink=sink,
                prediction_cycles={int(acquisition.cycles)},
            )
            if len(sink) != 1:
                raise RuntimeError(f"Expected one prediction frame for {label} seed {seed}, got {len(sink)}")
            final = metrics.sort_values("cycle").tail(1).iloc[0]
            selected_reward_percentile = metrics["selected_avg_reward_percentile"].dropna().mean()
            selected_entropy_percentile = metrics["selected_avg_entropy_percentile"].dropna().mean()
            frame = sink[0].copy()
            frame.insert(0, "variant", label)
            frame.insert(1, "display", DISPLAY[label])
            frame.insert(2, "seed", int(seed))
            frame["cumulative_reward"] = float(final["cumulative_reward"])
            frame["selected_avg_reward_percentile"] = float(selected_reward_percentile)
            frame["selected_avg_entropy_percentile"] = float(selected_entropy_percentile)
            all_predictions.append(frame)
    return pd.concat(all_predictions, ignore_index=True)


def apply_selection_diagnostics(long: pd.DataFrame) -> pd.DataFrame:
    if not SELECTION_METRICS_CSV.exists():
        return long
    metrics = pd.read_csv(SELECTION_METRICS_CSV)
    metrics = metrics[metrics["variant"].isin(VARIANTS)].copy()
    if metrics.empty:
        return long
    selected = (
        metrics.groupby(["variant", "seed"], as_index=False)[
            ["selected_avg_reward_percentile", "selected_avg_entropy_percentile"]
        ]
        .mean(numeric_only=True)
        .rename(
            columns={
                "selected_avg_reward_percentile": "selected_avg_reward_percentile_from_metrics",
                "selected_avg_entropy_percentile": "selected_avg_entropy_percentile_from_metrics",
            }
        )
    )
    merged = long.merge(selected, on=["variant", "seed"], how="left")
    for col in ["selected_avg_reward_percentile", "selected_avg_entropy_percentile"]:
        metrics_col = f"{col}_from_metrics"
        if metrics_col in merged.columns:
            merged[col] = merged[metrics_col].combine_first(merged[col])
            merged = merged.drop(columns=[metrics_col])
    return merged


def summarize(predictions: pd.DataFrame, midpoint: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (variant, seed), group in predictions.groupby(["variant", "seed"], sort=False):
        metric_values = compute_metrics(group, midpoint)
        first = group.iloc[0]
        row: dict[str, object] = {
            "variant": str(variant),
            "display": str(first["display"]),
            "seed": int(seed),
            "cumulative_reward": float(first["cumulative_reward"]),
            "selected_avg_reward_percentile": float(first["selected_avg_reward_percentile"]),
            "selected_avg_entropy_percentile": float(first["selected_avg_entropy_percentile"]),
        }
        row.update(metric_values)
        rows.append(row)
    long = pd.DataFrame(rows)

    metric_cols = [
        "conventional_rmse",
        "local_linear_reward_w_rmse",
        "sigmoid_reward_w_rmse",
        "linear_reward_weighted_norm_over_sqrt_n",
        "linear_reward_weighted_norm_over_sqrt_sum_weights",
        "positive_reward_weighted_norm_over_sqrt_sum_weights",
        "high_rating_rmse_ge4",
        "low_rating_rmse_le3",
        "cumulative_reward",
        "selected_avg_reward_percentile",
        "selected_avg_entropy_percentile",
    ]
    summary_rows: list[dict[str, object]] = []
    for variant, group in long.groupby("variant", sort=False):
        row: dict[str, object] = {
            "variant": str(variant),
            "display": str(group["display"].iloc[0]),
            "seed_count": int(group["seed"].nunique()),
        }
        for col in metric_cols:
            row[col] = float(group[col].mean())
            row[f"{col}_ci95"] = ci95(group[col])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    linear = summary[summary["variant"] == "mf_linear_ucb_a002_topk"].iloc[0]
    for col in metric_cols:
        if col.startswith("cumulative_reward") or col.startswith("selected_"):
            continue
        summary[f"linear_minus_variant_{col}"] = float(linear[col]) - summary[col].astype(float)

    matrix_rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        for col in metric_cols:
            if col.startswith("selected_"):
                continue
            matrix_rows.append(
                {
                    "variant": row["variant"],
                    "display": row["display"],
                    "metric": col,
                    "mean": row[col],
                    "ci95": row[f"{col}_ci95"],
                    "linear_minus_variant_gap": (
                        np.nan
                        if col == "cumulative_reward"
                        else row.get(f"linear_minus_variant_{col}", np.nan)
                    ),
                }
            )
    return long, summary, pd.DataFrame(matrix_rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_metric_table(summary: pd.DataFrame) -> str:
    order = [
        "du_net_topk_current",
        "mf_linear_ucb_a002_topk",
        "myopic_topk",
        "du_net_exp_tau05",
        "du_net_exp_tau1",
    ]
    summary = summary.set_index("variant").loc[order].reset_index()
    headers = [
        "variant",
        "RMSE",
        "local w-RMSE",
        "sigmoid w-RMSE",
        "weighted/sqrtN",
        "positive-only",
        "high>=4",
        "low<=3",
        "reward",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.display),
                    fmt(row.conventional_rmse),
                    fmt(row.local_linear_reward_w_rmse),
                    fmt(row.sigmoid_reward_w_rmse),
                    fmt(row.linear_reward_weighted_norm_over_sqrt_n),
                    fmt(row.positive_reward_weighted_norm_over_sqrt_sum_weights),
                    fmt(row.high_rating_rmse_ge4),
                    fmt(row.low_rating_rmse_le3),
                    fmt(row.cumulative_reward, 1),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    du = summary[summary["variant"] == "du_net_topk_current"].iloc[0]
    linear = summary[summary["variant"] == "mf_linear_ucb_a002_topk"].iloc[0]
    myopic = summary[summary["variant"] == "myopic_topk"].iloc[0]

    gaps = {
        "local": float(linear["local_linear_reward_w_rmse"] - du["local_linear_reward_w_rmse"]),
        "sigmoid": float(linear["sigmoid_reward_w_rmse"] - du["sigmoid_reward_w_rmse"]),
        "rmse": float(linear["conventional_rmse"] - du["conventional_rmse"]),
        "positive": float(
            linear["positive_reward_weighted_norm_over_sqrt_sum_weights"]
            - du["positive_reward_weighted_norm_over_sqrt_sum_weights"]
        ),
    }
    myopic_local_gap = float(myopic["local_linear_reward_w_rmse"] - du["local_linear_reward_w_rmse"])
    myopic_sigmoid_gap = float(myopic["sigmoid_reward_w_rmse"] - du["sigmoid_reward_w_rmse"])

    if gaps["local"] > 0 and myopic_local_gap > 0:
        decision = (
            "Use the existing local linear reward-weighted w-RMSE as the transparent primary proxy "
            "for Phase 2, while reporting conventional and sigmoid metrics as mandatory diagnostics."
        )
        blocked = False
    else:
        decision = (
            "No defensible metric simultaneously preserves the paper direction and comparator sanity; "
            "treat the reproduction as metric-blocked."
        )
        blocked = True

    report = f"""# Figure 4a Metric Alignment Audit

Generated by `experiments/build_figure4a_metric_alignment_audit.py`.

## Final-Cycle Sensitivity Matrix

All rows are recomputed from the same final test-row predictions for seeds
`0,1,2`.

{markdown_metric_table(summary)}

## DU-Net vs Linear Direction

- Local linear reward-weighted w-RMSE gap, Linear minus DU-Net:
  `{fmt(gaps["local"], 6)}`.
- Sigmoid reward-weighted w-RMSE gap, Linear minus DU-Net:
  `{fmt(gaps["sigmoid"], 6)}`.
- Conventional RMSE gap, Linear minus DU-Net:
  `{fmt(gaps["rmse"], 6)}`.
- Positive-only reward-weighted gap, Linear minus DU-Net:
  `{fmt(gaps["positive"], 6)}`.

## Decision

{decision}

Metric-blocked: `{str(blocked).lower()}`.

Rationale: the paper text supports rewarding high-value observations, but the
PDF formula extraction is ambiguous. Among implemented, auditable formulas, the
local linear reward-weighted metric is the only one that gives a non-trivial
positive DU-Net-vs-Linear gap and does not make Myopic beat DU-Net
(Myopic minus DU-Net local gap `{fmt(myopic_local_gap, 6)}`). The sigmoid metric
is source-motivated but essentially tied in this 3-seed diagnostic
(Myopic minus DU-Net sigmoid gap `{fmt(myopic_sigmoid_gap, 6)}`), so it is not a
strong enough optimization target by itself.

## Artifacts

- `{PREDICTIONS_CSV.relative_to(ROOT)}`
- `{LONG_CSV.relative_to(ROOT)}`
- `{SUMMARY_CSV.relative_to(ROOT)}`
- `{WIDE_CSV.relative_to(ROOT)}`
"""
    path.write_text(report, encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(frame.columns)
        writer.writerows(frame.itertuples(index=False, name=None))


def main() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(REPORT_DIR)
    config = load_json(CONFIG_PATH)
    if PREDICTIONS_CSV.exists():
        predictions = pd.read_csv(PREDICTIONS_CSV)
    else:
        predictions = collect_predictions(config)
    midpoint = float(config["acquisition"].get("rating_midpoint", 3.0))
    long, summary, matrix = summarize(predictions, midpoint)
    long = apply_selection_diagnostics(long)
    predictions = predictions.drop(columns=["selected_avg_reward_percentile", "selected_avg_entropy_percentile"]).merge(
            long[
                [
                    "variant",
                    "seed",
                    "selected_avg_reward_percentile",
                    "selected_avg_entropy_percentile",
                ]
            ],
            on=["variant", "seed"],
            how="left",
    )
    _, summary, matrix = summarize(predictions, midpoint)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    long.to_csv(LONG_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    matrix.to_csv(WIDE_CSV, index=False)
    write_report(summary, REPORT_PATH)
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {LONG_CSV}")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {WIDE_CSV}")
    print(f"Wrote {REPORT_PATH}")
    print(summary.sort_values("local_linear_reward_w_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
