"""Diagnose rating-bin errors for the current best MovieLens candidate.

This script writes CSV diagnostics only. It stores final-cycle predictions and
summarizes errors by true rating so the Figure 4a gap can be explained without
producing a narrative report.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import RATING_COLUMNS, RatingSplit, load_movielens_100k
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir


RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
TABLE_DIR = ROOT / "outputs" / "tables"
DEFAULT_PREFIX = "movielens_100k_official_u1_best_candidate_rating_bins_20seeds"


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


def make_head_split(seed: int, initial_fraction: float = 0.01) -> RatingSplit:
    train = read_fold(RAW_DIR / "u1.base")
    test = read_fold(RAW_DIR / "u1.test")
    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_initial = max(1, int(round(len(shuffled) * initial_fraction)))
    initial_mask = np.zeros(len(shuffled), dtype=bool)
    initial_mask[:n_initial] = True
    return RatingSplit(
        initial=shuffled.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True),
        candidate=shuffled.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True),
        test=test[RATING_COLUMNS].reset_index(drop=True),
    )


def variant_catalog() -> list[dict[str, object]]:
    return [
        {"variant": "graph_base", "policy": "du_net", "overrides": {}},
        {
            "variant": "graph_late_lambda_half_start16",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 16, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "graph_late_half_start16_frontier95",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "late_cycle_frontier_quantile": 0.95,
            },
        },
        {"variant": "myopic", "policy": "myopic", "overrides": {}},
        {"variant": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "overrides": {"ucb_alpha": 0.02}},
    ]


def weighted_norm(truth: np.ndarray, pred: np.ndarray, midpoint: float = 3.0) -> float:
    weights = np.maximum(truth - midpoint, 0.0) + 0.1
    return float(np.sqrt(np.sum((weights * (truth - pred)) ** 2) / len(truth)))


def positive_weighted_norm(truth: np.ndarray, pred: np.ndarray, midpoint: float = 3.0) -> float:
    weights = np.maximum(truth - midpoint, 0.0)
    return float(np.sqrt(np.sum((weights * (truth - pred)) ** 2) / len(truth)))


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (variant, seed), group in predictions.groupby(["variant", "seed"], sort=False):
        truth = group["rating"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        err = pred - truth
        high = truth >= 4.0
        low = truth <= 3.0
        rows.append(
            {
                "variant": str(variant),
                "seed": int(seed),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "linear_reward_norm_w_rmse": weighted_norm(truth, pred),
                "positive_reward_norm_w_rmse": positive_weighted_norm(truth, pred),
                "high_rating_rmse_ge4": float(np.sqrt(np.mean((err[high]) ** 2))) if high.any() else np.nan,
                "low_rating_rmse_le3": float(np.sqrt(np.mean((err[low]) ** 2))) if low.any() else np.nan,
                "rating5_rmse": float(np.sqrt(np.mean((err[truth == 5.0]) ** 2))) if np.any(truth == 5.0) else np.nan,
                "rating4_rmse": float(np.sqrt(np.mean((err[truth == 4.0]) ** 2))) if np.any(truth == 4.0) else np.nan,
                "pred_mean": float(np.mean(pred)),
                "bias": float(np.mean(err)),
                "high_bias_ge4": float(np.mean(err[high])) if high.any() else np.nan,
                "low_bias_le3": float(np.mean(err[low])) if low.any() else np.nan,
            }
        )
    per_seed = pd.DataFrame(rows)

    summary_rows: list[dict[str, object]] = []
    metric_cols = [col for col in per_seed.columns if col not in {"variant", "seed"}]
    for variant, group in per_seed.groupby("variant", sort=False):
        row: dict[str, object] = {"variant": str(variant), "seed_count": int(group["seed"].nunique())}
        for col in metric_cols:
            values = group[col].dropna().to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    bin_rows: list[dict[str, object]] = []
    for (variant, rating), group in predictions.groupby(["variant", "rating"], sort=False):
        truth = group["rating"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        err = pred - truth
        bin_rows.append(
            {
                "variant": str(variant),
                "rating": float(rating),
                "row_count": int(len(group)),
                "pred_mean": float(np.mean(pred)),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "pct_below_rating": float(np.mean(pred < truth)),
            }
        )
    rating_bins = pd.DataFrame(bin_rows)
    return per_seed, summary, rating_bins


def build_pairwise_gaps(per_seed: pd.DataFrame, comparator: str = "mf_linear_ucb_a002") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = [col for col in per_seed.columns if col not in {"variant", "seed"}]
    comp = per_seed[per_seed["variant"] == comparator].set_index("seed")
    for variant, group in per_seed.groupby("variant", sort=False):
        if variant == comparator:
            continue
        candidate = group.set_index("seed")
        common = sorted(set(candidate.index) & set(comp.index))
        for metric in metric_cols:
            gap = float(comp.loc[common, metric].mean() - candidate.loc[common, metric].mean())
            rows.append(
                {
                    "variant": str(variant),
                    "comparator": comparator,
                    "metric": metric,
                    "candidate_value": float(candidate.loc[common, metric].mean()),
                    "comparator_value": float(comp.loc[common, metric].mean()),
                    "gap_favoring_candidate": gap,
                    "seed_count": len(common),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(TABLE_DIR)
    dataset = load_movielens_100k(RAW_DIR)
    mf_config = MFConfig(rank=12, epochs=6, learning_rate=0.025, regularization=0.04, algorithm="svdpp")
    base_acquisition = AcquisitionConfig(
        cycles=20,
        batch_size=100,
        lambda_entropy=0.25,
        diffusion_prob=0.005,
        sigma=0.4,
        dls_top_k=20,
        dls_neighbor_mode="graph_edges",
        selection="topk",
        rating_midpoint=3.0,
        entropy_mode="hybrid",
        entropy_coverage_weight=0.0,
        sigma_grid=(0.1, 0.2, 0.4, 0.8),
        ucb_alpha=0.8,
        ucb_regularization=1.0,
        thompson_noise=0.08,
    )
    variants = variant_catalog()
    if args.variants:
        requested = set(args.variants)
        variants = [variant for variant in variants if str(variant["variant"]) in requested]
        missing = requested - {str(variant["variant"]) for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants requested: {sorted(missing)}")

    predictions_path = TABLE_DIR / f"{args.prefix}_predictions.csv"
    per_seed_path = TABLE_DIR / f"{args.prefix}_per_seed.csv"
    summary_path = TABLE_DIR / f"{args.prefix}_summary.csv"
    rating_bins_path = TABLE_DIR / f"{args.prefix}_rating_bins.csv"
    gaps_path = TABLE_DIR / f"{args.prefix}_gaps.csv"

    parts: list[pd.DataFrame] = []
    completed: set[tuple[str, int]] = set()
    wrote_header = False
    if args.resume and predictions_path.exists():
        existing = pd.read_csv(predictions_path)
        parts.append(existing)
        completed = {
            (str(row.variant), int(row.seed))
            for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True

    for seed in args.seeds:
        split = make_head_split(seed)
        for variant in variants:
            key = (str(variant["variant"]), seed)
            if key in completed:
                continue
            sink: list[pd.DataFrame] = []
            acquisition = replace(base_acquisition, **variant["overrides"])
            run_policy(
                policy=str(variant["policy"]),
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf_config,
                acquisition_config=acquisition,
                seed=seed,
                prediction_sink=sink,
                prediction_cycles={acquisition.cycles},
            )
            if len(sink) != 1:
                raise RuntimeError(f"Expected one prediction frame for {variant['variant']} seed {seed}, got {len(sink)}")
            frame = sink[0].copy()
            frame.insert(0, "variant", str(variant["variant"]))
            frame.insert(1, "seed", seed)
            parts.append(frame)
            frame.to_csv(predictions_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
            wrote_header = True

    predictions = pd.concat(parts, ignore_index=True)
    per_seed, summary, rating_bins = summarize_predictions(predictions)
    gaps = build_pairwise_gaps(per_seed)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    rating_bins.to_csv(rating_bins_path, index=False)
    gaps.to_csv(gaps_path, index=False)

    print(f"Wrote {predictions_path}")
    print(summary.to_string(index=False))
    focus = gaps[gaps["metric"].isin(["linear_reward_norm_w_rmse", "high_rating_rmse_ge4", "low_rating_rmse_le3", "rating5_rmse", "bias"])]
    print(focus.to_string(index=False))
    print(rating_bins.sort_values(["rating", "variant"]).to_string(index=False))


if __name__ == "__main__":
    main()
