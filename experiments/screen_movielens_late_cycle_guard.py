"""Screen late-cycle DU-Net guard schedules on MovieLens u1.

The script is intentionally diagnostic-only: it writes metrics, summaries, and
DU-vs-linear gaps as CSV tables without producing a narrative report.
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
DEFAULT_PREFIX = "movielens_100k_official_u1_late_cycle_guard_screen_5seeds"


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
        row: dict[str, float | str] = {"variant": str(variant), "policy": str(group["policy"].iloc[0])}
        for col in [
            "cumulative_reward",
            "rmse",
            "w_rmse",
            "sigmoid_reward_w_rmse",
            "mild_reward_w_rmse",
            "medium_reward_w_rmse",
            "linear_reward_norm_w_rmse",
            "positive_reward_norm_w_rmse",
            "observed_size",
        ]:
            values = group[col].dropna().to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_gaps(summary: pd.DataFrame, comparator: str = "mf_linear_ucb_a002") -> pd.DataFrame:
    if comparator not in set(summary["variant"]):
        return pd.DataFrame()
    comparator_row = summary[summary["variant"] == comparator].iloc[0]
    rows: list[dict[str, float | str]] = []
    for _, du in summary[summary["policy"] == "du_net"].iterrows():
        for metric in [
            "cumulative_reward",
            "rmse",
            "w_rmse",
            "sigmoid_reward_w_rmse",
            "mild_reward_w_rmse",
            "medium_reward_w_rmse",
            "linear_reward_norm_w_rmse",
            "positive_reward_norm_w_rmse",
        ]:
            du_value = float(du[metric])
            comparator_value = float(comparator_row[metric])
            if metric == "cumulative_reward":
                gap = du_value - comparator_value
            else:
                gap = comparator_value - du_value
            rows.append(
                {
                    "variant": str(du["variant"]),
                    "comparator": comparator,
                    "metric": metric,
                    "du_value": du_value,
                    "comparator_value": comparator_value,
                    "gap_favoring_du": gap,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 4, 5, 11, 19])
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(TABLE_DIR)
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
    variants = [
        {"variant": "graph_base", "policy": "du_net", "overrides": {}},
        {
            "variant": "graph_warm_start",
            "policy": "du_net",
            "overrides": {"warm_start_refit": True},
        },
        {
            "variant": "graph_mf_only",
            "policy": "du_net",
            "overrides": {},
            "mf_overrides": {"algorithm": "mf", "epochs": 8},
        },
        {
            "variant": "graph_svdpp_ep12",
            "policy": "du_net",
            "overrides": {},
            "mf_overrides": {"epochs": 12},
        },
        {
            "variant": "graph_svdpp_rank16_ep8",
            "policy": "du_net",
            "overrides": {},
            "mf_overrides": {"rank": 16, "epochs": 8, "learning_rate": 0.02, "regularization": 0.06},
        },
        {
            "variant": "late_lambda0_start10",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 10, "late_cycle_lambda_multiplier": 0.0},
        },
        {
            "variant": "late_lambda_half_start10",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 10, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "late_frontier95_start10",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 10, "late_cycle_frontier_quantile": 0.95},
        },
        {
            "variant": "late_frontier99_start10",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 10, "late_cycle_frontier_quantile": 0.99},
        },
        {
            "variant": "late_lambda0_start15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 15, "late_cycle_lambda_multiplier": 0.0},
        },
        {
            "variant": "late_lambda_half_start15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 15, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "late_frontier95_start15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 15, "late_cycle_frontier_quantile": 0.95},
        },
        {
            "variant": "late_frontier99_start15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 15, "late_cycle_frontier_quantile": 0.99},
        },
        {
            "variant": "reliability_shrink03",
            "policy": "du_net_shrinkage_dls",
            "overrides": {"dls_shrinkage": 0.3, "dls_shrinkage_mode": "reliability"},
        },
        {
            "variant": "reliability_shrink06",
            "policy": "du_net_shrinkage_dls",
            "overrides": {"dls_shrinkage": 0.6, "dls_shrinkage_mode": "reliability"},
        },
        {
            "variant": "reliability_shrink09",
            "policy": "du_net_shrinkage_dls",
            "overrides": {"dls_shrinkage": 0.9, "dls_shrinkage_mode": "reliability"},
        },
        {
            "variant": "adaptive_shrink06",
            "policy": "du_net_shrinkage_dls",
            "overrides": {"dls_shrinkage": 0.6, "dls_shrinkage_mode": "adaptive"},
        },
        {"variant": "lambda010", "policy": "du_net", "overrides": {"lambda_entropy": 0.10}},
        {"variant": "lambda050", "policy": "du_net", "overrides": {"lambda_entropy": 0.50}},
        {"variant": "sigma020", "policy": "du_net", "overrides": {"sigma": 0.20}},
        {"variant": "sigma080", "policy": "du_net", "overrides": {"sigma": 0.80}},
        {"variant": "diffusion000", "policy": "du_net", "overrides": {"diffusion_prob": 0.0}},
        {"variant": "diffusion010", "policy": "du_net", "overrides": {"diffusion_prob": 0.010}},
        {"variant": "diffusion030", "policy": "du_net", "overrides": {"diffusion_prob": 0.030}},
        {
            "variant": "neighbor_gain_p005",
            "policy": "du_net",
            "overrides": {"entropy_mode": "network_neighbor_prediction_gain", "diffusion_prob": 0.005},
        },
        {
            "variant": "neighbor_gain_p010",
            "policy": "du_net",
            "overrides": {"entropy_mode": "network_neighbor_prediction_gain", "diffusion_prob": 0.010},
        },
        {
            "variant": "neighbor_gain_p030",
            "policy": "du_net",
            "overrides": {"entropy_mode": "network_neighbor_prediction_gain", "diffusion_prob": 0.030},
        },
        {
            "variant": "neighbor_gain_l005",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "network_neighbor_prediction_gain",
                "diffusion_prob": 0.005,
                "lambda_entropy": 0.05,
            },
        },
        {
            "variant": "neighbor_gain_l010",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "network_neighbor_prediction_gain",
                "diffusion_prob": 0.005,
                "lambda_entropy": 0.10,
            },
        },
        {
            "variant": "neighbor_gain_reward_l005",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "network_neighbor_prediction_gain",
                "diffusion_prob": 0.005,
                "lambda_entropy": 0.05,
                "entropy_logdet_weighting": "reward",
                "entropy_logdet_weight_strength": 1.0,
            },
        },
        {
            "variant": "neighbor_gain_frontier95_l010",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "network_neighbor_prediction_gain",
                "diffusion_prob": 0.005,
                "lambda_entropy": 0.10,
                "entropy_reward_frontier_quantile": 0.95,
            },
        },
        {
            "variant": "hybrid_neighbor_w010",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "hybrid_neighbor_prediction_gain",
                "network_neighbor_gain_weight": 0.10,
            },
        },
        {
            "variant": "hybrid_neighbor_w025",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "hybrid_neighbor_prediction_gain",
                "network_neighbor_gain_weight": 0.25,
            },
        },
        {
            "variant": "hybrid_neighbor_w050",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "hybrid_neighbor_prediction_gain",
                "network_neighbor_gain_weight": 0.50,
            },
        },
        {"variant": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "overrides": {"ucb_alpha": 0.02}},
        {
            "variant": "mf_linear_ucb_a002_warm_start",
            "policy": "mf_linear_ucb",
            "overrides": {"ucb_alpha": 0.02, "warm_start_refit": True},
        },
    ]
    if args.variants:
        requested = set(args.variants)
        variants = [variant for variant in variants if variant["variant"] in requested]
        missing = requested - {variant["variant"] for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants requested: {sorted(missing)}")

    metrics_path = TABLE_DIR / f"{args.prefix}_metrics.csv"
    summary_path = TABLE_DIR / f"{args.prefix}_summary.csv"
    gaps_path = TABLE_DIR / f"{args.prefix}_gaps.csv"
    parts = []
    completed: set[tuple[str, int]] = set()
    wrote_header = False
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        parts.append(existing)
        completed = {
            (str(row.variant), int(row.seed))
            for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True
    for seed in args.seeds:
        split = make_split(train, test, seed)
        for variant in variants:
            if (variant["variant"], seed) in completed:
                continue
            acquisition = replace(base_acquisition, **variant["overrides"])
            variant_mf_config = replace(mf_config, **variant.get("mf_overrides", {}))
            metrics = run_policy(
                policy=variant["policy"],
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=variant_mf_config,
                acquisition_config=acquisition,
                seed=seed,
            )
            metrics.insert(0, "variant", variant["variant"])
            metrics.insert(1, "seed", seed)
            metrics.insert(2, "fold", "u1")
            parts.append(metrics)
            metrics.to_csv(metrics_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
            wrote_header = True

    metrics_df = pd.concat(parts, ignore_index=True)
    final = metrics_df.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    gaps = build_gaps(summary)
    summary.to_csv(summary_path, index=False)
    if not gaps.empty:
        gaps.to_csv(gaps_path, index=False)
    print(f"Wrote {metrics_path}")
    print(summary.sort_values("linear_reward_norm_w_rmse").to_string(index=False))
    if not gaps.empty:
        print(gaps[gaps["metric"].isin(["linear_reward_norm_w_rmse", "cumulative_reward"])].to_string(index=False))


if __name__ == "__main__":
    main()
