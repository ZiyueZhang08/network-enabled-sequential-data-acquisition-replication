"""Screen MovieLens Figure 4a acquisition-protocol sensitivity.

This diagnostic writes CSV tables only. It checks whether the remaining local
DU-Net-vs-Linear gap is mainly driven by initial/candidate/test construction
rather than entropy or backend choices.
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
from src.data_loader import RATING_COLUMNS, RatingSplit, load_movielens_100k, split_ratings
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir


RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
TABLE_DIR = ROOT / "outputs" / "tables"
DEFAULT_PREFIX = "movielens_100k_protocol_sensitivity_3seeds"


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


def make_initial_candidate(
    train: pd.DataFrame,
    seed: int,
    initial_fraction: float,
    initial_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_initial = max(1, int(round(len(shuffled) * initial_fraction)))
    if initial_mode == "head":
        initial_mask = np.zeros(len(shuffled), dtype=bool)
        initial_mask[:n_initial] = True
    elif initial_mode == "rng":
        initial_idx = rng.choice(len(shuffled), size=n_initial, replace=False)
        initial_mask = np.zeros(len(shuffled), dtype=bool)
        initial_mask[initial_idx] = True
    else:
        raise ValueError(f"Unknown initial_mode: {initial_mode}")
    initial = shuffled.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True)
    candidate = shuffled.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True)
    return initial, candidate


def make_official_u1_split(seed: int, initial_fraction: float, initial_mode: str) -> RatingSplit:
    train = read_fold(RAW_DIR / "u1.base")
    test = read_fold(RAW_DIR / "u1.test")
    initial, candidate = make_initial_candidate(train, seed, initial_fraction, initial_mode)
    return RatingSplit(initial=initial, candidate=candidate, test=test)


def make_random_split(seed: int, initial_fraction: float, initial_mode: str) -> RatingSplit:
    dataset = load_movielens_100k(RAW_DIR)
    if initial_mode == "rng":
        return split_ratings(dataset.ratings, seed=seed, test_ratio=0.2, initial_fraction_of_train=initial_fraction)
    base = split_ratings(dataset.ratings, seed=seed, test_ratio=0.2, initial_fraction_of_train=initial_fraction)
    train = pd.concat([base.initial, base.candidate], ignore_index=True)
    initial, candidate = make_initial_candidate(train, seed, initial_fraction, initial_mode)
    return RatingSplit(initial=initial, candidate=candidate, test=base.test.reset_index(drop=True))


def protocol_catalog() -> list[dict[str, object]]:
    table1_fraction = 1364 / 80000
    return [
        {
            "protocol": "official_u1_rng_init010",
            "split_mode": "official_u1",
            "initial_fraction": 0.010,
            "initial_mode": "rng",
        },
        {
            "protocol": "official_u1_head_init010",
            "split_mode": "official_u1",
            "initial_fraction": 0.010,
            "initial_mode": "head",
        },
        {
            "protocol": "official_u1_rng_table1_init017",
            "split_mode": "official_u1",
            "initial_fraction": table1_fraction,
            "initial_mode": "rng",
        },
        {
            "protocol": "official_u1_rng_init025",
            "split_mode": "official_u1",
            "initial_fraction": 0.025,
            "initial_mode": "rng",
        },
        {
            "protocol": "random80_rng_init010",
            "split_mode": "random80",
            "initial_fraction": 0.010,
            "initial_mode": "rng",
        },
        {
            "protocol": "random80_rng_table1_init017",
            "split_mode": "random80",
            "initial_fraction": table1_fraction,
            "initial_mode": "rng",
        },
    ]


def make_split_for_protocol(protocol: dict[str, object], seed: int) -> RatingSplit:
    initial_fraction = float(protocol["initial_fraction"])
    initial_mode = str(protocol["initial_mode"])
    split_mode = str(protocol["split_mode"])
    if split_mode == "official_u1":
        return make_official_u1_split(seed, initial_fraction, initial_mode)
    if split_mode == "random80":
        return make_random_split(seed, initial_fraction, initial_mode)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def summarize_final(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    group_cols = ["protocol", "variant", "policy"]
    for keys, group in final.groupby(group_cols, sort=False):
        protocol, variant, policy = keys
        row: dict[str, float | str] = {
            "protocol": str(protocol),
            "variant": str(variant),
            "policy": str(policy),
            "seed_count": float(group["seed"].nunique()),
        }
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
            "candidate_size",
        ]:
            values = group[col].dropna().to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    columns = [
        "protocol",
        "variant",
        "comparator",
        "metric",
        "du_value",
        "comparator_value",
        "gap_favoring_du",
    ]
    for protocol, group in summary.groupby("protocol", sort=False):
        comparator = group[group["variant"] == "mf_linear_ucb_a002"]
        du_rows = group[group["policy"] == "du_net"]
        if comparator.empty or du_rows.empty:
            continue
        comparator_row = comparator.iloc[0]
        for _, du in du_rows.iterrows():
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
                gap = du_value - comparator_value if metric == "cumulative_reward" else comparator_value - du_value
                rows.append(
                    {
                        "protocol": str(protocol),
                        "variant": str(du["variant"]),
                        "comparator": "mf_linear_ucb_a002",
                        "metric": metric,
                        "du_value": du_value,
                        "comparator_value": comparator_value,
                        "gap_favoring_du": float(gap),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 3, 5])
    parser.add_argument("--protocols", nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(TABLE_DIR)
    dataset = load_movielens_100k(RAW_DIR)
    mf_config = MFConfig(rank=12, epochs=6, learning_rate=0.025, regularization=0.04, algorithm="svdpp")
    acquisition = AcquisitionConfig(
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
            "variant": "graph_late_lambda0_start15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 15, "late_cycle_lambda_multiplier": 0.0},
        },
        {
            "variant": "graph_late_lambda0_start16",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 16, "late_cycle_lambda_multiplier": 0.0},
        },
        {
            "variant": "graph_late_lambda_half_start15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 15, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "graph_late_lambda_half_start16",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 16, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "graph_late_lambda_half_start16_p000",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "diffusion_prob": 0.0,
            },
        },
        {
            "variant": "graph_late_lambda_half_start16_p001",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "diffusion_prob": 0.001,
            },
        },
        {
            "variant": "backend_rank20_ep12",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 16, "late_cycle_lambda_multiplier": 0.5},
            "mf_overrides": {"rank": 20, "epochs": 12, "learning_rate": 0.015, "regularization": 0.04},
        },
        {
            "variant": "backend_rank30_ep15",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 16, "late_cycle_lambda_multiplier": 0.5},
            "mf_overrides": {"rank": 30, "epochs": 15, "learning_rate": 0.010, "regularization": 0.06},
        },
        {
            "variant": "dynamic_tune_lam_sig",
            "policy": "du_net_adaptive_grid",
            "overrides": {
                "diffusion_prob": 0.001,
                "adaptive_diffusion_prob_grid": (0.001,),
                "adaptive_lambda_grid": (0.05, 0.10, 0.20, 0.30),
                "sigma_grid": (0.2, 0.4, 0.8),
                "adaptive_dls_shrinkage_grid": (0.25, 0.50, 0.75),
                "dls_shrinkage_mode": "fixed",
                "adaptive_validation_metric": "linear_reward_norm_plus_rmse",
                "adaptive_prefilter_metric": "acquisition_score",
                "adaptive_max_refits": 8,
            },
        },
        {
            "variant": "residual_dls_alpha_schedule",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "dls_shrinkage": 0.75,
                "dls_shrinkage_mode": "fixed",
                "dls_shrinkage_schedule": (
                    {"start_cycle": 1, "end_cycle": 5, "alpha": 0.75, "mode": "fixed"},
                    {"start_cycle": 6, "end_cycle": 15, "alpha": 0.50, "mode": "fixed"},
                    {"start_cycle": 16, "end_cycle": 20, "alpha": 0.25, "mode": "fixed"},
                ),
            },
        },
        {
            "variant": "entropy_rho010",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "entropy_coherence_blend": 0.10,
            },
        },
        {
            "variant": "divcap_user2_item5",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "selection": "diversity_cap",
                "selection_max_per_user": 2,
                "selection_max_per_item": 5,
            },
        },
        {
            "variant": "coherence_lite_p000",
            "policy": "du_net",
            "overrides": {
                "entropy_mode": "coherence_lite",
                "diffusion_prob": 0.0,
            },
        },
        {
            "variant": "dynamic_lambda_sigma_p000",
            "policy": "du_net_adaptive_grid",
            "overrides": {
                "diffusion_prob": 0.0,
                "adaptive_diffusion_prob_grid": (0.0,),
                "adaptive_validation_metric": "linear_reward_norm_plus_rmse",
                "adaptive_prefilter_metric": "acquisition_score",
                "adaptive_max_refits": 6,
                "adaptive_stage_grids": (
                    {
                        "start_cycle": 1,
                        "end_cycle": 5,
                        "lambda_grid": (0.25, 0.5, 0.75),
                        "sigma_grid": (0.2, 0.4, 0.8),
                    },
                    {
                        "start_cycle": 6,
                        "end_cycle": 15,
                        "lambda_grid": (0.125, 0.25, 0.5),
                        "sigma_grid": (0.2, 0.4, 0.8),
                    },
                    {
                        "start_cycle": 16,
                        "end_cycle": 20,
                        "lambda_grid": (0.05, 0.125, 0.25),
                        "sigma_grid": (0.1, 0.2, 0.4),
                    },
                ),
            },
        },
        {
            "variant": "coherence_lite_dynamic_p000",
            "policy": "du_net_adaptive_grid",
            "overrides": {
                "entropy_mode": "coherence_lite",
                "diffusion_prob": 0.0,
                "adaptive_diffusion_prob_grid": (0.0,),
                "adaptive_validation_metric": "linear_reward_norm_plus_rmse",
                "adaptive_prefilter_metric": "acquisition_score",
                "adaptive_max_refits": 6,
                "adaptive_stage_grids": (
                    {
                        "start_cycle": 1,
                        "end_cycle": 5,
                        "lambda_grid": (0.25, 0.5, 0.75),
                        "sigma_grid": (0.2, 0.4, 0.8),
                    },
                    {
                        "start_cycle": 6,
                        "end_cycle": 15,
                        "lambda_grid": (0.125, 0.25, 0.5),
                        "sigma_grid": (0.2, 0.4, 0.8),
                    },
                    {
                        "start_cycle": 16,
                        "end_cycle": 20,
                        "lambda_grid": (0.05, 0.125, 0.25),
                        "sigma_grid": (0.1, 0.2, 0.4),
                    },
                ),
            },
        },
        {
            "variant": "graph_late_lambda_half_start16_topm_exp_tau010",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "selection": "top_m_exp",
                "selection_top_m": 5000,
                "selection_temperature": 0.1,
            },
        },
        {
            "variant": "graph_late_lambda_half_start16_topm_exp_tau020",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "selection": "top_m_exp",
                "selection_top_m": 5000,
                "selection_temperature": 0.2,
            },
        },
        {
            "variant": "graph_late_lambda_half_start16_no_network",
            "policy": "du_net_no_network",
            "overrides": {"late_cycle_guard_start": 16, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "graph_late_lambda_half_start16_exp_tau1",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "selection": "exp",
                "selection_temperature": 1.0,
            },
        },
        {
            "variant": "graph_late_half_start16_frontier90",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "late_cycle_frontier_quantile": 0.90,
            },
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
        {
            "variant": "graph_late_half_start16_frontier95_floor25",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "late_cycle_frontier_quantile": 0.95,
                "late_cycle_frontier_floor": 0.25,
            },
        },
        {
            "variant": "graph_late_half_start16_frontier99",
            "policy": "du_net",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "late_cycle_frontier_quantile": 0.99,
            },
        },
        {
            "variant": "graph_late_lambda_half_start17",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 17, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "graph_late_lambda_half_start18",
            "policy": "du_net",
            "overrides": {"late_cycle_guard_start": 18, "late_cycle_lambda_multiplier": 0.5},
        },
        {
            "variant": "graph_adaptive_late_half_start16",
            "policy": "du_net_adaptive_late_guard",
            "overrides": {
                "late_cycle_guard_start": 16,
                "late_cycle_lambda_multiplier": 0.5,
                "adaptive_validation_metric": "linear_reward_norm",
                "adaptive_min_inner_reward_delta": 0.0,
            },
        },
        {
            "variant": "graph_adaptive_grid_phase4_narrow",
            "policy": "du_net_adaptive_grid",
            "overrides": {
                "adaptive_validation_metric": "linear_reward_norm",
                "adaptive_min_inner_reward_delta": 0.0,
                "adaptive_lambda_grid": (0.10, 0.25, 0.50),
                "sigma_grid": (0.2, 0.4, 0.8),
                "adaptive_diffusion_prob_grid": (0.0, 0.005, 0.010),
            },
        },
        {"variant": "myopic", "policy": "myopic", "overrides": {}},
        {"variant": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "overrides": {"ucb_alpha": 0.02}},
    ]
    if args.variants:
        requested_variants = set(args.variants)
        variants = [variant for variant in variants if str(variant["variant"]) in requested_variants]
        missing_variants = requested_variants - {str(variant["variant"]) for variant in variants}
        if missing_variants:
            raise ValueError(f"Unknown variants requested: {sorted(missing_variants)}")
    protocols = protocol_catalog()
    if args.protocols:
        requested = set(args.protocols)
        protocols = [protocol for protocol in protocols if str(protocol["protocol"]) in requested]
        missing = requested - {str(protocol["protocol"]) for protocol in protocols}
        if missing:
            raise ValueError(f"Unknown protocols requested: {sorted(missing)}")

    metrics_path = TABLE_DIR / f"{args.prefix}_metrics.csv"
    summary_path = TABLE_DIR / f"{args.prefix}_summary.csv"
    gaps_path = TABLE_DIR / f"{args.prefix}_gaps.csv"
    parts: list[pd.DataFrame] = []
    completed: set[tuple[str, str, int]] = set()
    wrote_header = False
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        parts.append(existing)
        completed = {
            (str(row.protocol), str(row.variant), int(row.seed))
            for row in existing[["protocol", "variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True

    for protocol in protocols:
        for seed in args.seeds:
            split = make_split_for_protocol(protocol, seed)
            for variant in variants:
                key = (str(protocol["protocol"]), str(variant["variant"]), seed)
                if key in completed:
                    continue
                metrics = run_policy(
                    policy=str(variant["policy"]),
                    split=split,
                    adjacency=dataset.adjacency,
                    num_users=dataset.num_users,
                    num_items=dataset.num_items,
                    mf_config=replace(mf_config, **variant.get("mf_overrides", {})),
                    acquisition_config=replace(acquisition, **variant["overrides"]),
                    seed=seed,
                )
                metrics.insert(0, "protocol", str(protocol["protocol"]))
                metrics.insert(1, "variant", str(variant["variant"]))
                metrics.insert(2, "seed", seed)
                metrics.insert(3, "split_mode", str(protocol["split_mode"]))
                metrics.insert(4, "initial_fraction", float(protocol["initial_fraction"]))
                metrics.insert(5, "initial_mode", str(protocol["initial_mode"]))
                parts.append(metrics)
                metrics.to_csv(metrics_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True

    metrics_df = pd.concat(parts, ignore_index=True)
    final = metrics_df.sort_values("cycle").groupby(["protocol", "variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    gaps = build_gaps(summary)
    summary.to_csv(summary_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    print(f"Wrote {metrics_path}")
    print(summary.sort_values(["protocol", "variant"]).to_string(index=False))
    printable_gaps = gaps[gaps["metric"].isin(["linear_reward_norm_w_rmse", "cumulative_reward"])]
    if printable_gaps.empty:
        print("No DU-vs-Linear gaps to print; include mf_linear_ucb_a002 to populate comparator gaps.")
    else:
        print(
            printable_gaps.sort_values(["metric", "gap_favoring_du"], ascending=[True, False]).to_string(index=False)
        )


if __name__ == "__main__":
    main()
