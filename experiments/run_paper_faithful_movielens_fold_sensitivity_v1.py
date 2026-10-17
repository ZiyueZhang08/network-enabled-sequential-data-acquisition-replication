from __future__ import annotations

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

PREFIX = "paper_faithful_movielens_softmax_tau1_fold_sensitivity_v1_3seeds"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
METRICS_PATH = TABLE_DIR / f"{PREFIX}_metrics.csv"
SUMMARY_PATH = TABLE_DIR / f"{PREFIX}_final_summary.csv"
GAPS_PATH = TABLE_DIR / f"{PREFIX}_fold_comparator_gaps.csv"

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
VARIANTS = [
    {
        "label": "schur_loglambda_tuned_full_pool_softmax_tau1_literal",
        "policy": "paper_faithful_dunet",
        "overrides": {"selection": "exp", "selection_temperature": 1.0},
    },
    {"label": "linucb_concat", "policy": "linucb_concat", "overrides": {"selection": "topk"}},
    {"label": "linucb_mf_interaction", "policy": "linucb_mf_interaction", "overrides": {"selection": "topk"}},
]
DUNET_VARIANT = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"


def read_fold(path: Path, num_items: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    frame = frame[frame["item_id"] < int(num_items)].copy()
    return frame[RATING_COLUMNS].reset_index(drop=True)


def make_split(raw_dir: Path, fold: int, seed: int, initial_size: int, num_items: int) -> RatingSplit:
    train = read_fold(raw_dir / f"u{fold}.base", num_items=num_items)
    test = read_fold(raw_dir / f"u{fold}.test", num_items=num_items)
    rng = np.random.default_rng(seed)
    size = min(int(initial_size), len(train))
    initial_idx = rng.choice(len(train), size=size, replace=False)
    initial_mask = np.zeros(len(train), dtype=bool)
    initial_mask[initial_idx] = True
    return RatingSplit(
        initial=train.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True),
        candidate=train.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True),
        test=test[RATING_COLUMNS].reset_index(drop=True),
    )


def summarize_final(final: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for key, group in final.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, float | str] = dict(zip(group_cols, key))
        row["policy"] = str(group["policy"].iloc[0])
        row["seed_count"] = float(group["seed"].nunique())
        for metric in [
            "standard_mask_rmse",
            "pdf_literal_norm_rmse",
            "cumulative_reward",
            "w_rmse",
            "reward_tail_proxy_rmse",
            "selected_avg_rating",
            "selected_unique_users",
            "selected_unique_items",
            "selected_avg_item_popularity",
        ]:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[metric] = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_gaps(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (fold, comp), _ in final[final["variant"] != DUNET_VARIANT].groupby(["fold", "variant"]):
        for seed in sorted(final["seed"].unique()):
            cur = final[(final["fold"] == fold) & (final["seed"] == seed)]
            du = cur[cur["variant"] == DUNET_VARIANT]
            baseline = cur[cur["variant"] == comp]
            if du.empty or baseline.empty:
                continue
            du_row = du.iloc[0]
            base_row = baseline.iloc[0]
            rows.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "comparator": comp,
                    "dunet_standard_mask_rmse": float(du_row["standard_mask_rmse"]),
                    "comparator_standard_mask_rmse": float(base_row["standard_mask_rmse"]),
                    "standard_gap_comparator_minus_dunet": float(
                        base_row["standard_mask_rmse"] - du_row["standard_mask_rmse"]
                    ),
                    "dunet_pdf_literal_norm_rmse": float(du_row["pdf_literal_norm_rmse"]),
                    "comparator_pdf_literal_norm_rmse": float(base_row["pdf_literal_norm_rmse"]),
                    "pdf_gap_comparator_minus_dunet": float(
                        base_row["pdf_literal_norm_rmse"] - du_row["pdf_literal_norm_rmse"]
                    ),
                    "dunet_cumulative_reward": float(du_row["cumulative_reward"]),
                    "comparator_cumulative_reward": float(base_row["cumulative_reward"]),
                    "reward_gap_dunet_minus_comparator": float(
                        du_row["cumulative_reward"] - base_row["cumulative_reward"]
                    ),
                    "standard_win": int(du_row["standard_mask_rmse"] < base_row["standard_mask_rmse"]),
                }
            )
    gaps = pd.DataFrame(rows)
    summary = (
        gaps.groupby(["fold", "comparator"], as_index=False)
        .agg(
            dunet_rmse=("dunet_standard_mask_rmse", "mean"),
            comparator_rmse=("comparator_standard_mask_rmse", "mean"),
            gap=("standard_gap_comparator_minus_dunet", "mean"),
            reward_gap=("reward_gap_dunet_minus_comparator", "mean"),
            win_rate=("standard_win", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["fold", "comparator"])
    )
    return summary


def plot_metric(metrics: pd.DataFrame, metric: str) -> None:
    plt.figure(figsize=(8.5, 5))
    for variant, group in metrics.groupby("variant", sort=False):
        cycle = group.groupby("cycle")[metric].mean()
        plt.plot(cycle.index, cycle.values, marker="o", label=variant)
    plt.xlabel("Acquisition cycle")
    plt.ylabel(metric)
    plt.title(f"{PREFIX}: {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{PREFIX}_{metric}.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)
    dataset = load_movielens_100k(RAW_DIR, network_mode="gcmc_features")
    network_embedding = dataset.user_features
    mf_config = MFConfig(rank=20, epochs=20, learning_rate=0.015, regularization=0.06, algorithm="svdpp")
    base_acquisition = AcquisitionConfig(
        cycles=20,
        batch_size=100,
        lambda_entropy=1.0,
        diffusion_prob=0.0,
        sigma=0.4,
        dls_top_k=20,
        dls_neighbor_mode="graph_edges",
        paper_dls_scope="full",
        selection="topk",
        selection_temperature=1.0,
        rating_midpoint=3.0,
        entropy_mode="schur_conditional_variance",
        entropy_score_transform="none",
        entropy_logdet_regularization=1.0,
        paper_entropy_correction_weight=1.0,
        paper_entropy_shortlist_size=20_000,
        paper_entropy_batch_size=4096,
        sigma_grid=(0.1, 0.2, 0.4, 0.8, 1.6),
        adaptive_lambda_grid=(0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
        adaptive_diffusion_prob_grid=(0.0,),
        paper_tuning_frequency=5,
        paper_tuning_metric="standard_mask_rmse",
        ucb_alpha=0.8,
        ucb_regularization=1.0,
    )

    completed: set[tuple[str, int, int]] = set()
    wrote_header = False
    parts: list[pd.DataFrame] = []
    if METRICS_PATH.exists():
        existing = pd.read_csv(METRICS_PATH)
        parts.append(existing)
        completed = {
            (str(row.variant), int(str(row.fold).lstrip("u")), int(row.seed))
            for row in existing[["variant", "fold", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True

    for fold in FOLDS:
        for seed in SEEDS:
            split = make_split(RAW_DIR, fold=fold, seed=seed, initial_size=1364, num_items=1681)
            for variant in VARIANTS:
                label = str(variant["label"])
                if (label, fold, seed) in completed:
                    continue
                acquisition = replace(base_acquisition, **variant["overrides"])
                metrics = run_policy(
                    policy=str(variant["policy"]),
                    split=split,
                    adjacency=dataset.adjacency,
                    num_users=943,
                    num_items=1681,
                    mf_config=mf_config,
                    acquisition_config=acquisition,
                    seed=seed + 100 * fold,
                    network_embedding=network_embedding,
                )
                metrics.insert(0, "protocol", "official_movielens_u1_u5_table1_initial")
                metrics.insert(1, "fold", f"u{fold}")
                metrics.insert(2, "variant", label)
                metrics.insert(3, "seed", int(seed))
                metrics.insert(4, "initial_size", len(split.initial))
                metrics.insert(5, "test_size", len(split.test))
                parts.append(metrics)
                metrics.to_csv(METRICS_PATH, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True

    metrics_df = pd.concat(parts, ignore_index=True)
    final = metrics_df.sort_values("cycle").groupby(["fold", "variant", "seed"], as_index=False).tail(1)
    summarize_final(final, ["fold", "variant"]).to_csv(SUMMARY_PATH, index=False)
    gaps = build_gaps(final)
    gaps.to_csv(GAPS_PATH, index=False)
    plot_metric(metrics_df, "standard_mask_rmse")
    plot_metric(metrics_df, "cumulative_reward")
    print(f"Wrote metrics: {METRICS_PATH}")
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
