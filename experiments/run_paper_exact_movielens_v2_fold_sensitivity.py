from __future__ import annotations

import argparse
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
from src.utils import ensure_dir, load_json


DEFAULT_PREFIX = "paper_exact_attempt_movielens_v2_fold_sensitivity_5seeds"
DUNET_VARIANT = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
DEFAULT_VARIANTS = [DUNET_VARIANT, "linucb_concat", "linucb_mf_interaction"]


def safe_fold(value: str | int) -> int:
    text = str(value).strip().lower()
    return int(text[1:] if text.startswith("u") else text)


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


def summarize_final(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = [
        "standard_mask_rmse",
        "pdf_literal_norm_rmse",
        "cumulative_reward",
        "centered_cumulative_reward",
        "w_rmse",
        "reward_tail_proxy_rmse",
        "observed_size",
        "candidate_size",
    ]
    for (fold, variant, policy), group in final.groupby(["fold", "variant", "policy"], sort=False):
        row: dict[str, object] = {
            "fold": fold,
            "variant": variant,
            "policy": policy,
            "seed_count": int(group["seed"].nunique()),
        }
        for col in metric_cols:
            values = group[col].dropna().to_numpy(dtype=float)
            row[col] = float(np.mean(values)) if len(values) else np.nan
            row[f"{col}_ci95"] = (
                float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_gaps(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, comparator), _ in final[final["variant"] != DUNET_VARIANT].groupby(["fold", "variant"]):
        for seed in sorted(final["seed"].dropna().unique()):
            cur = final[(final["fold"] == fold) & (final["seed"] == seed)]
            du = cur[cur["variant"] == DUNET_VARIANT]
            comp = cur[cur["variant"] == comparator]
            if du.empty or comp.empty:
                continue
            du_row = du.iloc[0]
            comp_row = comp.iloc[0]
            rows.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "comparator": comparator,
                    "dunet_standard_mask_rmse": float(du_row["standard_mask_rmse"]),
                    "comparator_standard_mask_rmse": float(comp_row["standard_mask_rmse"]),
                    "standard_gap_comparator_minus_dunet": float(comp_row["standard_mask_rmse"] - du_row["standard_mask_rmse"]),
                    "dunet_cumulative_reward": float(du_row["cumulative_reward"]),
                    "comparator_cumulative_reward": float(comp_row["cumulative_reward"]),
                    "reward_gap_dunet_minus_comparator": float(du_row["cumulative_reward"] - comp_row["cumulative_reward"]),
                    "standard_win": int(float(du_row["standard_mask_rmse"]) < float(comp_row["standard_mask_rmse"])),
                    "reward_win": int(float(du_row["cumulative_reward"]) > float(comp_row["cumulative_reward"])),
                }
            )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["fold", "comparator"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_standard_gap=("standard_gap_comparator_minus_dunet", "mean"),
            mean_reward_gap=("reward_gap_dunet_minus_comparator", "mean"),
            standard_win_rate=("standard_win", "mean"),
            reward_win_rate=("reward_win", "mean"),
        )
        .sort_values(["fold", "comparator"])
    )
    return summary


def build_cycle_positivity(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fold, comparator), _ in metrics[metrics["variant"] != DUNET_VARIANT].groupby(["fold", "variant"]):
        per_cycle: list[dict[str, object]] = []
        for cycle in sorted(metrics["cycle"].dropna().unique()):
            cur = metrics[(metrics["fold"] == fold) & (metrics["cycle"] == cycle)]
            gaps = []
            reward_gaps = []
            for seed in sorted(cur["seed"].dropna().unique()):
                pair = cur[cur["seed"] == seed]
                du = pair[pair["variant"] == DUNET_VARIANT]
                comp = pair[pair["variant"] == comparator]
                if du.empty or comp.empty:
                    continue
                gaps.append(float(comp.iloc[0]["standard_mask_rmse"] - du.iloc[0]["standard_mask_rmse"]))
                reward_gaps.append(float(du.iloc[0]["cumulative_reward"] - comp.iloc[0]["cumulative_reward"]))
            if gaps:
                per_cycle.append(
                    {
                        "fold": fold,
                        "comparator": comparator,
                        "cycle": int(cycle),
                        "mean_standard_gap_comparator_minus_dunet": float(np.mean(gaps)),
                        "mean_reward_gap_dunet_minus_comparator": float(np.mean(reward_gaps)),
                        "seed_count": len(gaps),
                    }
                )
        cycle_frame = pd.DataFrame(per_cycle)
        if cycle_frame.empty:
            continue
        rows.append(
            {
                "fold": fold,
                "comparator": comparator,
                "cycles_observed": int(cycle_frame["cycle"].nunique()),
                "positive_standard_gap_cycles": int((cycle_frame["mean_standard_gap_comparator_minus_dunet"] > 0).sum()),
                "positive_standard_gap_share": float((cycle_frame["mean_standard_gap_comparator_minus_dunet"] > 0).mean()),
                "positive_reward_gap_cycles": int((cycle_frame["mean_reward_gap_dunet_minus_comparator"] > 0).sum()),
                "positive_reward_gap_share": float((cycle_frame["mean_reward_gap_dunet_minus_comparator"] > 0).mean()),
                "mean_standard_gap_all_cycles": float(cycle_frame["mean_standard_gap_comparator_minus_dunet"].mean()),
                "mean_reward_gap_all_cycles": float(cycle_frame["mean_reward_gap_dunet_minus_comparator"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "comparator"]) if rows else pd.DataFrame()


def plot_metric(metrics: pd.DataFrame, prefix: str, metric: str, figure_dir: Path) -> None:
    if metrics.empty:
        return
    plt.figure(figsize=(9, 5))
    for (fold, variant), group in metrics.groupby(["fold", "variant"], sort=False):
        curve = group.groupby("cycle")[metric].mean()
        plt.plot(curve.index, curve.values, marker="o", linewidth=1.5, label=f"{fold}:{variant}")
    plt.xlabel("Acquisition cycle")
    plt.ylabel(metric)
    plt.title(f"{prefix}: {metric}")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{prefix}_{metric}.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_exact_attempt_movielens_v2.json"))
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--folds", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    metrics_path = table_dir / f"{args.output_prefix}_metrics.csv"
    summary_path = table_dir / f"{args.output_prefix}_final_summary.csv"
    gaps_path = table_dir / f"{args.output_prefix}_fold_comparator_gaps.csv"
    positivity_path = table_dir / f"{args.output_prefix}_cycle_positivity.csv"

    dataset_cfg = config["dataset"]
    raw_dir = ROOT / dataset_cfg["raw_dir"]
    dataset = load_movielens_100k(raw_dir, network_mode=dataset_cfg.get("network_mode", "gcmc_features"))
    network_embedding = dataset.user_features
    split_cfg = config["split"]
    num_items = int(split_cfg.get("num_items", 1681))
    num_users = int(split_cfg.get("num_users", dataset.num_users))
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    variants = [variant for variant in config["variants"] if variant["label"] in set(args.variants)]
    missing = sorted(set(args.variants) - {variant["label"] for variant in variants})
    if missing:
        raise ValueError(f"Requested variants not found in config: {missing}")

    all_metrics: list[pd.DataFrame] = []
    completed: set[tuple[str, str, int]] = set()
    wrote_header = False
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        all_metrics.append(existing)
        completed = {
            (str(row.fold), str(row.variant), int(row.seed))
            for row in existing[["fold", "variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True

    for fold_num in [safe_fold(fold) for fold in args.folds]:
        fold_label = f"u{fold_num}"
        for seed in args.seeds:
            split = make_split(raw_dir, fold=fold_num, seed=int(seed), initial_size=int(split_cfg["initial_size"]), num_items=num_items)
            for variant in variants:
                label = str(variant["label"])
                if (fold_label, label, int(seed)) in completed:
                    continue
                acquisition = base_acquisition
                if variant.get("acquisition_overrides"):
                    acquisition = replace(base_acquisition, **variant["acquisition_overrides"])
                metrics = run_policy(
                    policy=str(variant["policy"]),
                    split=split,
                    adjacency=dataset.adjacency,
                    num_users=num_users,
                    num_items=num_items,
                    mf_config=mf_config,
                    acquisition_config=acquisition,
                    seed=int(seed) + 100 * fold_num,
                    network_embedding=network_embedding,
                )
                metrics.insert(0, "protocol", "official_movielens_u1_u5_table1_initial")
                metrics.insert(1, "fold", fold_label)
                metrics.insert(2, "variant", label)
                metrics.insert(3, "seed", int(seed))
                metrics.insert(4, "initial_size", len(split.initial))
                metrics.insert(5, "test_size", len(split.test))
                all_metrics.append(metrics)
                metrics.to_csv(metrics_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    final = metrics_df.sort_values("cycle").groupby(["fold", "variant", "seed"], as_index=False).tail(1)
    summarize_final(final).to_csv(summary_path, index=False)
    build_final_gaps(final).to_csv(gaps_path, index=False)
    build_cycle_positivity(metrics_df).to_csv(positivity_path, index=False)
    plot_metric(metrics_df, str(args.output_prefix), "standard_mask_rmse", figure_dir)
    plot_metric(metrics_df, str(args.output_prefix), "cumulative_reward", figure_dir)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote final summary: {summary_path}")
    print(f"Wrote final gaps: {gaps_path}")
    print(f"Wrote cycle positivity: {positivity_path}")


if __name__ == "__main__":
    main()
