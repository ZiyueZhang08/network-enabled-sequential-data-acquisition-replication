"""Confirm the Figure 4a rating-tail diagnosis over 20 MovieLens seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, _mf_linear_ucb_scores, _policy_scores, _select_rows  # noqa: E402
from src.data_loader import load_dataset_from_config, split_dataset  # noqa: E402
from src.dls import compute_dls_neighbors, smooth_dls_user_factors  # noqa: E402
from src.graph_embedding import user_embedding  # noqa: E402
from src.matrix_factorization import MFConfig, MatrixFactorization  # noqa: E402
from src.reward import true_centered_reward  # noqa: E402
from src.utils import load_json  # noqa: E402


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"
OUT_SEED_METRICS = TABLE_DIR / "figure4a_rating_tail_20seed_confirmation_seed_metrics.csv"
OUT_FINAL = TABLE_DIR / "figure4a_rating_tail_20seed_confirmation.csv"
OUT_GAPS = TABLE_DIR / "figure4a_rating_tail_20seed_confirmation_gaps.csv"
OUT_REPORT = REPORT_DIR / "figure4a_rating_tail_20seed_confirmation.md"

VARIANTS = [
    {"label": "du_net", "policy": "du_net", "acquisition_overrides": {}},
    {"label": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "acquisition_overrides": {"ucb_alpha": 0.02}},
]


def ci95(values: list[float]) -> float:
    clean = [float(value) for value in values if not np.isnan(float(value))]
    if len(clean) < 2:
        return 0.0
    return 1.96 * statistics.stdev(clean) / math.sqrt(len(clean))


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def append_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def raw_predict(model: MatrixFactorization, ratings: pd.DataFrame, user_factors: np.ndarray) -> np.ndarray:
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    preds = (
        model.global_mean
        + model.user_bias[users]
        + model.item_bias[items]
        + np.sum(user_factors[users] * model.item_factors[items], axis=1)
    )
    return np.clip(preds, 1.0, 5.0)


def weighted_error(truth: np.ndarray, preds: np.ndarray, weights: np.ndarray) -> float:
    weights = weights.astype(float)
    if float(np.sum(weights)) <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(weights * (truth - preds) ** 2) / np.sum(weights)))


def evaluate_final(
    *,
    variant: str,
    policy: str,
    seed: int,
    observed_size: int,
    candidate_size: int,
    cumulative_reward: float,
    initial_size: int,
    model: MatrixFactorization,
    test: pd.DataFrame,
    user_factors: np.ndarray,
    midpoint: float,
) -> list[dict[str, object]]:
    truth = test["rating"].to_numpy(dtype=float)
    preds = raw_predict(model, test, user_factors)
    weights = {
        "rmse_all": np.ones(len(truth)),
        "rmse_high_rating_ge4": (truth >= 4.0).astype(float),
        "rmse_low_rating_le3": (truth <= 3.0).astype(float),
        "rmse_rating_eq4": (truth == 4.0).astype(float),
        "rmse_rating_eq5": (truth == 5.0).astype(float),
        "local_reward_w_rmse": np.maximum(truth - midpoint, 0.0) + 0.1,
        "sigmoid_reward_w_rmse": 1.0 / (1.0 + np.exp(-(truth - midpoint))),
    }
    acquired_count = observed_size - initial_size
    avg_acquired_rating = midpoint + cumulative_reward / acquired_count if acquired_count else float("nan")
    rows: list[dict[str, object]] = []
    for metric, metric_weights in weights.items():
        rows.append(
            {
                "variant": variant,
                "policy": policy,
                "seed": seed,
                "metric_name": metric,
                "metric_value": weighted_error(truth, preds, metric_weights),
                "observed_size": observed_size,
                "candidate_size": candidate_size,
                "cumulative_reward": cumulative_reward,
                "avg_acquired_rating": avg_acquired_rating,
                "test_count": len(test),
                "high_rating_test_count": int(np.sum(truth >= 4.0)),
                "low_rating_test_count": int(np.sum(truth <= 3.0)),
                "clip_low_share": float(np.mean(preds <= 1.0)),
                "clip_high_share": float(np.mean(preds >= 5.0)),
            }
        )
    return rows


def replay_final_rows(
    *,
    variant: dict[str, object],
    seed: int,
    split,
    adjacency: np.ndarray,
    num_users: int,
    num_items: int,
    mf_config: MFConfig,
    base_acquisition: AcquisitionConfig,
    network_embedding: np.ndarray,
) -> list[dict[str, object]]:
    label = str(variant["label"])
    policy = str(variant["policy"])
    acquisition = replace(base_acquisition, **variant.get("acquisition_overrides", {}))
    rng = np.random.default_rng(seed)
    observed = split.initial.copy()
    candidates = split.candidate.copy()
    initial_size = len(observed)
    cumulative_reward = 0.0
    dls_neighbor_idx, dls_neighbor_dist_sq = compute_dls_neighbors(
        network_embedding,
        top_k=acquisition.dls_top_k,
    )
    for cycle in range(acquisition.cycles + 1):
        model = MatrixFactorization(num_users, num_items, mf_config, seed=seed * 1009 + cycle).fit(observed)
        base_user_factors = model.user_representations()
        dls_user_factors, _ = smooth_dls_user_factors(
            base_user_factors,
            dls_neighbor_idx,
            dls_neighbor_dist_sq,
            sigma=acquisition.sigma,
        )
        if cycle == acquisition.cycles or candidates.empty:
            eval_factors = dls_user_factors if policy == "du_net" else base_user_factors
            return evaluate_final(
                variant=label,
                policy=policy,
                seed=seed,
                observed_size=len(observed),
                candidate_size=len(candidates),
                cumulative_reward=cumulative_reward,
                initial_size=initial_size,
                model=model,
                test=split.test,
                user_factors=eval_factors,
                midpoint=acquisition.rating_midpoint,
            )
        if policy == "mf_linear_ucb":
            score = _mf_linear_ucb_scores(model, candidates, observed, base_user_factors, acquisition)
            scores = {"score": score}
        else:
            scores = _policy_scores(
                policy,
                model,
                candidates,
                observed,
                adjacency,
                num_users,
                num_items,
                dls_user_factors,
                acquisition,
            )
        selected_pos = _select_rows(
            scores["score"],
            acquisition.batch_size,
            rng,
            acquisition.selection,
            acquisition.selection_temperature,
        )
        selected = candidates.iloc[selected_pos].copy()
        cumulative_reward += true_centered_reward(selected, midpoint=acquisition.rating_midpoint)
        observed = pd.concat([observed, selected], ignore_index=True)
        candidates = candidates.drop(candidates.index[selected_pos]).reset_index(drop=True)
    raise RuntimeError("Unreachable trajectory state")


def run_confirmation(seeds: list[int], resume: bool) -> pd.DataFrame:
    config = load_json(CONFIG_PATH)
    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    completed: set[tuple[str, int]] = set()
    if resume and OUT_SEED_METRICS.exists():
        existing = pd.read_csv(OUT_SEED_METRICS)
        completed = {
            (str(row.variant), int(row.seed))
            for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
    for seed in seeds:
        split = split_dataset(dataset, seed=seed, **config["split"])
        embedding = user_embedding(
            dataset.adjacency,
            dim=mf_config.rank,
            method=base_acquisition.graph_embedding_method,
            seed=seed,
        )
        for variant in VARIANTS:
            label = str(variant["label"])
            if (label, seed) in completed:
                continue
            rows = replay_final_rows(
                variant=variant,
                seed=seed,
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf_config,
                base_acquisition=base_acquisition,
                network_embedding=embedding,
            )
            append_rows(rows, OUT_SEED_METRICS)
            print(f"Finished {label} / seed {seed}")
    return pd.read_csv(OUT_SEED_METRICS)


def final_summary(seed_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (variant, policy, metric), group in seed_metrics.groupby(["variant", "policy", "metric_name"], sort=False):
        values = group["metric_value"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "policy": policy,
                "metric_name": metric,
                "seed_count": int(group["seed"].nunique()),
                "metric_value": float(np.nanmean(values)),
                "metric_value_ci95": ci95([float(value) for value in values]),
                "cumulative_reward": float(np.nanmean(group["cumulative_reward"].to_numpy(dtype=float))),
                "cumulative_reward_ci95": ci95([float(value) for value in group["cumulative_reward"].to_numpy(dtype=float)]),
                "avg_acquired_rating": float(np.nanmean(group["avg_acquired_rating"].to_numpy(dtype=float))),
                "clip_low_share": float(np.nanmean(group["clip_low_share"].to_numpy(dtype=float))),
                "clip_high_share": float(np.nanmean(group["clip_high_share"].to_numpy(dtype=float))),
            }
        )
    return rows


def gap_summary(seed_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, group in seed_metrics.groupby("metric_name", sort=False):
        gaps: list[float] = []
        reward_gaps: list[float] = []
        for seed, seed_group in group.groupby("seed", sort=False):
            by_variant = {str(row.variant): row for row in seed_group.itertuples(index=False)}
            if "du_net" not in by_variant or "mf_linear_ucb_a002" not in by_variant:
                continue
            du = by_variant["du_net"]
            linear = by_variant["mf_linear_ucb_a002"]
            gaps.append(float(linear.metric_value - du.metric_value))
            reward_gaps.append(float(du.cumulative_reward - linear.cumulative_reward))
        rows.append(
            {
                "metric_name": metric,
                "seed_count": len(gaps),
                "linear_minus_du": float(np.nanmean(gaps)),
                "linear_minus_du_ci95": ci95(gaps),
                "du_wins": int(np.sum(np.asarray(gaps) > 0)),
                "du_minus_linear_reward": float(np.nanmean(reward_gaps)),
                "du_minus_linear_reward_ci95": ci95(reward_gaps),
                "du_reward_wins": int(np.sum(np.asarray(reward_gaps) > 0)),
            }
        )
    return rows


def row_by_metric(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    for row in rows:
        if row["metric_name"] == metric:
            return row
    raise KeyError(metric)


def write_report(final_rows: list[dict[str, object]], gap_rows: list[dict[str, object]]) -> None:
    focus_metrics = [
        "rmse_all",
        "rmse_high_rating_ge4",
        "rmse_low_rating_le3",
        "local_reward_w_rmse",
        "sigmoid_reward_w_rmse",
        "rmse_rating_eq4",
        "rmse_rating_eq5",
    ]
    by_variant_metric = {
        (str(row["variant"]), str(row["metric_name"])): row
        for row in final_rows
    }
    lines = [
        "# Figure 4a Rating-Tail 20-Seed Confirmation",
        "",
        "Generated by `experiments/build_figure4a_rating_tail_20seed_confirmation.py`.",
        "",
        "This confirms whether the 3-seed mechanism audit's high-rating win and low-rating loss persist over the paper-scale 20 random initializations.",
        "Gap convention: `Linear - DU`; positive error gaps favor DU-Net.",
        "",
        "## Metric Gaps",
        "",
        "| metric | DU-Net | Linear UCB | Linear - DU | DU seed wins | interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in focus_metrics:
        gap = row_by_metric(gap_rows, metric)
        du = by_variant_metric[("du_net", metric)]
        linear = by_variant_metric[("mf_linear_ucb_a002", metric)]
        interpretation = "favors DU-Net" if float(gap["linear_minus_du"]) > 0 else "favors Linear UCB"
        if abs(float(gap["linear_minus_du"])) < 0.001:
            interpretation = "near tie"
        lines.append(
            f"| {metric} | {fmt(du['metric_value'])} | {fmt(linear['metric_value'])} | "
            f"{fmt(gap['linear_minus_du'])} | {gap['du_wins']}/{gap['seed_count']} | {interpretation} |"
        )
    reward_gap = row_by_metric(gap_rows, "rmse_all")
    high_gap = row_by_metric(gap_rows, "rmse_high_rating_ge4")
    low_gap = row_by_metric(gap_rows, "rmse_low_rating_le3")
    sigmoid_gap = row_by_metric(gap_rows, "sigmoid_reward_w_rmse")
    local_gap = row_by_metric(gap_rows, "local_reward_w_rmse")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Full-test RMSE remains Linear-UCB-favored: gap {fmt(reward_gap['linear_minus_du'])}, DU-Net wins {reward_gap['du_wins']}/{reward_gap['seed_count']} seeds.",
            f"- High-rating RMSE is DU-Net-favored at 20 seeds: gap {fmt(high_gap['linear_minus_du'])}, DU-Net wins {high_gap['du_wins']}/{high_gap['seed_count']} seeds.",
            f"- Low-rating RMSE remains Linear-UCB-favored: gap {fmt(low_gap['linear_minus_du'])}, DU-Net wins {low_gap['du_wins']}/{low_gap['seed_count']} seeds.",
            f"- Reward-weighted metrics split: local weighted gap {fmt(local_gap['linear_minus_du'])}; sigmoid gap {fmt(sigmoid_gap['linear_minus_du'])}.",
            f"- DU-Net cumulative reward advantage is {fmt(reward_gap['du_minus_linear_reward'])}, with DU-Net winning reward on {reward_gap['du_reward_wins']}/{reward_gap['seed_count']} seeds.",
            "",
            "CSV artifacts:",
            "",
            f"- `{OUT_FINAL.relative_to(ROOT)}`",
            f"- `{OUT_GAPS.relative_to(ROOT)}`",
            f"- `{OUT_SEED_METRICS.relative_to(ROOT)}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        seed_metrics = pd.read_csv(OUT_SEED_METRICS)
    else:
        config = load_json(CONFIG_PATH)
        seeds = args.seeds if args.seeds else [int(seed) for seed in config["seeds"]]
        seed_metrics = run_confirmation(seeds=seeds, resume=args.resume)
    final_rows = final_summary(seed_metrics)
    gap_rows = gap_summary(seed_metrics)
    write_csv(final_rows, OUT_FINAL)
    write_csv(gap_rows, OUT_GAPS)
    write_report(final_rows, gap_rows)
    print(f"Wrote {OUT_FINAL}")
    print(f"Wrote {OUT_GAPS}")
    print(f"Wrote {OUT_SEED_METRICS}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
