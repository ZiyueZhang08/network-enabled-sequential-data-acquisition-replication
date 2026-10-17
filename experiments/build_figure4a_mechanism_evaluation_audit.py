"""Audit Figure 4a mechanism/evaluation choices on MovieLens trajectories."""

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

from src.acquisition import (  # noqa: E402
    AcquisitionConfig,
    _mf_linear_ucb_scores,
    _policy_scores,
    _select_rows,
)
from src.data_loader import load_dataset_from_config, split_dataset  # noqa: E402
from src.dls import compute_dls_neighbors, smooth_dls_user_factors  # noqa: E402
from src.graph_embedding import user_embedding  # noqa: E402
from src.matrix_factorization import MFConfig, MatrixFactorization  # noqa: E402
from src.reward import true_centered_reward  # noqa: E402
from src.utils import load_json  # noqa: E402


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"
OUT_METRICS = TABLE_DIR / "figure4a_mechanism_evaluation_audit_metrics.csv"
OUT_FINAL = TABLE_DIR / "figure4a_mechanism_evaluation_audit_final.csv"
OUT_GAPS = TABLE_DIR / "figure4a_mechanism_evaluation_audit_gaps.csv"
OUT_TRAJECTORY = TABLE_DIR / "figure4a_mechanism_evaluation_audit_trajectory.csv"
OUT_REPORT = REPORT_DIR / "figure4a_mechanism_evaluation_audit.md"

SEEDS = [0, 1, 2]
VARIANTS = [
    {"label": "du_net", "policy": "du_net", "acquisition_overrides": {}},
    {"label": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "acquisition_overrides": {"ucb_alpha": 0.02}},
]


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def raw_predict(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    user_factors: np.ndarray,
    clipped: bool,
) -> np.ndarray:
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    preds = (
        model.global_mean
        + model.user_bias[users]
        + model.item_bias[items]
        + np.sum(user_factors[users] * model.item_factors[items], axis=1)
    )
    if clipped:
        preds = np.clip(preds, 1.0, 5.0)
    return preds


def weighted_error(truth: np.ndarray, preds: np.ndarray, weights: np.ndarray) -> float:
    weights = weights.astype(float)
    if len(weights) == 0 or float(np.sum(weights)) <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(weights * (truth - preds) ** 2) / np.sum(weights)))


def eval_rows(
    *,
    variant: str,
    policy: str,
    seed: int,
    cycle: int,
    observed_size: int,
    candidate_size: int,
    cumulative_reward: float,
    initial_size: int,
    model: MatrixFactorization,
    test: pd.DataFrame,
    base_user_factors: np.ndarray,
    dls_user_factors: np.ndarray,
    midpoint: float,
) -> list[dict[str, object]]:
    truth = test["rating"].to_numpy(dtype=float)
    factors = {
        "raw_mf_eval": base_user_factors,
        "dls_eval": dls_user_factors,
        "current_policy_eval": dls_user_factors if policy == "du_net" else base_user_factors,
    }
    metric_weights = {
        "rmse_all": np.ones(len(truth)),
        "rmse_high_rating_ge4": (truth >= 4.0).astype(float),
        "rmse_low_rating_le3": (truth <= 3.0).astype(float),
        "local_reward_w_rmse": np.maximum(truth - midpoint, 0.0) + 0.1,
        "sigmoid_reward_w_rmse": 1.0 / (1.0 + np.exp(-(truth - midpoint))),
    }
    rows: list[dict[str, object]] = []
    acquired_count = observed_size - initial_size
    avg_acquired_rating = midpoint + cumulative_reward / acquired_count if acquired_count > 0 else float("nan")
    for eval_mode, user_factors in factors.items():
        for prediction_treatment, clipped in [("raw_prediction", False), ("clipped_prediction", True)]:
            preds = raw_predict(model, test, user_factors, clipped=clipped)
            clip_low_share = float(np.mean(preds <= 1.0)) if clipped else float(np.mean(preds < 1.0))
            clip_high_share = float(np.mean(preds >= 5.0)) if clipped else float(np.mean(preds > 5.0))
            for metric_name, weights in metric_weights.items():
                rows.append(
                    {
                        "variant": variant,
                        "policy": policy,
                        "seed": seed,
                        "cycle": cycle,
                        "observed_size": observed_size,
                        "candidate_size": candidate_size,
                        "cumulative_reward": cumulative_reward,
                        "avg_acquired_rating": avg_acquired_rating,
                        "eval_mode": eval_mode,
                        "prediction_treatment": prediction_treatment,
                        "metric_name": metric_name,
                        "metric_value": weighted_error(truth, preds, weights),
                        "clip_low_share": clip_low_share,
                        "clip_high_share": clip_high_share,
                    }
                )
    return rows


def run_variant(
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
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    label = str(variant["label"])
    policy = str(variant["policy"])
    acquisition = replace(base_acquisition, **variant.get("acquisition_overrides", {}))
    rng = np.random.default_rng(seed)
    observed = split.initial.copy()
    candidates = split.candidate.copy()
    initial_size = len(observed)
    cumulative_reward = 0.0
    metric_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    dls_neighbor_idx, dls_neighbor_dist_sq = compute_dls_neighbors(
        network_embedding,
        top_k=acquisition.dls_top_k,
    )
    for cycle in range(acquisition.cycles + 1):
        model = MatrixFactorization(num_users, num_items, mf_config, seed=seed * 1009 + cycle).fit(observed)
        base_user_factors = model.user_representations()
        dls_user_factors, effective_n = smooth_dls_user_factors(
            base_user_factors,
            dls_neighbor_idx,
            dls_neighbor_dist_sq,
            sigma=acquisition.sigma,
        )
        metric_rows.extend(
            eval_rows(
                variant=label,
                policy=policy,
                seed=seed,
                cycle=cycle,
                observed_size=len(observed),
                candidate_size=len(candidates),
                cumulative_reward=cumulative_reward,
                initial_size=initial_size,
                model=model,
                test=split.test,
                base_user_factors=base_user_factors,
                dls_user_factors=dls_user_factors,
                midpoint=acquisition.rating_midpoint,
            )
        )
        if cycle == acquisition.cycles or candidates.empty:
            break
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
        selected_reward = true_centered_reward(selected, midpoint=acquisition.rating_midpoint)
        cumulative_reward += selected_reward
        trajectory_rows.append(
            {
                "variant": label,
                "policy": policy,
                "seed": seed,
                "cycle": cycle,
                "selected_count": len(selected),
                "selected_mean_rating": float(selected["rating"].mean()) if len(selected) else float("nan"),
                "selected_centered_reward": selected_reward,
                "cumulative_reward_after_selection": cumulative_reward,
                "effective_neighborhood_mean": float(np.mean(effective_n)),
            }
        )
        observed = pd.concat([observed, selected], ignore_index=True)
        candidates = candidates.drop(candidates.index[selected_pos]).reset_index(drop=True)
    return metric_rows, trajectory_rows


def run_audit(resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_json(CONFIG_PATH)
    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    completed: set[tuple[str, int]] = set()
    metric_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []
    if resume and OUT_METRICS.exists() and OUT_TRAJECTORY.exists():
        metrics = pd.read_csv(OUT_METRICS)
        trajectory = pd.read_csv(OUT_TRAJECTORY)
        metric_frames.append(metrics)
        trajectory_frames.append(trajectory)
        completed = {
            (str(row.variant), int(row.seed))
            for row in metrics[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
    wrote_metrics = resume and OUT_METRICS.exists()
    wrote_trajectory = resume and OUT_TRAJECTORY.exists()
    for seed in SEEDS:
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
            metric_rows, trajectory_rows = run_variant(
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
            metric_frame = pd.DataFrame(metric_rows)
            trajectory_frame = pd.DataFrame(trajectory_rows)
            metric_frames.append(metric_frame)
            trajectory_frames.append(trajectory_frame)
            metric_frame.to_csv(OUT_METRICS, mode="w" if not wrote_metrics else "a", header=not wrote_metrics, index=False)
            trajectory_frame.to_csv(
                OUT_TRAJECTORY,
                mode="w" if not wrote_trajectory else "a",
                header=not wrote_trajectory,
                index=False,
            )
            wrote_metrics = True
            wrote_trajectory = True
            print(f"Finished {label} / seed {seed}")
    return pd.concat(metric_frames, ignore_index=True), pd.concat(trajectory_frames, ignore_index=True)


def final_summary(metrics: pd.DataFrame) -> list[dict[str, object]]:
    final_cycle = int(metrics["cycle"].max())
    final = metrics[metrics["cycle"] == final_cycle]
    rows: list[dict[str, object]] = []
    group_cols = ["variant", "policy", "eval_mode", "prediction_treatment", "metric_name"]
    for keys, group in final.groupby(group_cols, sort=False):
        variant, policy, eval_mode, prediction_treatment, metric_name = keys
        values = group["metric_value"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "policy": policy,
                "eval_mode": eval_mode,
                "prediction_treatment": prediction_treatment,
                "metric_name": metric_name,
                "seed_count": int(group["seed"].nunique()),
                "metric_value": float(np.nanmean(values)),
                "metric_value_ci95": ci95([float(value) for value in values if not np.isnan(value)]),
                "clip_low_share": float(np.nanmean(group["clip_low_share"].to_numpy(dtype=float))),
                "clip_high_share": float(np.nanmean(group["clip_high_share"].to_numpy(dtype=float))),
                "cumulative_reward": float(np.nanmean(group["cumulative_reward"].to_numpy(dtype=float))),
                "avg_acquired_rating": float(np.nanmean(group["avg_acquired_rating"].to_numpy(dtype=float))),
            }
        )
    return rows


def gap_summary(metrics: pd.DataFrame) -> list[dict[str, object]]:
    final_cycle = int(metrics["cycle"].max())
    final = metrics[metrics["cycle"] == final_cycle]
    rows: list[dict[str, object]] = []
    group_cols = ["eval_mode", "prediction_treatment", "metric_name"]
    for keys, group in final.groupby(group_cols, sort=False):
        eval_mode, prediction_treatment, metric_name = keys
        seed_gaps: list[float] = []
        for seed, seed_group in group.groupby("seed", sort=False):
            by_variant = {str(row.variant): float(row.metric_value) for row in seed_group.itertuples(index=False)}
            if "du_net" not in by_variant or "mf_linear_ucb_a002" not in by_variant:
                continue
            seed_gaps.append(by_variant["mf_linear_ucb_a002"] - by_variant["du_net"])
        rows.append(
            {
                "eval_mode": eval_mode,
                "prediction_treatment": prediction_treatment,
                "metric_name": metric_name,
                "seed_count": len(seed_gaps),
                "linear_minus_du": float(np.nanmean(seed_gaps)),
                "linear_minus_du_ci95": ci95([gap for gap in seed_gaps if not np.isnan(gap)]),
                "du_wins": int(np.sum(np.asarray(seed_gaps) > 0)),
            }
        )
    return rows


def trajectory_summary(trajectory: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant, group in trajectory.groupby("variant", sort=False):
        final_by_seed = group.sort_values("cycle").groupby("seed", as_index=False).tail(1)
        rows.append(
            {
                "variant": variant,
                "seed_count": int(group["seed"].nunique()),
                "selected_mean_rating": float(group["selected_mean_rating"].mean()),
                "selected_mean_rating_ci95": ci95(
                    [float(value) for value in final_by_seed["selected_mean_rating"].to_numpy(dtype=float)]
                ),
                "final_cumulative_reward": float(final_by_seed["cumulative_reward_after_selection"].mean()),
                "final_cumulative_reward_ci95": ci95(
                    [float(value) for value in final_by_seed["cumulative_reward_after_selection"].to_numpy(dtype=float)]
                ),
            }
        )
    return rows


def lookup_gap(gaps: list[dict[str, object]], eval_mode: str, treatment: str, metric: str) -> dict[str, object]:
    for row in gaps:
        if (
            row["eval_mode"] == eval_mode
            and row["prediction_treatment"] == treatment
            and row["metric_name"] == metric
        ):
            return row
    raise KeyError((eval_mode, treatment, metric))


def write_report(final_rows: list[dict[str, object]], gaps: list[dict[str, object]], trajectory_rows: list[dict[str, object]]) -> None:
    focus = [
        ("current_policy_eval", "clipped_prediction", "rmse_all"),
        ("raw_mf_eval", "clipped_prediction", "rmse_all"),
        ("dls_eval", "clipped_prediction", "rmse_all"),
        ("current_policy_eval", "raw_prediction", "rmse_all"),
        ("current_policy_eval", "clipped_prediction", "sigmoid_reward_w_rmse"),
        ("current_policy_eval", "clipped_prediction", "local_reward_w_rmse"),
        ("current_policy_eval", "clipped_prediction", "rmse_high_rating_ge4"),
        ("current_policy_eval", "clipped_prediction", "rmse_low_rating_le3"),
    ]
    lines = [
        "# Figure 4a Mechanism/Evaluation Audit",
        "",
        "Generated by `experiments/build_figure4a_mechanism_evaluation_audit.py`.",
        "",
        "This audit reruns the current MovieLens Figure 4a DU-Net and MF-centered Linear UCB trajectories on seeds 0-2.",
        "For the same trajectories it evaluates raw-MF factors, DLS-smoothed factors, raw predictions, clipped predictions, and reward/test-distribution weightings.",
        "",
        "Gap convention: `Linear - DU`; positive error gaps favor DU-Net.",
        "",
        "## Focus Gaps",
        "",
        "| eval mode | prediction | metric | Linear - DU | DU seed wins | interpretation |",
        "|---|---|---|---:|---:|---|",
    ]
    for eval_mode, treatment, metric in focus:
        gap = lookup_gap(gaps, eval_mode, treatment, metric)
        if float(gap["linear_minus_du"]) > 0:
            interpretation = "favors DU-Net"
        elif abs(float(gap["linear_minus_du"])) < 0.001:
            interpretation = "near tie"
        else:
            interpretation = "favors Linear UCB"
        lines.append(
            f"| {eval_mode} | {treatment} | {metric} | "
            f"{fmt(gap['linear_minus_du'])} | {gap['du_wins']}/{gap['seed_count']} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory Summary",
            "",
            "| variant | avg selected rating | final reward |",
            "|---|---:|---:|",
        ]
    )
    for row in trajectory_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['selected_mean_rating'])} | {fmt(row['final_cumulative_reward'])} |"
        )
    status_gap = lookup_gap(gaps, "current_policy_eval", "clipped_prediction", "rmse_all")
    dls_gap = lookup_gap(gaps, "dls_eval", "clipped_prediction", "rmse_all")
    sigmoid_gap = lookup_gap(gaps, "current_policy_eval", "clipped_prediction", "sigmoid_reward_w_rmse")
    high_gap = lookup_gap(gaps, "current_policy_eval", "clipped_prediction", "rmse_high_rating_ge4")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Status-quo clipped conventional RMSE gap is {fmt(status_gap['linear_minus_du'])}; this still favors Linear UCB.",
            f"- Giving both policies DLS-smoothed evaluation factors changes the conventional RMSE gap to {fmt(dls_gap['linear_minus_du'])}; if this remains negative, DLS evaluation alone is not the missing mechanism.",
            f"- Sigmoid reward-weighted gap is {fmt(sigmoid_gap['linear_minus_du'])}, while high-rating-only RMSE gap is {fmt(high_gap['linear_minus_du'])}. This separates reward-region recovery from global test-set RMSE.",
            "- Raw-vs-clipped prediction rows show whether range clipping is materially changing the conclusion; compare the paired rows in the CSV.",
            "",
            "CSV artifacts:",
            "",
            f"- `{OUT_FINAL.relative_to(ROOT)}`",
            f"- `{OUT_GAPS.relative_to(ROOT)}`",
            f"- `{OUT_TRAJECTORY.relative_to(ROOT)}`",
            f"- `{OUT_METRICS.relative_to(ROOT)}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        metrics = pd.read_csv(OUT_METRICS)
        trajectory = pd.read_csv(OUT_TRAJECTORY)
    else:
        metrics, trajectory = run_audit(resume=args.resume)
    final_rows = final_summary(metrics)
    gap_rows = gap_summary(metrics)
    trajectory_rows = trajectory_summary(trajectory)
    write_csv(final_rows, OUT_FINAL)
    write_csv(gap_rows, OUT_GAPS)
    write_report(final_rows, gap_rows, trajectory_rows)
    print(f"Wrote {OUT_FINAL}")
    print(f"Wrote {OUT_GAPS}")
    print(f"Wrote {OUT_TRAJECTORY}")
    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
