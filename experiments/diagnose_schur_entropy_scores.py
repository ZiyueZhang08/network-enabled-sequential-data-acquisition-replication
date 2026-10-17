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

from experiments.run_paper_faithful_movielens import make_u1_fallback_split
from src.acquisition import (
    AcquisitionConfig,
    _policy_scores,
    _rank_percentiles,
    _select_rows,
    _transform_entropy_score,
)
from src.data_loader import load_movielens_100k
from src.dls import compute_dls_neighbors
from src.dls_paper import scoped_gaussian_dls_user_factors
from src.matrix_factorization import MFConfig, MatrixFactorization
from src.utils import ensure_dir, load_json, zscore


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _diagnose_seed(
    seed: int,
    config: dict,
    dataset,
    network_embedding: np.ndarray,
) -> list[dict[str, float | int | str]]:
    split_cfg = config["split"]
    raw_dir = ROOT / config["dataset"]["raw_dir"]
    split = make_u1_fallback_split(
        raw_dir,
        seed=int(seed),
        initial_size=int(split_cfg.get("initial_size", 1364)),
        num_items=int(split_cfg.get("num_items", 1681)),
    )
    mf_config = MFConfig(**config["mf"])
    acquisition = AcquisitionConfig(**config["acquisition"])
    acquisition = replace(acquisition, paper_tuning_frequency=0)
    num_users = int(split_cfg.get("num_users", dataset.num_users))
    num_items = int(split_cfg.get("num_items", dataset.num_items))
    neighbor_idx, neighbor_dist_sq = compute_dls_neighbors(network_embedding, top_k=acquisition.dls_top_k)
    rng = np.random.default_rng(seed)
    observed = split.initial.copy()
    candidates = split.candidate.copy()
    rows: list[dict[str, float | int | str]] = []

    for cycle in range(acquisition.cycles):
        model = MatrixFactorization(num_users, num_items, mf_config, seed=seed * 1009 + cycle).fit(observed)
        user_dls, _ = scoped_gaussian_dls_user_factors(
            model.user_representations(),
            network_embedding,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=acquisition.sigma,
            scope=acquisition.paper_dls_scope,
        )
        scores = _policy_scores(
            "paper_faithful_dunet",
            model,
            candidates,
            observed,
            dataset.adjacency,
            num_users,
            num_items,
            user_dls,
            acquisition,
        )
        reward_z = zscore(scores["reward_score"])
        entropy_z = zscore(_transform_entropy_score(scores["entropy_score"], acquisition.entropy_score_transform))
        entropy_contribution = acquisition.lambda_entropy * entropy_z
        final_score = scores["score"]
        selected_pos = _select_rows(
            final_score,
            acquisition.batch_size,
            rng,
            acquisition.selection,
            acquisition.selection_temperature,
            acquisition.selection_top_m,
            acquisition.selection_top_fraction,
        )
        selected = candidates.iloc[selected_pos].copy()
        entropy_pct = _rank_percentiles(scores["entropy_score"])
        reward_pct = _rank_percentiles(scores["reward_score"])
        score_pct = _rank_percentiles(final_score)
        row: dict[str, float | int | str] = {
            "seed": int(seed),
            "cycle": int(cycle),
            "candidate_size": int(len(candidates)),
            "selected_size": int(len(selected)),
            "reward_entropy_corr": _corr(reward_z, entropy_z),
            "score_reward_corr": _corr(final_score, reward_z),
            "score_entropy_corr": _corr(final_score, entropy_z),
            "selected_avg_rating": float(selected["rating"].mean()) if len(selected) else np.nan,
            "selected_unique_users": int(selected["user_id"].nunique()) if len(selected) else 0,
            "selected_unique_items": int(selected["item_id"].nunique()) if len(selected) else 0,
            "selected_user_coverage": float(selected["user_id"].nunique() / len(selected)) if len(selected) else np.nan,
            "selected_item_coverage": float(selected["item_id"].nunique() / len(selected)) if len(selected) else np.nan,
            "selected_entropy_quantile": float(np.mean(entropy_pct[selected_pos])) if len(selected_pos) else np.nan,
            "selected_reward_quantile": float(np.mean(reward_pct[selected_pos])) if len(selected_pos) else np.nan,
            "selected_score_quantile": float(np.mean(score_pct[selected_pos])) if len(selected_pos) else np.nan,
            "lambda_entropy": float(acquisition.lambda_entropy),
            "sigma": float(acquisition.sigma),
            "reward_contribution_std": float(np.std(reward_z)),
            "entropy_contribution_std": float(np.std(entropy_contribution)),
            "entropy_to_reward_std_ratio": float(np.std(entropy_contribution) / max(float(np.std(reward_z)), 1e-12)),
        }
        row.update(_summary("z_reward", reward_z))
        row.update(_summary("z_entropy_schur", entropy_z))
        rows.append(row)

        observed = pd.concat([observed, selected], ignore_index=True)
        candidates = candidates.drop(candidates.index[selected_pos]).reset_index(drop=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "paper_faithful_movielens_schur_conditional_variance_3seeds.json"),
    )
    parser.add_argument("--output-prefix", default="paper_faithful_movielens_schur_score_diagnostics_3seeds")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    config = load_json(args.config)
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    dataset = load_movielens_100k(
        ROOT / config["dataset"]["raw_dir"],
        network_mode=config["dataset"].get("network_mode", "gcmc_features"),
    )
    if dataset.user_features is None:
        raise ValueError("Schur score diagnostics require MovieLens user features as DLS coordinates.")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        rows.extend(_diagnose_seed(int(seed), config, dataset, dataset.user_features))

    out_dir = ensure_dir(ROOT / "outputs" / "tables")
    by_cycle = pd.DataFrame(rows)
    by_cycle_path = out_dir / f"{args.output_prefix}_by_cycle.csv"
    by_cycle.to_csv(by_cycle_path, index=False)
    summary = by_cycle.groupby("cycle", as_index=False).mean(numeric_only=True)
    summary_path = out_dir / f"{args.output_prefix}_cycle_mean.csv"
    summary.to_csv(summary_path, index=False)
    key = summary[
        [
            "cycle",
            "reward_entropy_corr",
            "entropy_to_reward_std_ratio",
            "z_entropy_schur_std",
            "selected_avg_rating",
            "selected_user_coverage",
            "selected_item_coverage",
            "selected_entropy_quantile",
        ]
    ]
    print(f"Wrote diagnostics: {by_cycle_path}")
    print(f"Wrote cycle means: {summary_path}")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
