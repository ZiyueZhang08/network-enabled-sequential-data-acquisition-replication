from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, _dls_neighbors_for_config, _du_net_combined_score, _select_rows
from src.data_loader import RATING_COLUMNS, RatingSplit, load_movielens_100k
from src.dls_paper import scoped_gaussian_dls_user_factors
from src.entropy_schur import schur_conditional_variance_entropy_scores
from src.matrix_factorization import MFConfig, MatrixFactorization
from src.metrics import standard_mask_rmse
from src.reward import predicted_reward_scores
from src.tuning import tune_dunet_params_paper_style
from src.utils import load_json


PREFIX = "paper_exact_attempt_movielens_v2"
OUT_CSV = ROOT / "outputs" / "tables" / f"{PREFIX}_schur_shortlist_vs_full_audit.csv"
OUT_REPORT = ROOT / "reports" / f"{PREFIX}_schur_shortlist_vs_full_audit.md"


def read_fold(path: Path, num_items: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"], engine="python")
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    frame = frame[frame["item_id"] < int(num_items)].copy()
    return frame[RATING_COLUMNS].reset_index(drop=True)


def make_u1_fallback_split(raw_dir: Path, seed: int, initial_size: int, num_items: int) -> RatingSplit:
    train = read_fold(raw_dir / "u1.base", num_items=num_items)
    test = read_fold(raw_dir / "u1.test", num_items=num_items)
    rng = np.random.default_rng(seed)
    size = min(int(initial_size), len(train))
    initial_idx = rng.choice(len(train), size=size, replace=False)
    mask = np.zeros(len(train), dtype=bool)
    mask[initial_idx] = True
    return RatingSplit(
        initial=train.loc[mask, RATING_COLUMNS].reset_index(drop=True),
        candidate=train.loc[~mask, RATING_COLUMNS].reset_index(drop=True),
        test=test[RATING_COLUMNS].reset_index(drop=True),
    )


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            view[col] = view[col].fillna("").astype(str)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return float(len(set(map(int, a)).intersection(set(map(int, b)))) / max(min(len(a), len(b)), 1))


def rng_from_state(state: dict[str, object]) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng


def one_step_standard_rmse(
    selected_pos: np.ndarray,
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    split: RatingSplit,
    network_embedding: np.ndarray,
    dls_neighbor_idx: np.ndarray,
    dls_neighbor_dist_sq: np.ndarray,
    num_users: int,
    num_items: int,
    mf_config: MFConfig,
    acquisition: AcquisitionConfig,
    seed: int,
    cycle: int,
) -> float:
    if len(selected_pos) == 0:
        return np.nan
    selected = candidates.iloc[selected_pos].copy()
    alt_observed = pd.concat([observed, selected], ignore_index=True)
    next_model = MatrixFactorization(num_users, num_items, mf_config, seed=seed * 1009 + cycle + 1).fit(alt_observed)
    alt_user_factors, _ = scoped_gaussian_dls_user_factors(
        next_model.user_representations(),
        network_embedding,
        dls_neighbor_idx,
        dls_neighbor_dist_sq,
        sigma=float(acquisition.sigma),
        scope=str(acquisition.paper_dls_scope),
    )
    return standard_mask_rmse(next_model, split.test, user_factors=alt_user_factors)


def tune_cycle_config(
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    network_embedding: np.ndarray,
    acquisition: AcquisitionConfig,
    latest_config: AcquisitionConfig,
    mf_config: MFConfig,
    num_users: int,
    num_items: int,
    seed: int,
    cycle: int,
) -> AcquisitionConfig:
    tuning_frequency = int(acquisition.paper_tuning_frequency)
    if tuning_frequency <= 0 or cycle >= acquisition.cycles or candidates.empty or cycle % tuning_frequency != 0:
        return latest_config
    tune_result = tune_dunet_params_paper_style(
        observed,
        network_embedding,
        candidates,
        {
            "lambda_grid": acquisition.adaptive_lambda_grid,
            "sigma_grid": acquisition.sigma_grid,
            "p_grid": (0.0,),
            "batch_size": acquisition.batch_size,
            "dls_top_k": acquisition.dls_top_k,
            "paper_dls_scope": acquisition.paper_dls_scope,
            "rating_midpoint": acquisition.rating_midpoint,
            "selection": acquisition.selection,
            "selection_temperature": acquisition.selection_temperature,
            "entropy_regularization": acquisition.entropy_logdet_regularization,
            "entropy_correction_weight": acquisition.paper_entropy_correction_weight,
            "entropy_shortlist_size": acquisition.paper_entropy_shortlist_size,
            "entropy_batch_size": acquisition.paper_entropy_batch_size,
            "entropy_mode": acquisition.entropy_mode,
            "entropy_score_transform": acquisition.entropy_score_transform,
        },
        {
            "rank": mf_config.rank,
            "epochs": mf_config.epochs,
            "learning_rate": mf_config.learning_rate,
            "regularization": mf_config.regularization,
            "algorithm": mf_config.algorithm,
            "num_users": num_users,
            "num_items": num_items,
            "batch_size": acquisition.batch_size,
            "rating_midpoint": acquisition.rating_midpoint,
            "selection": acquisition.selection,
            "selection_temperature": acquisition.selection_temperature,
            "entropy_score_transform": acquisition.entropy_score_transform,
        },
        acquisition.paper_tuning_metric,
        seed=seed * 1009 + cycle,
    )
    if not np.isfinite(float(tune_result["validation_loss"])):
        return latest_config
    return replace(
        latest_config,
        lambda_entropy=float(tune_result["lambda_entropy"]),
        sigma=float(tune_result["sigma"]),
        diffusion_prob=0.0,
    )


def score_short_and_full(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    model: MatrixFactorization,
    user_factors: np.ndarray,
    acquisition: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reward = predicted_reward_scores(model, candidates, user_factors=user_factors)
    entropy_short = schur_conditional_variance_entropy_scores(
        candidates,
        observed,
        user_factors,
        model.item_factors,
        lambda_noise=float(acquisition.entropy_logdet_regularization),
        shortlist_size=int(acquisition.paper_entropy_shortlist_size),
        batch_size=int(acquisition.paper_entropy_batch_size),
    )
    entropy_full = schur_conditional_variance_entropy_scores(
        candidates,
        observed,
        user_factors,
        model.item_factors,
        lambda_noise=float(acquisition.entropy_logdet_regularization),
        shortlist_size=0,
        batch_size=int(acquisition.paper_entropy_batch_size),
    )
    score_short = _du_net_combined_score(reward, entropy_short, acquisition)
    score_full = _du_net_combined_score(reward, entropy_full, acquisition)
    return reward, entropy_short, entropy_full, score_short, score_full


def audit_cycle(
    seed: int,
    cycle: int,
    split: RatingSplit,
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    model: MatrixFactorization,
    user_factors: np.ndarray,
    network_embedding: np.ndarray,
    dls_neighbor_idx: np.ndarray,
    dls_neighbor_dist_sq: np.ndarray,
    num_users: int,
    num_items: int,
    mf_config: MFConfig,
    acquisition: AcquisitionConfig,
    rng_state: dict[str, object],
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    reward, entropy_short, entropy_full, score_short, score_full = score_short_and_full(
        candidates,
        observed,
        model,
        user_factors,
        acquisition,
    )
    top100_short = np.argsort(score_short)[-100:]
    top100_full = np.argsort(score_full)[-100:]
    rng_short = rng_from_state(rng_state)
    rng_full = rng_from_state(rng_state)
    batch_short = _select_rows(
        score_short,
        int(acquisition.batch_size),
        rng_short,
        str(acquisition.selection),
        temperature=float(acquisition.selection_temperature),
    )
    batch_full = _select_rows(
        score_full,
        int(acquisition.batch_size),
        rng_full,
        str(acquisition.selection),
        temperature=float(acquisition.selection_temperature),
    )
    changed = sorted(set(map(int, batch_short)).symmetric_difference(set(map(int, batch_full))))
    if set(map(int, batch_short)) == set(map(int, batch_full)):
        one_step_short = 0.0
        one_step_full = 0.0
        one_step_diff = 0.0
    else:
        one_step_short = one_step_standard_rmse(
            batch_short,
            observed,
            candidates,
            split,
            network_embedding,
            dls_neighbor_idx,
            dls_neighbor_dist_sq,
            num_users,
            num_items,
            mf_config,
            acquisition,
            seed,
            cycle,
        )
        one_step_full = one_step_standard_rmse(
            batch_full,
            observed,
            candidates,
            split,
            network_embedding,
            dls_neighbor_idx,
            dls_neighbor_dist_sq,
            num_users,
            num_items,
            mf_config,
            acquisition,
            seed,
            cycle,
        )
        one_step_diff = float(one_step_full - one_step_short)
    row = {
        "seed": seed,
        "cycle": cycle,
        "candidate_count": int(len(candidates)),
        "shortlist_size": int(acquisition.paper_entropy_shortlist_size),
        "top100_overlap": overlap(top100_short, top100_full),
        "selected_batch_overlap": overlap(batch_short, batch_full),
        "selected_changed_count": int(len(changed)),
        "mean_entropy_short_selected": float(np.mean(entropy_short[batch_short])) if len(batch_short) else np.nan,
        "mean_entropy_full_selected": float(np.mean(entropy_full[batch_full])) if len(batch_full) else np.nan,
        "selected_entropy_diff_full_minus_short": (
            float(np.mean(entropy_full[batch_full]) - np.mean(entropy_short[batch_short])) if len(batch_short) and len(batch_full) else np.nan
        ),
        "mean_rating_short_selected": float(np.mean(candidates.iloc[batch_short]["rating"])) if len(batch_short) else np.nan,
        "mean_rating_full_selected": float(np.mean(candidates.iloc[batch_full]["rating"])) if len(batch_full) else np.nan,
        "selected_rating_diff_full_minus_short": (
            float(np.mean(candidates.iloc[batch_full]["rating"]) - np.mean(candidates.iloc[batch_short]["rating"]))
            if len(batch_short) and len(batch_full)
            else np.nan
        ),
        "one_step_standard_rmse_short": one_step_short,
        "one_step_standard_rmse_full": one_step_full,
        "one_step_standard_rmse_diff_full_minus_short": one_step_diff,
        "mean_score_abs_diff": float(np.mean(np.abs(score_short - score_full))),
        "max_score_abs_diff": float(np.max(np.abs(score_short - score_full))),
        "mean_entropy_abs_diff": float(np.mean(np.abs(entropy_short - entropy_full))),
        "max_entropy_abs_diff": float(np.max(np.abs(entropy_short - entropy_full))),
        "full_entropy_nonzero_share": float(np.mean(entropy_full > 0)),
        "short_entropy_nonzero_share": float(np.mean(entropy_short > 0)),
        "lambda_entropy": float(acquisition.lambda_entropy),
        "sigma": float(acquisition.sigma),
        "audit_scope": "cycles_0_5_10_15_20_frozen_v2_mainline_replay",
    }
    return row, batch_short, copy.deepcopy(rng_short.bit_generator.state)


def run_seed(seed: int, config: dict, audit_cycles: set[int]) -> list[dict[str, object]]:
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]
    raw_dir = ROOT / dataset_cfg["raw_dir"]
    dataset = load_movielens_100k(raw_dir, network_mode=dataset_cfg.get("network_mode", "gcmc_features"))
    if dataset.user_features is None:
        raise ValueError("MovieLens user_features are required for this audit.")

    split = make_u1_fallback_split(
        raw_dir,
        seed=seed,
        initial_size=int(split_cfg["initial_size"]),
        num_items=int(split_cfg["num_items"]),
    )
    mf_config = MFConfig(**config["mf"])
    acquisition = AcquisitionConfig(**config["acquisition"])
    num_users = int(split_cfg["num_users"])
    num_items = int(split_cfg["num_items"])
    dls_neighbor_idx, dls_neighbor_dist_sq = _dls_neighbors_for_config(
        dataset.user_features,
        dataset.adjacency,
        acquisition,
    )
    rng = np.random.default_rng(seed)
    observed = split.initial.copy()
    candidates = split.candidate.copy()
    latest_config = acquisition
    rows: list[dict[str, object]] = []
    for cycle in range(acquisition.cycles + 1):
        model = MatrixFactorization(num_users, num_items, mf_config, seed=seed * 1009 + cycle).fit(observed)
        cycle_config = tune_cycle_config(
            observed,
            candidates,
            dataset.user_features,
            acquisition,
            latest_config,
            mf_config,
            num_users,
            num_items,
            seed,
            cycle,
        )
        latest_config = cycle_config
        dls_user_factors, _ = scoped_gaussian_dls_user_factors(
            model.user_representations(),
            dataset.user_features,
            dls_neighbor_idx,
            dls_neighbor_dist_sq,
            sigma=float(cycle_config.sigma),
            scope=str(cycle_config.paper_dls_scope),
        )
        if cycle in audit_cycles and not candidates.empty:
            row, selected_pos, next_state = audit_cycle(
                seed,
                cycle,
                split,
                observed,
                candidates,
                model,
                dls_user_factors,
                dataset.user_features,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                num_users,
                num_items,
                mf_config,
                cycle_config,
                rng.bit_generator.state,
            )
            rows.append(row)
            rng.bit_generator.state = next_state
        else:
            _, _, _, score_short, _ = score_short_and_full(candidates, observed, model, dls_user_factors, cycle_config)
            selected_pos = _select_rows(
                score_short,
                int(cycle_config.batch_size),
                rng,
                str(cycle_config.selection),
                temperature=float(cycle_config.selection_temperature),
            )
        if cycle == acquisition.cycles or candidates.empty:
            break
        selected = candidates.iloc[selected_pos].copy()
        observed = pd.concat([observed, selected], ignore_index=True)
        candidates = candidates.drop(candidates.index[selected_pos]).reset_index(drop=True)
    return rows


def write_report(rows: pd.DataFrame) -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        rows.groupby("cycle", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            min_top100_overlap=("top100_overlap", "min"),
            min_selected_batch_overlap=("selected_batch_overlap", "min"),
            max_selected_changed_count=("selected_changed_count", "max"),
            max_abs_one_step_rmse_diff=("one_step_standard_rmse_diff_full_minus_short", lambda value: float(np.max(np.abs(value)))),
            max_abs_rating_diff=("selected_rating_diff_full_minus_short", lambda value: float(np.max(np.abs(value)))),
        )
        .sort_values("cycle")
        if not rows.empty
        else pd.DataFrame()
    )
    lines = [
        "# Schur Shortlist vs Full Audit v2",
        "",
        "Scope: Phase 4 diagnostic for the 20k Schur shortlist approximation. This replays the frozen v2 tau=1 mainline and compares shortlist-vs-full Schur scoring at selected cycles.",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Results",
        "",
        md_table(rows),
        "",
        "## Decision",
        "",
        "If top100 or selected-batch overlap is below 0.95, or one-step RMSE changes materially, the shortlist approximation remains a live exactness risk and requires a production all-candidate rerun. This audit still does not replace an end-to-end full-candidate 20-cycle run.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / f"{PREFIX}.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cycles", nargs="+", type=int, default=[0, 5, 10, 15, 20])
    args = parser.parse_args()

    config = load_json(args.config)
    audit_cycles = {int(cycle) for cycle in args.cycles}
    rows = pd.DataFrame([row for seed in args.seeds for row in run_seed(seed, config, audit_cycles)])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
