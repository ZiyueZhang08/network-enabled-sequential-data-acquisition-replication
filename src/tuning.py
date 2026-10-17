from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dls import compute_dls_neighbors
from .dls_paper import scoped_gaussian_dls_user_factors
from .entropy_paper import entropy_observational_lite_scores
from .entropy_schur import schur_conditional_variance_entropy_scores
from .matrix_factorization import MFConfig, MatrixFactorization
from .metrics import pdf_literal_norm_rmse, reward_tail_proxy_rmse, standard_mask_rmse
from .reward import predicted_reward_scores
from .utils import zscore


def _transform_entropy_score(entropy_score: np.ndarray, transform: str) -> np.ndarray:
    mode = str(transform or "none").lower()
    values = np.asarray(entropy_score, dtype=float)
    if mode in {"", "none"}:
        return values
    if mode in {"log", "paper_log", "positive_log"}:
        return np.log(np.maximum(values, 1e-12))
    if mode in {"log1p", "positive_log1p"}:
        return np.log1p(np.maximum(values, 0.0))
    if mode in {"shift_log1p", "shifted_log1p"}:
        shifted = values - float(np.min(values)) if len(values) else values
        return np.log1p(np.maximum(shifted, 0.0))
    if mode in {"signed_log1p", "slog1p"}:
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unknown entropy_score_transform: {transform}")


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(getattr(config, "__dict__", {}))


def _mf_config(config: Any) -> MFConfig:
    if isinstance(config, MFConfig):
        return config
    values = _config_dict(config)
    return MFConfig(
        rank=int(values.get("rank", 12)),
        epochs=int(values.get("epochs", 6)),
        learning_rate=float(values.get("learning_rate", 0.025)),
        regularization=float(values.get("regularization", 0.04)),
        algorithm=str(values.get("algorithm", "svdpp")),
    )


def _infer_shape(observed: pd.DataFrame, candidate: pd.DataFrame, user_graph: np.ndarray, config: dict[str, Any]) -> tuple[int, int]:
    num_users = int(config.get("num_users", user_graph.shape[0]))
    if "num_items" in config:
        num_items = int(config["num_items"])
    else:
        frames = [frame for frame in (observed, candidate) if not frame.empty]
        num_items = int(max(frame["item_id"].max() for frame in frames) + 1) if frames else 0
    return num_users, num_items


def _split_observed(
    observed: pd.DataFrame,
    batch_size: int,
    seed: int,
    acquisition_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(observed))
    val_test = max(1, int(round(len(observed) * float(test_fraction))))
    val_acq = max(batch_size, int(round(len(observed) * float(acquisition_fraction))))
    val_test = min(val_test, max(1, len(observed) // 4))
    val_acq = min(val_acq, max(1, (len(observed) - val_test) // 3))
    train_start = val_test + val_acq
    if len(observed) - train_start < max(20, batch_size):
        return observed.reset_index(drop=True), observed.iloc[[]].copy(), observed.iloc[[]].copy()
    test_idx = order[:val_test]
    acq_idx = order[val_test:train_start]
    train_idx = order[train_start:]
    return (
        observed.iloc[train_idx].reset_index(drop=True),
        observed.iloc[acq_idx].reset_index(drop=True),
        observed.iloc[test_idx].reset_index(drop=True),
    )


def _softmax_sample_without_replacement(
    scores: np.ndarray,
    size: int,
    rng: np.random.Generator,
    temperature: float,
) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=int)
    tau = max(float(temperature), 1e-8)
    shifted = (scores - float(np.max(scores))) / tau
    probabilities = np.exp(np.clip(shifted, -60.0, 0.0))
    total = float(probabilities.sum())
    if total <= 0.0 or not np.isfinite(total):
        probabilities = np.ones(len(scores), dtype=float) / max(len(scores), 1)
    else:
        probabilities = probabilities / total
    return rng.choice(len(scores), size=min(size, len(scores)), replace=False, p=probabilities)


def _select_inner_batch(
    scores: np.ndarray,
    size: int,
    rng: np.random.Generator,
    temperature: float,
    mode: str,
    top_m: int = 0,
    top_fraction: float = 0.0,
) -> np.ndarray:
    if size <= 0 or len(scores) == 0:
        return np.array([], dtype=int)
    batch_size = min(int(size), len(scores))
    selection = str(mode or "topk").lower()
    if selection in {"topk", "top_k"}:
        return np.argsort(scores)[-batch_size:]
    if selection in {"exp", "softmax", "exponential"}:
        return _softmax_sample_without_replacement(scores, batch_size, rng, temperature)
    if selection in {"topm_exp", "top_m_exp", "exp_topm", "exp_top_m"}:
        fraction_size = int(np.ceil(len(scores) * float(np.clip(top_fraction, 0.0, 1.0))))
        requested = max(int(top_m), fraction_size, batch_size)
        pool_size = min(len(scores), requested)
        pool_idx = np.argsort(scores)[-pool_size:]
        selected = _softmax_sample_without_replacement(scores[pool_idx], batch_size, rng, temperature)
        return pool_idx[selected]
    raise ValueError(f"Unknown paper-style tuning selection: {mode}")


def _metric_loss(
    metric_name: str,
    model: MatrixFactorization,
    validation: pd.DataFrame,
    user_factors: np.ndarray,
    midpoint: float,
) -> float:
    mode = metric_name.lower()
    if mode in {"pdf_literal", "pdf_literal_norm_rmse"}:
        return pdf_literal_norm_rmse(model, validation, user_factors=user_factors)
    if mode in {"standard", "standard_rmse", "standard_mask_rmse", "rmse"}:
        return standard_mask_rmse(model, validation, user_factors=user_factors)
    if mode in {"reward_tail_proxy", "reward_tail_proxy_rmse", "linear_reward_norm"}:
        return reward_tail_proxy_rmse(model, validation, midpoint=midpoint, user_factors=user_factors)
    raise ValueError(f"Unknown paper-style tuning metric: {metric_name}")


def tune_dunet_params_paper_style(
    observed_data: pd.DataFrame,
    user_graph: np.ndarray,
    candidate_data: pd.DataFrame,
    param_grid: dict[str, Any],
    backend_config: Any,
    metric_name: str,
    seed: int,
) -> dict[str, float | int]:
    """Section 3.5.1 style inner validation for lambda and sigma.

    The current observed ratings are partitioned into inner train,
    acquisition-validation, and test-validation sets. Each grid point selects an
    acquisition batch with reward + observational entropy, refits, and is scored
    on the held-out validation ratings.
    """
    config = _config_dict(backend_config)
    mf_config = _mf_config(backend_config)
    lambda_grid = tuple(float(v) for v in param_grid.get("lambda_grid", (0.05, 0.1, 0.25, 0.5)))
    sigma_grid = tuple(float(v) for v in param_grid.get("sigma_grid", (0.1, 0.2, 0.4, 0.8, 1.6)))
    p_grid = tuple(float(v) for v in param_grid.get("p_grid", (0.0,)))
    batch_size = int(param_grid.get("batch_size", config.get("batch_size", 100)))
    top_k = int(param_grid.get("dls_top_k", config.get("dls_top_k", 30)))
    dls_scope = str(param_grid.get("paper_dls_scope", config.get("paper_dls_scope", "neighbors")))
    midpoint = float(param_grid.get("rating_midpoint", config.get("rating_midpoint", 3.0)))
    selection_mode = str(param_grid.get("selection", config.get("selection", "topk")))
    temperature = float(param_grid.get("selection_temperature", config.get("selection_temperature", 1.0)))
    selection_top_m = int(param_grid.get("selection_top_m", config.get("selection_top_m", 0)))
    selection_top_fraction = float(param_grid.get("selection_top_fraction", config.get("selection_top_fraction", 0.0)))
    acq_fraction = float(param_grid.get("inner_acquisition_fraction", config.get("adaptive_inner_acquisition_fraction", 0.15)))
    test_fraction = float(param_grid.get("inner_validation_fraction", config.get("adaptive_inner_validation_fraction", 0.15)))
    entropy_correction_weight = float(param_grid.get("entropy_correction_weight", config.get("paper_entropy_correction_weight", 1.0)))
    entropy_shortlist_size = int(param_grid.get("entropy_shortlist_size", config.get("paper_entropy_shortlist_size", 20_000)))
    entropy_batch_size = int(param_grid.get("entropy_batch_size", config.get("paper_entropy_batch_size", 4096)))
    entropy_mode = str(param_grid.get("entropy_mode", config.get("entropy_mode", "entropy_observational_lite"))).lower()
    entropy_score_transform = str(param_grid.get("entropy_score_transform", config.get("entropy_score_transform", "none")))

    if len(observed_data) < max(60, batch_size):
        return {
            "lambda_entropy": lambda_grid[0],
            "sigma": sigma_grid[0],
            "diffusion_prob": p_grid[0],
            "validation_loss": np.nan,
            "evaluated_configs": 0,
        }

    rng = np.random.default_rng(seed)
    num_users, num_items = _infer_shape(observed_data, candidate_data, np.asarray(user_graph), config)
    inner_train, inner_pool, inner_val = _split_observed(
        observed_data,
        batch_size=batch_size,
        seed=seed,
        acquisition_fraction=acq_fraction,
        test_fraction=test_fraction,
    )
    if inner_pool.empty or inner_val.empty:
        return {
            "lambda_entropy": lambda_grid[0],
            "sigma": sigma_grid[0],
            "diffusion_prob": p_grid[0],
            "validation_loss": np.nan,
            "evaluated_configs": 0,
        }

    graph_array = np.asarray(user_graph, dtype=float)
    if graph_array.ndim == 2 and graph_array.shape[0] == graph_array.shape[1]:
        graph_embedding = graph_array
    else:
        graph_embedding = graph_array
    neighbor_idx, neighbor_dist_sq = compute_dls_neighbors(graph_embedding, top_k=top_k)

    inner_model = MatrixFactorization(num_users, num_items, mf_config, seed=seed).fit(inner_train)
    base_user_factors = inner_model.user_representations()
    best = {
        "lambda_entropy": lambda_grid[0],
        "sigma": sigma_grid[0],
        "diffusion_prob": p_grid[0],
        "validation_loss": float("inf"),
        "evaluated_configs": 0,
    }

    for sigma in sigma_grid:
        user_dls, _ = scoped_gaussian_dls_user_factors(
            base_user_factors,
            graph_embedding,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=sigma,
            scope=dls_scope,
        )
        entropy_regularization = float(param_grid.get("entropy_regularization", config.get("entropy_logdet_regularization", 1.0)))
        if entropy_mode in {
            "schur_conditional_variance",
            "entropy_schur_conditional_variance",
            "paper_schur_conditional_variance",
            "schur_cv",
        }:
            entropy = schur_conditional_variance_entropy_scores(
                inner_pool,
                inner_train,
                user_dls,
                inner_model.item_factors,
                lambda_noise=entropy_regularization,
                shortlist_size=entropy_shortlist_size,
                batch_size=entropy_batch_size,
            )
        else:
            entropy = entropy_observational_lite_scores(
                inner_pool,
                inner_train,
                user_dls,
                inner_model.item_factors,
                regularization=entropy_regularization,
                correction_weight=entropy_correction_weight,
                shortlist_size=entropy_shortlist_size,
                batch_size=entropy_batch_size,
            )
        reward = predicted_reward_scores(inner_model, inner_pool, user_factors=user_dls)
        for lambda_entropy in lambda_grid:
            for diffusion_prob in p_grid:
                if diffusion_prob != 0.0:
                    continue
                scores = zscore(reward) + float(lambda_entropy) * zscore(
                    _transform_entropy_score(entropy, entropy_score_transform)
                )
                selected_pos = _select_inner_batch(
                    scores,
                    batch_size,
                    rng,
                    temperature,
                    selection_mode,
                    selection_top_m,
                    selection_top_fraction,
                )
                selected = inner_pool.iloc[selected_pos].reset_index(drop=True)
                trial_train = pd.concat([inner_train, selected], ignore_index=True)
                eval_idx = int(best["evaluated_configs"]) + 1
                trial_model = MatrixFactorization(
                    num_users,
                    num_items,
                    mf_config,
                    seed=seed * 1009 + eval_idx,
                ).fit(trial_train)
                trial_dls, _ = scoped_gaussian_dls_user_factors(
                    trial_model.user_representations(),
                    graph_embedding,
                    neighbor_idx,
                    neighbor_dist_sq,
                    sigma=sigma,
                    scope=dls_scope,
                )
                loss = _metric_loss(metric_name, trial_model, inner_val, trial_dls, midpoint=midpoint)
                best["evaluated_configs"] = eval_idx
                if np.isfinite(loss) and loss < float(best["validation_loss"]):
                    best.update(
                        {
                            "lambda_entropy": float(lambda_entropy),
                            "sigma": float(sigma),
                            "diffusion_prob": float(diffusion_prob),
                            "validation_loss": float(loss),
                        }
                    )

    return best
