from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data_loader import RatingSplit
from .dls import compute_dls_neighbors, smooth_dls_user_factors
from .dls import shrink_dls_user_factors
from .dls_paper import scoped_gaussian_dls_user_factors
from .entropy import (
    coherence_lite_entropy_scores,
    coverage_entropy_scores,
    hybrid_entropy_scores,
    mixed_observational_entropy_scores,
    network_observational_expected_entropy_scores,
    network_observational_reliability_stochastic_entropy_scores,
    network_observational_stochastic_entropy_scores,
    network_observational_weighted_entropy_scores,
    network_neighbor_prediction_gain_entropy_scores,
    network_entropy_scores,
    network_observational_entropy_scores,
    observational_logdet_entropy_scores,
    observational_entropy_scores,
    subspace_entropy_scores,
)
from .entropy_paper import entropy_coherence_base_scores, entropy_observational_lite_scores
from .entropy_schur import schur_conditional_variance_entropy_scores
from .graph_embedding import user_embedding
from .matrix_factorization import MFConfig, MatrixFactorization
from .metrics import (
    cumulative_reward as paper_cumulative_reward,
    linear_reward_norm_weighted_rmse,
    pdf_literal_norm_rmse,
    positive_reward_norm_weighted_rmse,
    reward_weighted_rmse,
    reward_tail_proxy_rmse,
    rmse,
    sigmoid_reward_weighted_rmse,
    standard_mask_rmse,
    weighted_rmse,
)
from .reward import network_reward_scores, predicted_reward_scores, true_centered_reward
from .tuning import tune_dunet_params_paper_style
from .utils import zscore


PAPER_FAITHFUL_POLICIES = {
    "paper_faithful_dunet",
    "paper_faithful_dunet_no_entropy",
    "paper_faithful_dunet_no_reward",
}


@dataclass(frozen=True)
class AcquisitionConfig:
    cycles: int = 20
    batch_size: int = 100
    lambda_entropy: float = 1.0
    diffusion_prob: float = 0.03
    sigma: float = 0.4
    dls_top_k: int = 30
    dls_neighbor_mode: str = "embedding_topk"
    selection: str = "topk"
    response_alpha: float = 0.0
    rating_midpoint: float = 3.0
    sigma_grid: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8, 1.6)
    entropy_mode: str = "coverage"
    entropy_coverage_weight: float = 0.5
    ucb_alpha: float = 0.8
    ucb_regularization: float = 1.0
    oful_delta: float = 0.05
    oful_noise_scale: float = 1.0
    oful_param_bound: float = 1.0
    oful_beta_scale: float = 1.0
    diffusion_normalization: str = "none"
    graph_embedding_method: str = "spectral"
    thompson_noise: float = 0.08
    logistic_bandit_temperature: float = 1.0
    finite_bandit_epsilon: float = 0.1
    finite_bandit_explore_fraction: float = 0.25
    adaptive_greedy_min_epsilon: float = 0.02
    selection_temperature: float = 1.0
    entropy_logdet_regularization: float = 1.0
    entropy_logdet_weighting: str = "none"
    entropy_logdet_weight_strength: float = 1.0
    entropy_response_weighting: str = "none"
    entropy_response_weight_strength: float = 1.0
    dls_shrinkage: float = 0.0
    dls_shrinkage_mode: str = "adaptive"
    dls_shrinkage_schedule: tuple[dict[str, object], ...] = ()
    batch_composition_reward_fraction: float = 0.7
    batch_composition_entropy_fraction: float = 0.3
    batch_composition_diversity_fraction: float = 0.0
    batch_composition_pool_multiplier: float = 0.0
    batch_composition_network_penalty: float = 0.25
    batch_composition_network_scan_multiplier: float = 50.0
    network_entropy_max_neighbors: int = 8
    network_entropy_stochastic_samples: int = 8
    network_entropy_reliability_strength: float = 1.0
    network_neighbor_gain_weight: float = 0.25
    score_interaction_weight: float = 0.0
    entropy_reward_frontier_quantile: float = 0.0
    entropy_reward_frontier_floor: float = 0.0
    entropy_reward_frontier_softness: float = 0.0
    late_cycle_guard_start: int = 0
    late_cycle_lambda_multiplier: float = 1.0
    late_cycle_frontier_quantile: float = 0.0
    late_cycle_frontier_floor: float = 0.0
    entropy_score_transform: str = "none"
    residual_bandit_weight: float = 1.0
    adaptive_lambda_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    adaptive_diffusion_prob_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.03)
    adaptive_dls_shrinkage_grid: tuple[float, ...] = ()
    adaptive_inner_validation_fraction: float = 0.15
    adaptive_inner_acquisition_fraction: float = 0.15
    adaptive_max_refits: int = 8
    adaptive_min_validation_improvement: float = 0.0
    adaptive_min_inner_reward_delta: float = -1.0e18
    adaptive_validation_metric: str = "weighted_rmse"
    adaptive_prefilter_metric: str = "inner_reward"
    adaptive_stage_grids: tuple[dict[str, object], ...] = ()
    warm_start_refit: bool = False
    selection_top_m: int = 0
    selection_top_fraction: float = 0.0
    selection_max_per_user: int = 0
    selection_max_per_item: int = 0
    coherence_observed_penalty: float = 0.5
    entropy_coherence_blend: float = 0.0
    paper_tuning_frequency: int = 5
    paper_tuning_metric: str = "pdf_literal_norm_rmse"
    paper_dls_scope: str = "neighbors"
    paper_entropy_correction_weight: float = 1.0
    paper_entropy_shortlist_size: int = 20_000
    paper_entropy_batch_size: int = 4096


def _select_rows(
    scores: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    mode: str,
    temperature: float = 1.0,
    top_m: int = 0,
    top_fraction: float = 0.0,
) -> np.ndarray:
    size = min(batch_size, len(scores))
    if size <= 0:
        return np.array([], dtype=int)
    if mode == "random":
        return rng.choice(len(scores), size=size, replace=False)
    if mode in {"topm_exp", "top_m_exp", "exp_topm", "exp_top_m"}:
        fraction_size = int(np.ceil(len(scores) * float(np.clip(top_fraction, 0.0, 1.0))))
        requested = max(int(top_m), fraction_size, size)
        pool_size = min(len(scores), requested)
        pool_idx = np.argsort(scores)[-pool_size:]
        tau = max(float(temperature), 1e-8)
        pool_scores = scores[pool_idx]
        shifted = (pool_scores - float(pool_scores.max())) / tau
        probs = np.exp(np.clip(shifted, -40, 40))
        probs = probs / probs.sum()
        chosen = rng.choice(len(pool_idx), size=size, replace=False, p=probs)
        return pool_idx[chosen]
    if mode == "exp":
        tau = max(float(temperature), 1e-8)
        shifted = (scores - float(scores.max())) / tau
        probs = np.exp(np.clip(shifted, -40, 40))
        probs = probs / probs.sum()
        return rng.choice(len(scores), size=size, replace=False, p=probs)
    if mode == "topk":
        return np.argsort(scores)[-size:]
    raise ValueError(f"Unknown selection mode: {mode}")


def _select_rows_with_diversity_cap(
    scores: np.ndarray,
    candidates: pd.DataFrame,
    batch_size: int,
    max_per_user: int,
    max_per_item: int,
) -> np.ndarray:
    size = min(batch_size, len(scores))
    if size <= 0:
        return np.array([], dtype=int)
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    user_cap = max(int(max_per_user), 0)
    item_cap = max(int(max_per_item), 0)
    user_count: dict[int, int] = {}
    item_count: dict[int, int] = {}
    selected: list[int] = []
    selected_mask = np.zeros(len(scores), dtype=bool)
    for pos in np.argsort(scores)[::-1]:
        row = int(pos)
        user = int(users[row])
        item = int(items[row])
        if user_cap > 0 and user_count.get(user, 0) >= user_cap:
            continue
        if item_cap > 0 and item_count.get(item, 0) >= item_cap:
            continue
        selected.append(row)
        selected_mask[row] = True
        user_count[user] = user_count.get(user, 0) + 1
        item_count[item] = item_count.get(item, 0) + 1
        if len(selected) >= size:
            return np.array(selected, dtype=int)
    _take_top_unique(scores, size, selected, selected_mask)
    return np.array(selected[:size], dtype=int)


def _take_top_unique(
    values: np.ndarray,
    size: int,
    selected: list[int],
    selected_mask: np.ndarray,
) -> None:
    if size <= 0:
        return
    for idx in np.argsort(values)[::-1]:
        pos = int(idx)
        if not selected_mask[pos]:
            selected.append(pos)
            selected_mask[pos] = True
            if len(selected) >= size:
                return


def _composition_counts(batch_size: int, reward_fraction: float, entropy_fraction: float, diversity_fraction: float) -> tuple[int, int, int]:
    fractions = np.array([reward_fraction, entropy_fraction, diversity_fraction], dtype=float)
    fractions = np.maximum(fractions, 0.0)
    if float(fractions.sum()) <= 0:
        fractions = np.array([0.7, 0.3, 0.0], dtype=float)
    fractions = fractions / float(fractions.sum())
    raw = fractions * batch_size
    counts = np.floor(raw).astype(int)
    remainder = int(batch_size - counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _diversity_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    return 1.0 / np.sqrt(user_counts[users] + 1.0) + 1.0 / np.sqrt(item_counts[items] + 1.0)


def _select_batch_composition(
    scores: dict[str, np.ndarray],
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    batch_size: int,
    config: AcquisitionConfig,
) -> np.ndarray:
    size = min(batch_size, len(candidates))
    if size <= 0:
        return np.array([], dtype=int)
    reward = scores.get("reward_score", scores["score"])
    entropy = scores.get("entropy_score", scores["score"])
    diversity = scores.get("diversity_score")
    if diversity is None:
        diversity = _diversity_scores(candidates, observed, num_users, num_items)
    if config.batch_composition_pool_multiplier > 0:
        pool_size = min(len(candidates), max(size, int(round(size * config.batch_composition_pool_multiplier))))
        pool_idx = np.argsort(reward)[-pool_size:]
        pool_mask = np.zeros(len(candidates), dtype=bool)
        pool_mask[pool_idx] = True
        entropy = np.where(pool_mask, entropy, -np.inf)
        diversity = np.where(pool_mask, diversity, -np.inf)
    reward_n, entropy_n, diversity_n = _composition_counts(
        size,
        config.batch_composition_reward_fraction,
        config.batch_composition_entropy_fraction,
        config.batch_composition_diversity_fraction,
    )
    selected: list[int] = []
    selected_mask = np.zeros(len(candidates), dtype=bool)
    _take_top_unique(reward, reward_n, selected, selected_mask)
    _take_top_unique(entropy, min(size, reward_n + entropy_n), selected, selected_mask)
    _take_top_unique(diversity, min(size, reward_n + entropy_n + diversity_n), selected, selected_mask)
    _take_top_unique(scores["score"], size, selected, selected_mask)
    return np.array(selected[:size], dtype=int)


def _network_redundancy(
    row: int,
    selected_mask: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    adjacency: np.ndarray,
) -> float:
    selected_rows = np.flatnonzero(selected_mask)
    if len(selected_rows) == 0:
        return 0.0
    same_user = float(np.sum(users[selected_rows] == users[row]))
    same_item_rows = selected_rows[items[selected_rows] == items[row]]
    same_item = float(len(same_item_rows))
    if len(same_item_rows) == 0:
        social_same_item = 0.0
    else:
        social_same_item = float(np.sum(adjacency[users[row], users[same_item_rows]]))
    return social_same_item + 0.25 * same_item + 0.25 * same_user


def _take_network_diverse_unique(
    values: np.ndarray,
    target_size: int,
    selected: list[int],
    selected_mask: np.ndarray,
    candidates: pd.DataFrame,
    adjacency: np.ndarray,
    config: AcquisitionConfig,
) -> None:
    if target_size <= len(selected):
        return
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    scan = max(target_size * 5, int(round(target_size * config.batch_composition_network_scan_multiplier)))
    scan = min(len(values), max(scan, target_size))
    order = np.argsort(values)[::-1][:scan]
    penalty_strength = max(float(config.batch_composition_network_penalty), 0.0)
    while len(selected) < target_size:
        best_pos = -1
        best_value = -np.inf
        for pos in order:
            row = int(pos)
            if selected_mask[row]:
                continue
            redundancy = _network_redundancy(row, selected_mask, users, items, adjacency)
            adjusted = float(values[row]) - penalty_strength * redundancy
            if adjusted > best_value:
                best_value = adjusted
                best_pos = row
        if best_pos < 0:
            _take_top_unique(values, target_size, selected, selected_mask)
            return
        selected.append(best_pos)
        selected_mask[best_pos] = True


def _select_network_batch_composition(
    scores: dict[str, np.ndarray],
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    adjacency: np.ndarray,
    num_users: int,
    num_items: int,
    batch_size: int,
    config: AcquisitionConfig,
) -> np.ndarray:
    size = min(batch_size, len(candidates))
    if size <= 0:
        return np.array([], dtype=int)
    reward = scores.get("reward_score", scores["score"])
    entropy = scores.get("entropy_score", scores["score"])
    diversity = scores.get("diversity_score")
    if diversity is None:
        diversity = _diversity_scores(candidates, observed, num_users, num_items)
    if config.batch_composition_pool_multiplier > 0:
        pool_size = min(len(candidates), max(size, int(round(size * config.batch_composition_pool_multiplier))))
        pool_idx = np.argsort(reward)[-pool_size:]
        pool_mask = np.zeros(len(candidates), dtype=bool)
        pool_mask[pool_idx] = True
        entropy = np.where(pool_mask, entropy, -np.inf)
        diversity = np.where(pool_mask, diversity, -np.inf)
    reward_n, entropy_n, diversity_n = _composition_counts(
        size,
        config.batch_composition_reward_fraction,
        config.batch_composition_entropy_fraction,
        config.batch_composition_diversity_fraction,
    )
    selected: list[int] = []
    selected_mask = np.zeros(len(candidates), dtype=bool)
    _take_top_unique(reward, reward_n, selected, selected_mask)
    _take_network_diverse_unique(
        entropy,
        min(size, reward_n + entropy_n),
        selected,
        selected_mask,
        candidates,
        adjacency,
        config,
    )
    _take_network_diverse_unique(
        diversity,
        min(size, reward_n + entropy_n + diversity_n),
        selected,
        selected_mask,
        candidates,
        adjacency,
        config,
    )
    _take_top_unique(scores["score"], size, selected, selected_mask)
    return np.array(selected[:size], dtype=int)


def _reward_frontier_guarded_entropy_z(
    reward_score: np.ndarray,
    entropy_z: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    quantile = float(np.clip(config.entropy_reward_frontier_quantile, 0.0, 1.0))
    if quantile <= 0.0 or len(reward_score) == 0:
        return entropy_z

    threshold = float(np.quantile(reward_score, quantile))
    floor = float(np.clip(config.entropy_reward_frontier_floor, 0.0, 1.0))
    softness = max(float(config.entropy_reward_frontier_softness), 0.0)
    if softness > 0:
        spread = float(np.std(reward_score))
        if spread <= 1e-12:
            gate = np.ones_like(reward_score, dtype=float)
        else:
            scaled = np.clip((reward_score - threshold) / max(spread * softness, 1e-12), -40.0, 40.0)
            gate = 1.0 / (1.0 + np.exp(-scaled))
    else:
        gate = (reward_score >= threshold).astype(float)
    gate = floor + (1.0 - floor) * gate
    entropy_bonus = np.maximum(entropy_z, 0.0) * gate
    entropy_penalty = np.minimum(entropy_z, 0.0)
    return entropy_penalty + entropy_bonus


def _du_net_combined_score(reward_score: np.ndarray, entropy_score: np.ndarray, config: AcquisitionConfig) -> np.ndarray:
    reward_z = zscore(reward_score)
    entropy_z = _reward_frontier_guarded_entropy_z(
        reward_score,
        zscore(_transform_entropy_score(entropy_score, config.entropy_score_transform)),
        config,
    )
    score = reward_z + config.lambda_entropy * entropy_z
    interaction_weight = max(float(config.score_interaction_weight), 0.0)
    if interaction_weight > 0:
        joint_high_signal = np.maximum(reward_z, 0.0) * np.maximum(entropy_z, 0.0)
        score = score + interaction_weight * joint_high_signal
    return score


def _scheduled_acquisition_config(config: AcquisitionConfig, cycle: int) -> AcquisitionConfig:
    updated = config
    start = int(config.late_cycle_guard_start)
    if start > 0 and cycle >= start:
        lambda_multiplier = max(float(config.late_cycle_lambda_multiplier), 0.0)
        frontier_quantile = max(
            float(config.entropy_reward_frontier_quantile),
            float(config.late_cycle_frontier_quantile),
        )
        frontier_floor = max(float(config.entropy_reward_frontier_floor), float(config.late_cycle_frontier_floor))
        updated = replace(
            updated,
            lambda_entropy=float(config.lambda_entropy) * lambda_multiplier,
            entropy_reward_frontier_quantile=frontier_quantile,
            entropy_reward_frontier_floor=frontier_floor,
        )
    acquisition_cycle = int(cycle) + 1
    for stage in config.dls_shrinkage_schedule or ():
        stage_start = int(stage.get("start_cycle", stage.get("start", 1)))
        stage_end = int(stage.get("end_cycle", stage.get("end", config.cycles)))
        if stage_start <= acquisition_cycle <= stage_end:
            kwargs: dict[str, object] = {}
            if "dls_shrinkage" in stage:
                kwargs["dls_shrinkage"] = float(stage["dls_shrinkage"])
            if "alpha" in stage:
                kwargs["dls_shrinkage"] = float(stage["alpha"])
            if "dls_shrinkage_mode" in stage:
                kwargs["dls_shrinkage_mode"] = str(stage["dls_shrinkage_mode"])
            if "mode" in stage:
                kwargs["dls_shrinkage_mode"] = str(stage["mode"])
            if kwargs:
                updated = replace(updated, **kwargs)
            break
    return updated


def _apply_dls_shrinkage(
    raw_user_factors: np.ndarray,
    smoothed_user_factors: np.ndarray,
    effective_neighborhood: np.ndarray,
    observed: pd.DataFrame,
    user_degree: np.ndarray,
    config: AcquisitionConfig,
    num_users: int,
) -> tuple[np.ndarray, np.ndarray]:
    if float(config.dls_shrinkage) <= 0:
        return smoothed_user_factors, np.zeros(raw_user_factors.shape[0], dtype=float)
    user_observed_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    return shrink_dls_user_factors(
        raw_user_factors,
        smoothed_user_factors,
        effective_neighborhood,
        user_degree,
        config.dls_top_k,
        strength=config.dls_shrinkage,
        user_observed_counts=user_observed_counts,
        mode=config.dls_shrinkage_mode,
    )


def _transform_entropy_score(entropy_score: np.ndarray, transform: str) -> np.ndarray:
    mode = transform.lower()
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


def _dls_neighbors_for_config(
    network_embedding: np.ndarray,
    adjacency: np.ndarray,
    config: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mode = config.dls_neighbor_mode.lower()
    if mode in {"", "embedding_topk", "topk"}:
        return compute_dls_neighbors(network_embedding, top_k=config.dls_top_k)
    if mode not in {"graph_edges", "adjacency", "graph"}:
        raise ValueError(f"Unknown dls_neighbor_mode: {config.dls_neighbor_mode}")

    n = network_embedding.shape[0]
    k = min(max(1, config.dls_top_k), max(n, 1))
    diff = network_embedding[:, None, :] - network_embedding[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    all_idx, all_dist = compute_dls_neighbors(network_embedding, top_k=k)
    neighbor_idx = np.empty((n, k), dtype=int)
    neighbor_dist_sq = np.empty((n, k), dtype=float)
    for user in range(n):
        graph_neighbors = np.flatnonzero(adjacency[user])
        if len(graph_neighbors):
            order = np.argsort(dist_sq[user, graph_neighbors])
            chosen = graph_neighbors[order[:k]]
            if len(chosen) < k:
                fill = [idx for idx in all_idx[user] if idx not in set(chosen)]
                chosen = np.concatenate([chosen, np.asarray(fill[: k - len(chosen)], dtype=int)])
        else:
            chosen = all_idx[user]
        neighbor_idx[user] = chosen[:k]
        neighbor_dist_sq[user] = dist_sq[user, neighbor_idx[user]]
    return neighbor_idx, neighbor_dist_sq


def _adaptive_grids_for_cycle(
    config: AcquisitionConfig,
    cycle: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    lambda_grid = tuple(float(value) for value in config.adaptive_lambda_grid)
    sigma_grid = tuple(float(value) for value in config.sigma_grid)
    diffusion_grid = tuple(float(value) for value in config.adaptive_diffusion_prob_grid)
    dls_shrinkage_grid = tuple(float(value) for value in config.adaptive_dls_shrinkage_grid)
    if not dls_shrinkage_grid:
        dls_shrinkage_grid = (float(config.dls_shrinkage),)
    acquisition_cycle = int(cycle) + 1
    for stage in config.adaptive_stage_grids or ():
        start = int(stage.get("start_cycle", stage.get("start", 1)))
        end = int(stage.get("end_cycle", stage.get("end", config.cycles)))
        if start <= acquisition_cycle <= end:
            if "lambda_grid" in stage:
                lambda_grid = tuple(float(value) for value in stage["lambda_grid"])
            if "sigma_grid" in stage:
                sigma_grid = tuple(float(value) for value in stage["sigma_grid"])
            if "diffusion_prob_grid" in stage:
                diffusion_grid = tuple(float(value) for value in stage["diffusion_prob_grid"])
            if "dls_shrinkage_grid" in stage:
                dls_shrinkage_grid = tuple(float(value) for value in stage["dls_shrinkage_grid"])
            if "alpha_grid" in stage:
                dls_shrinkage_grid = tuple(float(value) for value in stage["alpha_grid"])
            break
    return lambda_grid, sigma_grid, diffusion_grid, dls_shrinkage_grid


def _policy_scores(
    policy: str,
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    adjacency: np.ndarray,
    num_users: int,
    num_items: int,
    user_factors_dls: np.ndarray,
    config: AcquisitionConfig,
) -> dict[str, np.ndarray]:
    direct_reward = predicted_reward_scores(model, candidates)
    if config.entropy_mode in {"entropy_coherence_base", "paper_coherence_base"}:
        base_entropy = entropy_coherence_base_scores(candidates, user_factors_dls, model.item_factors)
    elif config.entropy_mode in {
        "schur_conditional_variance",
        "entropy_schur_conditional_variance",
        "paper_schur_conditional_variance",
        "schur_cv",
    }:
        base_entropy = schur_conditional_variance_entropy_scores(
            candidates,
            observed,
            user_factors_dls,
            model.item_factors,
            lambda_noise=config.entropy_logdet_regularization,
            shortlist_size=config.paper_entropy_shortlist_size,
            batch_size=config.paper_entropy_batch_size,
        )
    elif config.entropy_mode in {"entropy_observational_lite", "paper_observational_lite"}:
        base_entropy = entropy_observational_lite_scores(
            candidates,
            observed,
            user_factors_dls,
            model.item_factors,
            regularization=config.entropy_logdet_regularization,
            correction_weight=config.paper_entropy_correction_weight,
            shortlist_size=config.paper_entropy_shortlist_size,
            batch_size=config.paper_entropy_batch_size,
        )
    elif config.entropy_mode == "subspace":
        base_entropy = subspace_entropy_scores(candidates, user_factors_dls, model.item_factors)
    elif config.entropy_mode == "coherence_lite":
        base_entropy = coherence_lite_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            observed_penalty_weight=config.coherence_observed_penalty,
        )
    elif config.entropy_mode == "observational":
        base_entropy = observational_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
        )
    elif config.entropy_mode == "observational_logdet":
        base_entropy = observational_logdet_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            regularization=config.entropy_logdet_regularization,
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "network_observational":
        base_entropy = network_observational_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "network_observational_expected":
        base_entropy = network_observational_expected_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "network_observational_stochastic":
        base_entropy = network_observational_stochastic_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            samples=config.network_entropy_stochastic_samples,
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "network_observational_stochastic_reliability":
        base_entropy = network_observational_reliability_stochastic_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            samples=config.network_entropy_stochastic_samples,
            reliability_strength=config.network_entropy_reliability_strength,
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "network_observational_weighted":
        base_entropy = network_observational_weighted_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            reliability_strength=config.network_entropy_reliability_strength,
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "network_neighbor_prediction_gain":
        base_entropy = network_neighbor_prediction_gain_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            reliability_strength=config.network_entropy_reliability_strength,
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
    elif config.entropy_mode == "hybrid_neighbor_prediction_gain":
        hybrid_entropy = network_entropy_scores(
            hybrid_entropy_scores(
                candidates,
                observed,
                num_users,
                num_items,
                user_factors_dls,
                model.item_factors,
                coverage_weight=config.entropy_coverage_weight,
            ),
            candidates,
            adjacency,
            config.diffusion_prob,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
        )
        neighbor_gain = network_neighbor_prediction_gain_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            adjacency,
            config.diffusion_prob,
            regularization=config.entropy_logdet_regularization,
            max_neighbors=config.network_entropy_max_neighbors,
            normalization="degree" if config.diffusion_normalization != "none" else "none",
            reliability_strength=config.network_entropy_reliability_strength,
            predicted_values=direct_reward,
            rating_midpoint=config.rating_midpoint,
            weighting=config.entropy_logdet_weighting,
            weight_strength=config.entropy_logdet_weight_strength,
        )
        base_entropy = zscore(hybrid_entropy) + max(float(config.network_neighbor_gain_weight), 0.0) * zscore(neighbor_gain)
    elif config.entropy_mode == "observational_hybrid":
        base_entropy = mixed_observational_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            coverage_weight=config.entropy_coverage_weight,
        )
    elif config.entropy_mode == "hybrid":
        base_entropy = hybrid_entropy_scores(
            candidates,
            observed,
            num_users,
            num_items,
            user_factors_dls,
            model.item_factors,
            coverage_weight=config.entropy_coverage_weight,
        )
    else:
        base_entropy = coverage_entropy_scores(candidates, observed, num_users, num_items)
    base_entropy = _apply_coherence_entropy_blend(
        base_entropy,
        candidates,
        observed,
        num_users,
        num_items,
        user_factors_dls,
        model.item_factors,
        config,
    )
    base_entropy = _apply_response_entropy_weighting(base_entropy, direct_reward, config)

    if policy == "random":
        score = np.zeros(len(candidates))
        reward_score = direct_reward
        entropy_score = base_entropy
    elif policy == "myopic":
        score = direct_reward
        reward_score = direct_reward
        entropy_score = base_entropy
    elif policy == "entropy":
        score = base_entropy
        reward_score = direct_reward
        entropy_score = base_entropy
    elif policy in {
        "du_net",
        "du_net_no_dls",
        "du_net_no_entropy",
        "du_net_no_reward",
        "du_net_no_network",
        "du_net_static_sigma",
        "du_net_dynamic_sigma",
        "du_net_adaptive_grid",
        "du_net_adaptive_late_guard",
        "du_net_shrinkage_dls",
        "du_net_logdet_entropy",
        "du_net_logdet_shrinkage_dls",
        "du_net_logdet_shrinkage_exp_tau05",
        "paper_faithful_dunet",
        "paper_faithful_dunet_no_entropy",
        "paper_faithful_dunet_no_reward",
        "graph_social",
    }:
        uses_dls = policy != "du_net_no_dls"
        uses_network = policy not in {"du_net_no_network", *PAPER_FAITHFUL_POLICIES}
        factors = user_factors_dls if uses_dls else None
        if uses_network:
            reward_score = network_reward_scores(
                model,
                candidates,
                adjacency,
                config.diffusion_prob,
                user_factors=factors,
                normalization=config.diffusion_normalization,
            )
            if config.entropy_mode in {
                "network_observational",
                "network_observational_expected",
                "network_observational_stochastic",
                "network_observational_stochastic_reliability",
                "network_observational_weighted",
                "network_neighbor_prediction_gain",
                "hybrid_neighbor_prediction_gain",
            }:
                entropy_score = base_entropy
            else:
                entropy_score = network_entropy_scores(
                    base_entropy,
                    candidates,
                    adjacency,
                    config.diffusion_prob,
                    normalization="degree" if config.diffusion_normalization != "none" else "none",
                )
        else:
            reward_score = predicted_reward_scores(model, candidates, user_factors=factors)
            entropy_score = base_entropy
        if policy == "graph_social":
            score = zscore(reward_score)
        elif policy in {"du_net_no_entropy", "paper_faithful_dunet_no_entropy"}:
            score = zscore(reward_score)
        elif policy in {"du_net_no_reward", "paper_faithful_dunet_no_reward"}:
            score = zscore(entropy_score)
        else:
            score = _du_net_combined_score(reward_score, entropy_score, config)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return {
        "score": score,
        "reward_score": reward_score,
        "entropy_score": entropy_score,
        "direct_reward": direct_reward,
    }


def _apply_response_entropy_weighting(
    entropy: np.ndarray,
    predicted_values: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    mode = config.entropy_response_weighting.lower()
    if mode in {"", "none"}:
        return entropy
    if config.response_alpha <= 0:
        return entropy

    response_prob = 0.8 + config.response_alpha * (predicted_values.astype(float) - config.rating_midpoint)
    response_prob = np.clip(response_prob, 0.0, 1.0)
    strength = max(float(config.entropy_response_weight_strength), 0.0)
    if mode == "expected_response":
        weight = np.power(response_prob, strength)
    elif mode == "centered_expected_response":
        centered = response_prob / max(float(np.mean(response_prob)), 1e-8)
        weight = np.power(np.clip(centered, 0.25, 2.0), strength)
    else:
        raise ValueError(f"Unknown entropy_response_weighting mode: {config.entropy_response_weighting}")
    return entropy * weight


def _apply_coherence_entropy_blend(
    entropy: np.ndarray,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    rho = float(np.clip(config.entropy_coherence_blend, 0.0, 1.0))
    if rho <= 0 or len(entropy) == 0:
        return entropy
    if config.entropy_mode == "coherence_lite":
        return entropy
    coherence = coherence_lite_entropy_scores(
        candidates,
        observed,
        num_users,
        num_items,
        user_factors,
        item_factors,
        observed_penalty_weight=config.coherence_observed_penalty,
    )
    return (1.0 - rho) * zscore(entropy) + rho * zscore(coherence)


def _choose_sigma(
    model: MatrixFactorization,
    observed: pd.DataFrame,
    neighbor_idx: np.ndarray,
    neighbor_dist_sq: np.ndarray,
    sigma_grid: tuple[float, ...],
) -> tuple[float, np.ndarray, np.ndarray]:
    best_sigma = sigma_grid[0]
    base_user_factors = model.user_representations()
    best_factors, best_effective = smooth_dls_user_factors(
        base_user_factors,
        neighbor_idx,
        neighbor_dist_sq,
        sigma=best_sigma,
    )
    if len(observed) < 10:
        return best_sigma, best_factors, best_effective

    users = observed["user_id"].to_numpy(dtype=int)
    items = observed["item_id"].to_numpy(dtype=int)
    truth = observed["rating"].to_numpy(dtype=float)
    best_err = float("inf")
    for sigma in sigma_grid:
        factors, effective = smooth_dls_user_factors(
            base_user_factors,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=sigma,
        )
        pred = model.predict_pairs(users, items, user_factors=factors)
        err = float(np.sqrt(np.mean((truth - pred) ** 2)))
        if err < best_err:
            best_err = err
            best_sigma = sigma
            best_factors = factors
        best_effective = effective
    return best_sigma, best_factors, best_effective


def _choose_adaptive_du_net_config(
    observed: pd.DataFrame,
    adjacency: np.ndarray,
    num_users: int,
    num_items: int,
    mf_config: MFConfig,
    config: AcquisitionConfig,
    neighbor_idx: np.ndarray,
    neighbor_dist_sq: np.ndarray,
    rng: np.random.Generator,
    seed: int,
    cycle: int,
) -> AcquisitionConfig:
    if len(observed) < max(60, config.batch_size):
        return config

    shuffled_pos = rng.permutation(len(observed))
    val_size = min(max(20, int(round(len(observed) * config.adaptive_inner_validation_fraction))), len(observed) // 4)
    remaining_after_val = len(observed) - val_size
    pool_size = min(
        max(config.batch_size, int(round(len(observed) * config.adaptive_inner_acquisition_fraction))),
        max(1, remaining_after_val // 3),
    )
    if val_size <= 0 or pool_size <= 0 or len(observed) - val_size - pool_size < 20:
        return config

    val_idx = shuffled_pos[:val_size]
    pool_idx = shuffled_pos[val_size : val_size + pool_size]
    train_idx = shuffled_pos[val_size + pool_size :]
    inner_train = observed.iloc[train_idx].reset_index(drop=True)
    inner_pool = observed.iloc[pool_idx].reset_index(drop=True)
    inner_val = observed.iloc[val_idx].reset_index(drop=True)

    inner_model = MatrixFactorization(num_users, num_items, mf_config, seed=seed).fit(inner_train)
    base_user_factors = inner_model.user_representations()
    user_degree = adjacency.astype(float).sum(axis=1)
    scored_configs: list[tuple[float, float, float, float, float, pd.DataFrame]] = []

    def validation_loss(
        trial_model: MatrixFactorization,
        validation: pd.DataFrame,
        user_factors: np.ndarray,
    ) -> float:
        mode = config.adaptive_validation_metric.lower()
        if mode in {"weighted_rmse", "local_weighted", "local_reward_weighted"}:
            return weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"linear_reward_norm", "paper_linear_norm", "norm_weighted"}:
            return linear_reward_norm_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"positive_reward_norm", "paper_positive_norm"}:
            return positive_reward_norm_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"sigmoid", "sigmoid_reward", "sigmoid_reward_weighted"}:
            return sigmoid_reward_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"rmse", "conventional_rmse"}:
            return rmse(trial_model, validation, user_factors=user_factors)
        if mode in {
            "linear_reward_norm_plus_rmse",
            "reward_tail_plus_rmse",
            "mixed_reward_rmse",
        }:
            reward_tail_loss = linear_reward_norm_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
            rmse_loss = rmse(trial_model, validation, user_factors=user_factors)
            return 0.5 * reward_tail_loss + 0.5 * rmse_loss
        raise ValueError(f"Unknown adaptive_validation_metric: {config.adaptive_validation_metric}")
    lambda_grid, sigma_grid, diffusion_prob_grid, dls_shrinkage_grid = _adaptive_grids_for_cycle(config, cycle)
    for sigma in sigma_grid:
        smoothed_dls, effective_n = smooth_dls_user_factors(
            base_user_factors,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=float(sigma),
        )
        for dls_shrinkage in dls_shrinkage_grid:
            dls_config = replace(config, sigma=float(sigma), dls_shrinkage=float(dls_shrinkage))
            user_dls, _ = _apply_dls_shrinkage(
                base_user_factors,
                smoothed_dls,
                effective_n,
                inner_train,
                user_degree,
                dls_config,
                num_users,
            )
            for lambda_entropy in lambda_grid:
                for diffusion_prob in diffusion_prob_grid:
                    trial_config = replace(
                        dls_config,
                        lambda_entropy=float(lambda_entropy),
                        diffusion_prob=float(diffusion_prob),
                    )
                    scores = _policy_scores(
                        "du_net",
                        inner_model,
                        inner_pool,
                        inner_train,
                        adjacency,
                        num_users,
                        num_items,
                        user_dls,
                        trial_config,
                    )
                    selected_pos = _select_rows(
                        scores["score"],
                        config.batch_size,
                        rng,
                        config.selection,
                        config.selection_temperature,
                        config.selection_top_m,
                        config.selection_top_fraction,
                    )
                    selected = inner_pool.iloc[selected_pos].copy()
                    cheap_reward = true_centered_reward(selected, midpoint=config.rating_midpoint)
                    selected_score = float(np.mean(scores["score"][selected_pos])) if len(selected_pos) else -np.inf
                    prefilter_mode = config.adaptive_prefilter_metric.lower()
                    if prefilter_mode in {"inner_reward", "reward", "observed_reward"}:
                        prefilter_key = -cheap_reward
                    elif prefilter_mode in {"acquisition_score", "score", "du_net_score"}:
                        prefilter_key = -selected_score
                    elif prefilter_mode in {"reward_score_mixed", "mixed"}:
                        avg_reward = cheap_reward / max(float(len(selected)), 1.0)
                        prefilter_key = -(avg_reward + selected_score)
                    else:
                        raise ValueError(f"Unknown adaptive_prefilter_metric: {config.adaptive_prefilter_metric}")
                    scored_configs.append(
                        (
                            float(prefilter_key),
                            float(lambda_entropy),
                            float(sigma),
                            float(diffusion_prob),
                            float(dls_shrinkage),
                            selected,
                        )
                    )

    if not scored_configs:
        return config

    def evaluate_selected(
        selected: pd.DataFrame,
        lambda_entropy: float,
        sigma: float,
        diffusion_prob: float,
        dls_shrinkage: float,
        eval_idx: int,
    ) -> tuple[float, tuple[float, float, float, float]]:
        trial_observed = pd.concat([inner_train, selected], ignore_index=True)
        trial_model = MatrixFactorization(
            num_users,
            num_items,
            mf_config,
            seed=seed * 1009 + eval_idx,
        ).fit(trial_observed)
        trial_base_user_factors = trial_model.user_representations()
        trial_dls, trial_effective_n = smooth_dls_user_factors(
            trial_base_user_factors,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=sigma,
        )
        trial_config = replace(
            config,
            lambda_entropy=float(lambda_entropy),
            sigma=float(sigma),
            diffusion_prob=float(diffusion_prob),
            dls_shrinkage=float(dls_shrinkage),
        )
        trial_dls, _ = _apply_dls_shrinkage(
            trial_base_user_factors,
            trial_dls,
            trial_effective_n,
            trial_observed,
            user_degree,
            trial_config,
            num_users,
        )
        loss = validation_loss(trial_model, inner_val, trial_dls)
        return loss, (lambda_entropy, sigma, diffusion_prob, dls_shrinkage)

    baseline_selected = next(
        (
            selected
            for _, lambda_entropy, sigma, diffusion_prob, dls_shrinkage, selected in scored_configs
            if np.isclose(lambda_entropy, config.lambda_entropy)
            and np.isclose(sigma, config.sigma)
            and np.isclose(diffusion_prob, config.diffusion_prob)
            and np.isclose(dls_shrinkage, config.dls_shrinkage)
        ),
        None,
    )
    if baseline_selected is None:
        baseline_dls, baseline_effective_n = smooth_dls_user_factors(
            base_user_factors,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=float(config.sigma),
        )
        baseline_dls, _ = _apply_dls_shrinkage(
            base_user_factors,
            baseline_dls,
            baseline_effective_n,
            inner_train,
            user_degree,
            config,
            num_users,
        )
        baseline_scores = _policy_scores(
            "du_net",
            inner_model,
            inner_pool,
            inner_train,
            adjacency,
            num_users,
            num_items,
            baseline_dls,
            config,
        )
        baseline_pos = _select_rows(
            baseline_scores["score"],
            config.batch_size,
            rng,
            config.selection,
            config.selection_temperature,
            config.selection_top_m,
            config.selection_top_fraction,
        )
        baseline_selected = inner_pool.iloc[baseline_pos].copy()
    baseline_reward = true_centered_reward(baseline_selected, midpoint=config.rating_midpoint)
    baseline_loss, baseline_tuple = evaluate_selected(
        baseline_selected,
        float(config.lambda_entropy),
        float(config.sigma),
        float(config.diffusion_prob),
        float(config.dls_shrinkage),
        0,
    )
    max_refits = max(1, int(config.adaptive_max_refits))
    best_loss = baseline_loss
    best_tuple = baseline_tuple
    evaluated = 0
    for _, lambda_entropy, sigma, diffusion_prob, dls_shrinkage, selected in sorted(scored_configs):
        candidate_reward = true_centered_reward(selected, midpoint=config.rating_midpoint)
        if candidate_reward < baseline_reward + float(config.adaptive_min_inner_reward_delta):
            continue
        evaluated += 1
        if evaluated > max_refits:
            break
        idx = evaluated
        loss, trial_tuple = evaluate_selected(selected, lambda_entropy, sigma, diffusion_prob, dls_shrinkage, idx)
        if loss < best_loss:
            best_loss = loss
            best_tuple = trial_tuple

    if best_loss > baseline_loss - float(config.adaptive_min_validation_improvement):
        return config

    return replace(
        config,
        lambda_entropy=float(best_tuple[0]),
        sigma=float(best_tuple[1]),
        diffusion_prob=float(best_tuple[2]),
        dls_shrinkage=float(best_tuple[3]),
    )


def _choose_adaptive_late_guard_config(
    observed: pd.DataFrame,
    adjacency: np.ndarray,
    num_users: int,
    num_items: int,
    mf_config: MFConfig,
    config: AcquisitionConfig,
    neighbor_idx: np.ndarray,
    neighbor_dist_sq: np.ndarray,
    rng: np.random.Generator,
    seed: int,
    cycle: int,
) -> AcquisitionConfig:
    guarded_config = _scheduled_acquisition_config(config, cycle)
    if guarded_config == config or len(observed) < max(60, config.batch_size):
        return guarded_config

    shuffled_pos = rng.permutation(len(observed))
    val_size = min(max(20, int(round(len(observed) * config.adaptive_inner_validation_fraction))), len(observed) // 4)
    remaining_after_val = len(observed) - val_size
    pool_size = min(
        max(config.batch_size, int(round(len(observed) * config.adaptive_inner_acquisition_fraction))),
        max(1, remaining_after_val // 3),
    )
    if val_size <= 0 or pool_size <= 0 or len(observed) - val_size - pool_size < 20:
        return guarded_config

    val_idx = shuffled_pos[:val_size]
    pool_idx = shuffled_pos[val_size : val_size + pool_size]
    train_idx = shuffled_pos[val_size + pool_size :]
    inner_train = observed.iloc[train_idx].reset_index(drop=True)
    inner_pool = observed.iloc[pool_idx].reset_index(drop=True)
    inner_val = observed.iloc[val_idx].reset_index(drop=True)

    inner_model = MatrixFactorization(num_users, num_items, mf_config, seed=seed).fit(inner_train)

    def validation_loss(
        trial_model: MatrixFactorization,
        validation: pd.DataFrame,
        user_factors: np.ndarray,
    ) -> float:
        mode = config.adaptive_validation_metric.lower()
        if mode in {"weighted_rmse", "local_weighted", "local_reward_weighted"}:
            return weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"linear_reward_norm", "paper_linear_norm", "norm_weighted"}:
            return linear_reward_norm_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"positive_reward_norm", "paper_positive_norm"}:
            return positive_reward_norm_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"sigmoid", "sigmoid_reward", "sigmoid_reward_weighted"}:
            return sigmoid_reward_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
        if mode in {"rmse", "conventional_rmse"}:
            return rmse(trial_model, validation, user_factors=user_factors)
        if mode in {
            "linear_reward_norm_plus_rmse",
            "reward_tail_plus_rmse",
            "mixed_reward_rmse",
        }:
            reward_tail_loss = linear_reward_norm_weighted_rmse(
                trial_model,
                validation,
                midpoint=config.rating_midpoint,
                user_factors=user_factors,
            )
            rmse_loss = rmse(trial_model, validation, user_factors=user_factors)
            return 0.5 * reward_tail_loss + 0.5 * rmse_loss
        raise ValueError(f"Unknown adaptive_validation_metric: {config.adaptive_validation_metric}")

    def evaluate_config(trial_config: AcquisitionConfig, eval_idx: int) -> tuple[float, float]:
        inner_dls, _ = smooth_dls_user_factors(
            inner_model.user_representations(),
            neighbor_idx,
            neighbor_dist_sq,
            sigma=float(trial_config.sigma),
        )
        scores = _policy_scores(
            "du_net",
            inner_model,
            inner_pool,
            inner_train,
            adjacency,
            num_users,
            num_items,
            inner_dls,
            trial_config,
        )
        selected_pos = _select_rows(
            scores["score"],
            trial_config.batch_size,
            rng,
            trial_config.selection,
            trial_config.selection_temperature,
            trial_config.selection_top_m,
            trial_config.selection_top_fraction,
        )
        selected = inner_pool.iloc[selected_pos].copy()
        selected_reward = true_centered_reward(selected, midpoint=config.rating_midpoint)
        trial_observed = pd.concat([inner_train, selected], ignore_index=True)
        trial_model = MatrixFactorization(
            num_users,
            num_items,
            mf_config,
            seed=seed * 1009 + eval_idx,
        ).fit(trial_observed)
        trial_dls, _ = smooth_dls_user_factors(
            trial_model.user_representations(),
            neighbor_idx,
            neighbor_dist_sq,
            sigma=float(trial_config.sigma),
        )
        return validation_loss(trial_model, inner_val, trial_dls), selected_reward

    baseline_loss, baseline_reward = evaluate_config(config, 0)
    guarded_loss, guarded_reward = evaluate_config(guarded_config, 1)
    if guarded_reward < baseline_reward + float(config.adaptive_min_inner_reward_delta):
        return config
    if guarded_loss < baseline_loss - float(config.adaptive_min_validation_improvement):
        return guarded_config
    return config


def _interaction_features(user_factors: np.ndarray, item_factors: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    return user_factors[users] * item_factors[items]


def _concat_features(user_factors: np.ndarray, item_factors: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    return np.hstack([user_factors[users], item_factors[items]])


def _bias_interaction_features(
    model: MatrixFactorization,
    user_factors: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    interaction = user_factors[users] * model.item_factors[items]
    return np.hstack(
        [
            user_factors[users],
            model.item_factors[items],
            interaction,
            model.user_bias[users, None],
            model.item_bias[items, None],
            np.ones((len(frame), 1), dtype=float),
        ]
    )


def _ridge_ucb_feature_scores(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_cand: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    dim = x_obs.shape[1]
    reg = max(config.ucb_regularization, 1e-8)
    gram = x_obs.T @ x_obs + reg * np.eye(dim)
    theta = np.linalg.solve(gram, x_obs.T @ y_obs)
    mean = x_cand @ theta
    solved = np.linalg.solve(gram, x_cand.T).T
    uncertainty = np.sqrt(np.maximum(np.sum(x_cand * solved, axis=1), 0.0))
    return mean + config.ucb_alpha * uncertainty


def _linear_ucb_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    x_obs = _interaction_features(user_factors, model.item_factors, observed)
    y_obs = observed["rating"].to_numpy(dtype=float) - config.rating_midpoint
    rank = x_obs.shape[1]
    reg = max(config.ucb_regularization, 1e-8)
    gram = x_obs.T @ x_obs + reg * np.eye(rank)
    rhs = x_obs.T @ y_obs
    theta = np.linalg.solve(gram, rhs)
    x_cand = _interaction_features(user_factors, model.item_factors, candidates)
    mean = x_cand @ theta
    solved = np.linalg.solve(gram, x_cand.T).T
    uncertainty = np.sqrt(np.maximum(np.sum(x_cand * solved, axis=1), 0.0))
    return mean + config.ucb_alpha * uncertainty


def _oful_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    x_obs = _interaction_features(user_factors, model.item_factors, observed)
    y_obs = observed["rating"].to_numpy(dtype=float) - config.rating_midpoint
    rank = x_obs.shape[1]
    reg = max(config.ucb_regularization, 1e-8)
    gram = x_obs.T @ x_obs + reg * np.eye(rank)
    rhs = x_obs.T @ y_obs
    theta = np.linalg.solve(gram, rhs)

    x_cand = _interaction_features(user_factors, model.item_factors, candidates)
    mean = x_cand @ theta
    solved = np.linalg.solve(gram, x_cand.T).T
    uncertainty = np.sqrt(np.maximum(np.sum(x_cand * solved, axis=1), 0.0))

    beta = _oful_beta(x_obs, x_cand, config) * max(config.oful_beta_scale, 0.0)
    return mean + beta * uncertainty


def _oful_beta(x_obs: np.ndarray, x_cand: np.ndarray, config: AcquisitionConfig) -> float:
    rank = x_obs.shape[1]
    reg = max(config.ucb_regularization, 1e-8)
    observed_norm = float(np.linalg.norm(x_obs, axis=1).max()) if len(x_obs) else 0.0
    candidate_norm = float(np.linalg.norm(x_cand, axis=1).max()) if len(x_cand) else 0.0
    feature_bound = max(observed_norm, candidate_norm, 1.0)
    delta = min(max(config.oful_delta, 1e-12), 1.0 - 1e-12)
    confidence = rank * np.log((1.0 + len(x_obs) * feature_bound**2 / reg) / delta)
    beta = config.oful_noise_scale * np.sqrt(max(confidence, 0.0))
    beta += np.sqrt(reg) * max(config.oful_param_bound, 0.0)
    return beta


def _mf_oful_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    mean = predicted_reward_scores(model, candidates, user_factors=user_factors)
    x_obs = _interaction_features(user_factors, model.item_factors, observed)
    x_cand = _interaction_features(user_factors, model.item_factors, candidates)
    uncertainty = _linear_ucb_uncertainty(candidates, observed, user_factors, model.item_factors, config)
    beta = _oful_beta(x_obs, x_cand, config)
    return mean + config.oful_beta_scale * beta * zscore(uncertainty)


def _linear_ucb_uncertainty(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    x_obs = _interaction_features(user_factors, item_factors, observed)
    rank = x_obs.shape[1]
    reg = max(config.ucb_regularization, 1e-8)
    gram = x_obs.T @ x_obs + reg * np.eye(rank)
    x_cand = _interaction_features(user_factors, item_factors, candidates)
    solved = np.linalg.solve(gram, x_cand.T).T
    return np.sqrt(np.maximum(np.sum(x_cand * solved, axis=1), 0.0))


def _count_uncertainty_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    return 1.0 / np.sqrt(user_counts[users] + 1.0) + 1.0 / np.sqrt(item_counts[items] + 1.0)


def _mf_linear_ucb_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    mean = predicted_reward_scores(model, candidates, user_factors=user_factors)
    uncertainty = _linear_ucb_uncertainty(candidates, observed, user_factors, model.item_factors, config)
    return mean + config.ucb_alpha * zscore(uncertainty)


def _linucb_concat_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    x_obs = _concat_features(user_factors, model.item_factors, observed)
    y_obs = observed["rating"].to_numpy(dtype=float) - config.rating_midpoint
    x_cand = _concat_features(user_factors, model.item_factors, candidates)
    return _ridge_ucb_feature_scores(x_obs, y_obs, x_cand, config)


def _linucb_bias_interaction_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    x_obs = _bias_interaction_features(model, user_factors, observed)
    y_obs = observed["rating"].to_numpy(dtype=float) - config.rating_midpoint
    x_cand = _bias_interaction_features(model, user_factors, candidates)
    return _ridge_ucb_feature_scores(x_obs, y_obs, x_cand, config)


def _residual_linear_bandit_components(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_obs = _interaction_features(user_factors, model.item_factors, observed)
    users = observed["user_id"].to_numpy(dtype=int)
    items = observed["item_id"].to_numpy(dtype=int)
    baseline = model.predict_pairs(users, items, user_factors=user_factors)
    residual = observed["rating"].to_numpy(dtype=float) - baseline
    rank = x_obs.shape[1]
    reg = max(config.ucb_regularization, 1e-8)
    gram = x_obs.T @ x_obs + reg * np.eye(rank)
    theta = np.linalg.solve(gram, x_obs.T @ residual)
    x_cand = _interaction_features(user_factors, model.item_factors, candidates)
    residual_mean = x_cand @ theta
    solved = np.linalg.solve(gram, x_cand.T).T
    uncertainty = np.sqrt(np.maximum(np.sum(x_cand * solved, axis=1), 0.0))
    return residual_mean, uncertainty


def _mf_residual_ucb_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    mean = predicted_reward_scores(model, candidates, user_factors=user_factors)
    residual_mean, uncertainty = _residual_linear_bandit_components(model, candidates, observed, user_factors, config)
    return mean + config.residual_bandit_weight * residual_mean + config.ucb_alpha * zscore(uncertainty)


def _low_rank_bandit_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    mean = model.predict_pairs(users, items, user_factors=user_factors)
    bonus = 1.0 / np.sqrt(user_counts[users] + 1.0) + 1.0 / np.sqrt(item_counts[items] + 1.0)
    return mean + config.ucb_alpha * bonus


def _mf_low_rank_ucb_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    mean = predicted_reward_scores(model, candidates, user_factors=user_factors)
    bonus = _count_uncertainty_scores(candidates, observed, num_users, num_items)
    return mean + config.ucb_alpha * zscore(bonus)


def _thompson_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = _low_rank_bandit_scores(model, candidates, observed, num_users, num_items, user_factors, config)
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    scale = config.thompson_noise * (
        1.0 / np.sqrt(user_counts[users] + 1.0) + 1.0 / np.sqrt(item_counts[items] + 1.0)
    )
    return mean + rng.normal(0.0, scale)


def _mf_thompson_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = predicted_reward_scores(model, candidates, user_factors=user_factors)
    bonus = _count_uncertainty_scores(candidates, observed, num_users, num_items)
    normalized_bonus = bonus / max(float(np.mean(bonus)), 1e-8)
    scale = config.thompson_noise * np.clip(normalized_bonus, 0.25, 4.0)
    return mean + rng.normal(0.0, scale)


def _mf_residual_thompson_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = predicted_reward_scores(model, candidates, user_factors=user_factors)
    residual_mean, uncertainty = _residual_linear_bandit_components(model, candidates, observed, user_factors, config)
    normalized_uncertainty = uncertainty / max(float(np.mean(uncertainty)), 1e-8)
    scale = config.thompson_noise * np.clip(normalized_uncertainty, 0.25, 4.0)
    return mean + config.residual_bandit_weight * residual_mean + rng.normal(0.0, scale)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _logistic_mean_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    predicted = predicted_reward_scores(model, candidates, user_factors=user_factors)
    tau = max(float(config.logistic_bandit_temperature), 1e-8)
    return _sigmoid((predicted - config.rating_midpoint) / tau)


def _logistic_ucb_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    mean = _logistic_mean_scores(model, candidates, user_factors, config)
    uncertainty = _linear_ucb_uncertainty(candidates, observed, user_factors, model.item_factors, config)
    return mean + config.ucb_alpha * zscore(uncertainty)


def _logistic_thompson_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = _logistic_mean_scores(model, candidates, user_factors, config)
    uncertainty = _linear_ucb_uncertainty(candidates, observed, user_factors, model.item_factors, config)
    normalized_uncertainty = uncertainty / max(float(np.mean(uncertainty)), 1e-8)
    scale = config.thompson_noise * np.clip(normalized_uncertainty, 0.25, 4.0)
    return mean + rng.normal(0.0, scale)


def _finite_arm_mean_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    ratings = observed["rating"].to_numpy(dtype=float)
    global_mean = float(np.mean(ratings)) if len(ratings) else 0.0
    obs_users = observed["user_id"].to_numpy(dtype=int)
    obs_items = observed["item_id"].to_numpy(dtype=int)
    user_sum = np.bincount(obs_users, weights=ratings, minlength=num_users)
    item_sum = np.bincount(obs_items, weights=ratings, minlength=num_items)
    user_count = np.bincount(obs_users, minlength=num_users)
    item_count = np.bincount(obs_items, minlength=num_items)
    numerator = global_mean + user_sum[users] + item_sum[items]
    denominator = 1.0 + user_count[users] + item_count[items]
    return numerator / denominator


def _finite_ucb_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    config: AcquisitionConfig,
) -> np.ndarray:
    mean = _finite_arm_mean_scores(candidates, observed, num_users, num_items)
    uncertainty = _count_uncertainty_scores(candidates, observed, num_users, num_items)
    return mean + config.ucb_alpha * zscore(uncertainty)


def _finite_greedy_with_random_exploration_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    config: AcquisitionConfig,
    rng: np.random.Generator,
    epsilon: float,
) -> np.ndarray:
    score = _finite_arm_mean_scores(candidates, observed, num_users, num_items)
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    explore_size = int(round(min(config.batch_size, len(score)) * epsilon))
    if explore_size <= 0:
        return score
    explore_pos = rng.choice(len(score), size=min(explore_size, len(score)), replace=False)
    random_priority = rng.random(len(score))
    score = score.copy()
    score[explore_pos] = float(score.max()) + 1.0 + random_priority[explore_pos]
    return score


def _prediction_uncertainty_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    predicted = predicted_reward_scores(model, candidates, user_factors=user_factors)
    return -np.abs(predicted - config.rating_midpoint)


def _leverage_uncertainty_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    return _linear_ucb_uncertainty(candidates, observed, user_factors, item_factors, config)


def _active_learning_hybrid_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    config: AcquisitionConfig,
) -> np.ndarray:
    margin = _prediction_uncertainty_scores(model, candidates, user_factors, config)
    leverage = _leverage_uncertainty_scores(candidates, observed, user_factors, model.item_factors, config)
    coverage = _count_uncertainty_scores(candidates, observed, num_users, num_items)
    return zscore(margin) + config.ucb_alpha * zscore(leverage) + 0.5 * config.ucb_alpha * zscore(coverage)


def _social_neighbor_item_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    adjacency: np.ndarray,
    user_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct = predicted_reward_scores(model, candidates, user_factors=user_factors)
    social = direct.copy()
    evidence = np.zeros(len(candidates), dtype=float)
    if candidates.empty or observed.empty:
        return social, evidence, direct

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    candidate_frame = pd.DataFrame({"row": np.arange(len(candidates)), "user": users, "item": items})
    observed_by_item = {
        int(item): group
        for item, group in observed.groupby("item_id", sort=False)
    }
    for item, candidate_group in candidate_frame.groupby("item", sort=False):
        observed_group = observed_by_item.get(int(item))
        if observed_group is None or observed_group.empty:
            continue
        rows = candidate_group["row"].to_numpy(dtype=int)
        candidate_users = candidate_group["user"].to_numpy(dtype=int)
        observed_users = observed_group["user_id"].to_numpy(dtype=int)
        observed_ratings = observed_group["rating"].to_numpy(dtype=float)
        weights = adjacency[np.ix_(candidate_users, observed_users)].astype(float)
        counts = weights.sum(axis=1)
        has_signal = counts > 0
        if not has_signal.any():
            continue
        neighbor_sum = weights @ observed_ratings
        social[rows[has_signal]] = neighbor_sum[has_signal] / counts[has_signal]
        evidence[rows] = counts
    return social, evidence, direct


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def _masked_weighted_rmse(
    model: MatrixFactorization,
    test: pd.DataFrame,
    mask: np.ndarray,
    midpoint: float,
    user_factors: np.ndarray | None,
) -> float:
    if not mask.any():
        return np.nan
    return weighted_rmse(model, test.loc[mask], midpoint=midpoint, user_factors=user_factors)


def _masked_rmse(
    model: MatrixFactorization,
    test: pd.DataFrame,
    mask: np.ndarray,
    user_factors: np.ndarray | None,
) -> float:
    if not mask.any():
        return np.nan
    return rmse(model, test.loc[mask], user_factors=user_factors)


def _degree_bin_masks(user_degree: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.quantile(user_degree, [1.0 / 3.0, 2.0 / 3.0])
    selected_degree = user_degree[users]
    low = selected_degree <= lo
    mid = (selected_degree > lo) & (selected_degree <= hi)
    high = selected_degree > hi
    return low, mid, high


def _row_level_prediction_frame(
    model: MatrixFactorization,
    frame: pd.DataFrame,
    policy: str,
    cycle: int,
    selected_or_test: str,
    midpoint: float,
    item_counts: np.ndarray,
    user_degree: np.ndarray,
    user_factors: np.ndarray | None,
) -> pd.DataFrame:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    truth = frame["rating"].to_numpy(dtype=float)
    raw_pred = model.predict_pairs_raw(users, items, user_factors=user_factors)
    clipped_pred = np.clip(raw_pred, 1.0, 5.0)
    err_raw = truth - raw_pred
    err_clipped = truth - clipped_pred
    reward = 1.0 / (1.0 + np.exp(-np.clip(truth - midpoint, -40.0, 40.0)))
    popularity = item_counts[items].astype(float) if len(items) else np.asarray([], dtype=float)
    degrees = user_degree[users].astype(float) if len(users) else np.asarray([], dtype=float)
    mean_popularity = float(np.mean(popularity)) if len(popularity) else 1.0
    mean_reward = float(np.mean(reward)) if len(reward) else 1.0
    source_index = frame.index.to_numpy(dtype=int)
    test_mask_id = source_index if selected_or_test == "test" else np.full(len(frame), -1, dtype=int)
    return pd.DataFrame(
        {
            "method": policy,
            "policy": policy,
            "cycle": int(cycle),
            "test_mask_id": test_mask_id,
            "row_source_index": source_index,
            "user_id": users,
            "item_id": items,
            "y_true": truth,
            "rating": truth,
            "y_pred_clipped": clipped_pred,
            "y_pred_raw": raw_pred,
            "err_raw": err_raw,
            "err_clipped": err_clipped,
            "prediction": clipped_pred,
            "selected_or_test": selected_or_test,
            "reward": reward,
            "item_popularity": popularity,
            "user_degree": degrees,
            "test_weight_candidates": np.ones(len(frame), dtype=float),
            "weight_rating": truth,
            "weight_rating_norm": truth / max(float(np.mean(truth)) if len(truth) else 1.0, 1e-12),
            "weight_rating_minus_midpoint_pos": np.maximum(truth - midpoint, 0.0),
            "weight_rating_minus_midpoint_pos_eps": np.maximum(truth - midpoint, 0.0) + 0.1,
            "weight_sigmoid_reward": reward,
            "weight_sigmoid_reward_norm": reward / max(mean_reward, 1e-12),
            "weight_selected_reward_contribution": reward,
            "weight_item_popularity": popularity,
            "weight_item_popularity_norm": popularity / max(mean_popularity, 1e-12),
            "weight_inverse_item_popularity": 1.0 / np.maximum(popularity, 1.0),
        }
    )


def run_policy(
    policy: str,
    split: RatingSplit,
    adjacency: np.ndarray,
    num_users: int,
    num_items: int,
    mf_config: MFConfig,
    acquisition_config: AcquisitionConfig,
    seed: int,
    network_embedding: np.ndarray | None = None,
    prediction_sink: list[pd.DataFrame] | None = None,
    prediction_cycles: set[int] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = split.initial.copy()
    candidates = split.candidate.copy()
    if network_embedding is None:
        network_embedding = user_embedding(
            adjacency,
            dim=mf_config.rank,
            method=acquisition_config.graph_embedding_method,
            seed=seed,
        )
    dls_neighbor_idx, dls_neighbor_dist_sq = _dls_neighbors_for_config(
        network_embedding,
        adjacency,
        acquisition_config,
    )
    user_degree = adjacency.astype(float).sum(axis=1)
    test_users = split.test["user_id"].to_numpy(dtype=int)
    cumulative_reward = 0.0
    centered_cumulative_reward = 0.0
    attempted_count = 0
    responded_count = 0
    rows: list[dict[str, float | int | str]] = []
    previous_model: MatrixFactorization | None = None
    latest_paper_config = acquisition_config

    for cycle in range(acquisition_config.cycles + 1):
        model = MatrixFactorization(num_users, num_items, mf_config, seed=seed * 1009 + cycle)
        if acquisition_config.warm_start_refit and previous_model is not None:
            model.user_factors = previous_model.user_factors.copy()
            model.item_factors = previous_model.item_factors.copy()
            model.implicit_factors = previous_model.implicit_factors.copy()
            model.user_bias = previous_model.user_bias.copy()
            model.item_bias = previous_model.item_bias.copy()
            model.global_mean = previous_model.global_mean
        model.fit(observed)
        previous_model = model
        cycle_config = acquisition_config
        base_user_factors = model.user_representations()
        avg_dls_shrinkage = np.nan
        if policy in PAPER_FAITHFUL_POLICIES:
            cycle_config = latest_paper_config
            tuning_frequency = int(acquisition_config.paper_tuning_frequency)
            if tuning_frequency > 0 and cycle < acquisition_config.cycles and not candidates.empty and cycle % tuning_frequency == 0:
                tune_result = tune_dunet_params_paper_style(
                    observed,
                    network_embedding,
                    candidates,
                    {
                        "lambda_grid": acquisition_config.adaptive_lambda_grid,
                        "sigma_grid": acquisition_config.sigma_grid,
                        "p_grid": (0.0,),
                        "batch_size": acquisition_config.batch_size,
                        "dls_top_k": acquisition_config.dls_top_k,
                        "paper_dls_scope": acquisition_config.paper_dls_scope,
                        "rating_midpoint": acquisition_config.rating_midpoint,
                        "selection": acquisition_config.selection,
                        "selection_temperature": acquisition_config.selection_temperature,
                        "entropy_regularization": acquisition_config.entropy_logdet_regularization,
                        "entropy_correction_weight": acquisition_config.paper_entropy_correction_weight,
                        "entropy_shortlist_size": acquisition_config.paper_entropy_shortlist_size,
                        "entropy_batch_size": acquisition_config.paper_entropy_batch_size,
                        "entropy_mode": acquisition_config.entropy_mode,
                        "entropy_score_transform": acquisition_config.entropy_score_transform,
                    },
                    {
                        "rank": mf_config.rank,
                        "epochs": mf_config.epochs,
                        "learning_rate": mf_config.learning_rate,
                        "regularization": mf_config.regularization,
                        "algorithm": mf_config.algorithm,
                        "num_users": num_users,
                        "num_items": num_items,
                        "batch_size": acquisition_config.batch_size,
                        "rating_midpoint": acquisition_config.rating_midpoint,
                        "selection": acquisition_config.selection,
                        "selection_temperature": acquisition_config.selection_temperature,
                        "entropy_score_transform": acquisition_config.entropy_score_transform,
                    },
                    acquisition_config.paper_tuning_metric,
                    seed=seed * 1009 + cycle,
                )
                if np.isfinite(float(tune_result["validation_loss"])):
                    cycle_config = replace(
                        latest_paper_config,
                        lambda_entropy=float(tune_result["lambda_entropy"]),
                        sigma=float(tune_result["sigma"]),
                        diffusion_prob=0.0,
                    )
                    latest_paper_config = cycle_config
            sigma_t = cycle_config.sigma
            user_dls, effective_n = scoped_gaussian_dls_user_factors(
                base_user_factors,
                network_embedding,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                sigma=sigma_t,
                scope=cycle_config.paper_dls_scope,
            )
        elif policy == "du_net_dynamic_sigma":
            sigma_t, user_dls, effective_n = _choose_sigma(
                model,
                observed,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                acquisition_config.sigma_grid,
            )
        else:
            sigma_t = acquisition_config.sigma
            user_dls, effective_n = smooth_dls_user_factors(
                base_user_factors,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                sigma=sigma_t,
            )
        if policy == "du_net_adaptive_grid" and cycle < acquisition_config.cycles and not candidates.empty:
            cycle_config = _choose_adaptive_du_net_config(
                observed,
                adjacency,
                num_users,
                num_items,
                mf_config,
                acquisition_config,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                rng,
                seed=seed * 1009 + cycle,
                cycle=cycle,
            )
            sigma_t = cycle_config.sigma
            user_dls, effective_n = smooth_dls_user_factors(
                base_user_factors,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                sigma=sigma_t,
            )
        if policy == "du_net_adaptive_late_guard" and cycle < acquisition_config.cycles and not candidates.empty:
            cycle_config = _choose_adaptive_late_guard_config(
                observed,
                adjacency,
                num_users,
                num_items,
                mf_config,
                acquisition_config,
                dls_neighbor_idx,
                dls_neighbor_dist_sq,
                rng,
                seed=seed * 1009 + cycle,
                cycle=cycle,
            )
        else:
            if policy not in PAPER_FAITHFUL_POLICIES:
                cycle_config = _scheduled_acquisition_config(cycle_config, cycle)
        if cycle_config.dls_shrinkage > 0 and policy not in {"du_net_no_dls", *PAPER_FAITHFUL_POLICIES}:
            user_dls, shrinkage_rho = _apply_dls_shrinkage(
                base_user_factors,
                user_dls,
                effective_n,
                observed,
                user_degree,
                cycle_config,
                num_users,
            )
            avg_dls_shrinkage = float(np.mean(shrinkage_rho))
        eval_dls = policy in {
            "du_net",
            "du_net_no_entropy",
            "du_net_no_reward",
            "du_net_no_network",
            "du_net_static_sigma",
            "du_net_dynamic_sigma",
            "du_net_adaptive_grid",
            "du_net_adaptive_late_guard",
            "paper_faithful_dunet",
            "paper_faithful_dunet_no_entropy",
            "paper_faithful_dunet_no_reward",
        }
        eval_dls = eval_dls or policy in {
            "du_net_shrinkage_dls",
            "du_net_logdet_entropy",
            "du_net_logdet_shrinkage_dls",
            "du_net_logdet_shrinkage_exp_tau05",
        }
        eval_user_factors = user_dls if eval_dls else None
        item_counts_for_predictions = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
        if prediction_sink is not None and (prediction_cycles is None or cycle in prediction_cycles):
            prediction_sink.append(
                _row_level_prediction_frame(
                    model,
                    split.test,
                    policy,
                    cycle,
                    "test",
                    acquisition_config.rating_midpoint,
                    item_counts_for_predictions,
                    user_degree,
                    eval_user_factors,
                )
            )
        user_counts_for_groups = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
        cold_threshold_for_groups = float(np.quantile(user_counts_for_groups, 0.2))
        cold_test_mask = user_counts_for_groups[test_users] <= cold_threshold_for_groups
        warm_test_mask = ~cold_test_mask
        low_degree_test, mid_degree_test, high_degree_test = _degree_bin_masks(user_degree, test_users)
        high_rating_test = split.test["rating"].to_numpy(dtype=float) >= 4.0
        low_rating_test = split.test["rating"].to_numpy(dtype=float) <= 2.0
        rows.append(
            {
                "policy": policy,
                "cycle": cycle,
                "observed_size": len(observed),
                "candidate_size": len(candidates),
                "cumulative_reward": cumulative_reward,
                "centered_cumulative_reward": centered_cumulative_reward,
                "pdf_literal_norm_rmse": pdf_literal_norm_rmse(
                    model,
                    split.test,
                    user_factors=eval_user_factors,
                ),
                "standard_mask_rmse": standard_mask_rmse(
                    model,
                    split.test,
                    user_factors=eval_user_factors,
                ),
                "reward_tail_proxy_rmse": reward_tail_proxy_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    user_factors=eval_user_factors,
                ),
                "rmse": rmse(model, split.test, user_factors=eval_user_factors),
                "w_rmse": weighted_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    user_factors=eval_user_factors,
                ),
                "sigmoid_reward_w_rmse": sigmoid_reward_weighted_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    user_factors=eval_user_factors,
                ),
                "mild_reward_w_rmse": reward_weighted_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    beta=0.5,
                    user_factors=eval_user_factors,
                ),
                "medium_reward_w_rmse": reward_weighted_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    beta=1.0,
                    user_factors=eval_user_factors,
                ),
                "linear_reward_norm_w_rmse": linear_reward_norm_weighted_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    user_factors=eval_user_factors,
                ),
                "positive_reward_norm_w_rmse": positive_reward_norm_weighted_rmse(
                    model,
                    split.test,
                    midpoint=acquisition_config.rating_midpoint,
                    user_factors=eval_user_factors,
                ),
                "cold_user_w_rmse": _masked_weighted_rmse(
                    model,
                    split.test,
                    cold_test_mask,
                    acquisition_config.rating_midpoint,
                    eval_user_factors,
                ),
                "warm_user_w_rmse": _masked_weighted_rmse(
                    model,
                    split.test,
                    warm_test_mask,
                    acquisition_config.rating_midpoint,
                    eval_user_factors,
                ),
                "low_degree_w_rmse": _masked_weighted_rmse(
                    model,
                    split.test,
                    low_degree_test,
                    acquisition_config.rating_midpoint,
                    eval_user_factors,
                ),
                "mid_degree_w_rmse": _masked_weighted_rmse(
                    model,
                    split.test,
                    mid_degree_test,
                    acquisition_config.rating_midpoint,
                    eval_user_factors,
                ),
                "high_degree_w_rmse": _masked_weighted_rmse(
                    model,
                    split.test,
                    high_degree_test,
                    acquisition_config.rating_midpoint,
                    eval_user_factors,
                ),
                "high_rating_rmse": _masked_rmse(
                    model,
                    split.test,
                    high_rating_test,
                    eval_user_factors,
                ),
                "low_rating_rmse": _masked_rmse(
                    model,
                    split.test,
                    low_rating_test,
                    eval_user_factors,
                ),
                "avg_effective_neighborhood": float(np.mean(effective_n)),
                "avg_dls_shrinkage": avg_dls_shrinkage,
                "sigma": float(sigma_t),
                "adaptive_lambda_entropy": float(cycle_config.lambda_entropy),
                "adaptive_diffusion_prob": float(cycle_config.diffusion_prob),
                "response_rate": float(responded_count / attempted_count) if attempted_count else 1.0,
                "selected_avg_reward_score": np.nan,
                "selected_avg_entropy_score": np.nan,
                "selected_avg_reward_percentile": np.nan,
                "selected_avg_entropy_percentile": np.nan,
                "selected_avg_user_degree": np.nan,
                "selected_avg_rating": np.nan,
                "selected_rating_5_share": np.nan,
                "selected_rating_ge4_share": np.nan,
                "selected_unique_users": np.nan,
                "selected_unique_items": np.nan,
                "selected_avg_item_popularity": np.nan,
                "selected_cold_user_share": np.nan,
                "selected_low_degree_share": np.nan,
                "selected_mid_degree_share": np.nan,
                "selected_high_degree_share": np.nan,
            }
        )
        if cycle == acquisition_config.cycles or candidates.empty:
            break

        if policy == "linear_ucb":
            score = _linear_ucb_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "oful":
            score = _oful_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "mf_linear_ucb":
            score = _mf_linear_ucb_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "linucb_mf_interaction":
            score = _mf_linear_ucb_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "linucb_concat":
            score = _linucb_concat_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "linucb_bias_interaction":
            score = _linucb_bias_interaction_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "mf_oful":
            score = _mf_oful_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "mf_residual_ucb":
            score = _mf_residual_ucb_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "low_rank_bandit":
            score = _low_rank_bandit_scores(
                model,
                candidates,
                observed,
                num_users,
                num_items,
                base_user_factors,
                acquisition_config,
            )
            scores = {"score": score}
        elif policy == "mf_low_rank_ucb":
            score = _mf_low_rank_ucb_scores(
                model,
                candidates,
                observed,
                num_users,
                num_items,
                base_user_factors,
                acquisition_config,
            )
            scores = {"score": score}
        elif policy == "thompson_sampling":
            score = _thompson_scores(
                model,
                candidates,
                observed,
                num_users,
                num_items,
                base_user_factors,
                acquisition_config,
                rng,
            )
            scores = {"score": score}
        elif policy == "mf_thompson_sampling":
            score = _mf_thompson_scores(
                model,
                candidates,
                observed,
                num_users,
                num_items,
                base_user_factors,
                acquisition_config,
                rng,
            )
            scores = {"score": score}
        elif policy == "mf_residual_thompson_sampling":
            score = _mf_residual_thompson_scores(
                model,
                candidates,
                observed,
                base_user_factors,
                acquisition_config,
                rng,
            )
            scores = {"score": score}
        elif policy == "logistic_ucb":
            score = _logistic_ucb_scores(model, candidates, observed, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "logistic_thompson_sampling":
            score = _logistic_thompson_scores(
                model,
                candidates,
                observed,
                base_user_factors,
                acquisition_config,
                rng,
            )
            scores = {"score": score}
        elif policy == "finite_ucb":
            score = _finite_ucb_scores(candidates, observed, num_users, num_items, acquisition_config)
            scores = {"score": score}
        elif policy == "epsilon_greedy":
            score = _finite_greedy_with_random_exploration_scores(
                candidates,
                observed,
                num_users,
                num_items,
                acquisition_config,
                rng,
                acquisition_config.finite_bandit_epsilon,
            )
            scores = {"score": score}
        elif policy == "adaptive_greedy":
            observed_batches = max(len(observed) / max(float(acquisition_config.batch_size), 1.0), 1.0)
            epsilon = max(
                float(acquisition_config.adaptive_greedy_min_epsilon),
                float(acquisition_config.finite_bandit_epsilon) / np.sqrt(observed_batches),
            )
            score = _finite_greedy_with_random_exploration_scores(
                candidates,
                observed,
                num_users,
                num_items,
                acquisition_config,
                rng,
                epsilon,
            )
            scores = {"score": score}
        elif policy == "explore_then_exploit":
            fraction = float(np.clip(acquisition_config.finite_bandit_explore_fraction, 0.0, 1.0))
            explore_cycles = int(np.ceil(acquisition_config.cycles * fraction))
            if cycle < explore_cycles:
                score = rng.random(len(candidates))
            else:
                score = _finite_arm_mean_scores(candidates, observed, num_users, num_items)
            scores = {"score": score}
        elif policy == "prediction_uncertainty":
            score = _prediction_uncertainty_scores(model, candidates, base_user_factors, acquisition_config)
            scores = {"score": score}
        elif policy == "leverage_uncertainty":
            score = _leverage_uncertainty_scores(
                candidates,
                observed,
                base_user_factors,
                model.item_factors,
                acquisition_config,
            )
            scores = {"score": score}
        elif policy == "active_learning_hybrid":
            score = _active_learning_hybrid_scores(
                model,
                candidates,
                observed,
                num_users,
                num_items,
                base_user_factors,
                acquisition_config,
            )
            scores = {"score": score}
        elif policy == "social_neighbor_item":
            social, evidence, direct = _social_neighbor_item_scores(
                model,
                candidates,
                observed,
                adjacency,
                base_user_factors,
            )
            scores = {
                "score": social,
                "reward_score": social,
                "entropy_score": evidence,
                "direct_reward": direct,
            }
        elif policy == "social_mf_hybrid":
            social, evidence, direct = _social_neighbor_item_scores(
                model,
                candidates,
                observed,
                adjacency,
                base_user_factors,
            )
            reliability_strength = max(float(acquisition_config.network_entropy_reliability_strength), 1e-8)
            reliability = evidence / (evidence + reliability_strength)
            score = direct + acquisition_config.residual_bandit_weight * reliability * (social - direct)
            scores = {
                "score": score,
                "reward_score": score,
                "entropy_score": evidence,
                "direct_reward": direct,
            }
        else:
            scores = _policy_scores(
                policy,
                model,
                candidates,
                observed,
                adjacency,
                num_users,
                num_items,
                user_dls,
                cycle_config,
            )
        select_mode = "random" if policy == "random" else cycle_config.selection
        if select_mode == "batch_composition":
            selected_pos = _select_batch_composition(
                scores,
                candidates,
                observed,
                num_users,
                num_items,
                cycle_config.batch_size,
                cycle_config,
            )
        elif select_mode == "network_batch_composition":
            selected_pos = _select_network_batch_composition(
                scores,
                candidates,
                observed,
                adjacency,
                num_users,
                num_items,
                cycle_config.batch_size,
                cycle_config,
            )
        elif select_mode in {"diversity_cap", "topk_diversity_cap", "soft_diversity_cap"}:
            selected_pos = _select_rows_with_diversity_cap(
                scores["score"],
                candidates,
                cycle_config.batch_size,
                cycle_config.selection_max_per_user,
                cycle_config.selection_max_per_item,
            )
        else:
            selected_pos = _select_rows(
                scores["score"],
                cycle_config.batch_size,
                rng,
                select_mode,
                cycle_config.selection_temperature,
                cycle_config.selection_top_m,
                cycle_config.selection_top_fraction,
            )
        selected = candidates.iloc[selected_pos].copy()
        if prediction_sink is not None and (prediction_cycles is None or cycle in prediction_cycles) and len(selected):
            prediction_sink.append(
                _row_level_prediction_frame(
                    model,
                    selected,
                    policy,
                    cycle,
                    "selected",
                    acquisition_config.rating_midpoint,
                    item_counts_for_predictions,
                    user_degree,
                    eval_user_factors,
                )
            )
        if len(selected_pos):
            selected_users_all = selected["user_id"].to_numpy(dtype=int)
            selected_items_all = selected["item_id"].to_numpy(dtype=int)
            selected_ratings = selected["rating"].to_numpy(dtype=float)
            item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
            rows[-1]["selected_avg_rating"] = float(np.mean(selected_ratings))
            rows[-1]["selected_rating_5_share"] = float(np.mean(selected_ratings >= 5.0))
            rows[-1]["selected_rating_ge4_share"] = float(np.mean(selected_ratings >= 4.0))
            rows[-1]["selected_unique_users"] = float(len(np.unique(selected_users_all)))
            rows[-1]["selected_unique_items"] = float(len(np.unique(selected_items_all)))
            rows[-1]["selected_avg_item_popularity"] = float(np.mean(item_counts[selected_items_all]))
        if "reward_score" in scores and "entropy_score" in scores and len(selected_pos):
            reward_pct = _rank_percentiles(scores["reward_score"])
            entropy_pct = _rank_percentiles(scores["entropy_score"])
            selected_users = selected["user_id"].to_numpy(dtype=int)
            user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
            cold_threshold = float(np.quantile(user_counts, 0.2))
            rows[-1]["selected_avg_reward_score"] = float(np.mean(scores["reward_score"][selected_pos]))
            rows[-1]["selected_avg_entropy_score"] = float(np.mean(scores["entropy_score"][selected_pos]))
            rows[-1]["selected_avg_reward_percentile"] = float(np.mean(reward_pct[selected_pos]))
            rows[-1]["selected_avg_entropy_percentile"] = float(np.mean(entropy_pct[selected_pos]))
            rows[-1]["selected_avg_user_degree"] = float(np.mean(user_degree[selected_users]))
            rows[-1]["selected_cold_user_share"] = float(np.mean(user_counts[selected_users] <= cold_threshold))
            low_selected, mid_selected, high_selected = _degree_bin_masks(user_degree, selected_users)
            rows[-1]["selected_low_degree_share"] = float(np.mean(low_selected))
            rows[-1]["selected_mid_degree_share"] = float(np.mean(mid_selected))
            rows[-1]["selected_high_degree_share"] = float(np.mean(high_selected))
        observed_selected = selected
        if acquisition_config.response_alpha > 0:
            probs = 0.8 + acquisition_config.response_alpha * (
                selected["rating"].to_numpy(dtype=float) - acquisition_config.rating_midpoint
            )
            probs = np.clip(probs, 0.0, 1.0)
            mask = rng.random(len(selected)) < probs
            observed_selected = selected.loc[mask].copy()
            attempted_count += int(len(mask))
            responded_count += int(mask.sum())
        else:
            attempted_count += int(len(selected))
            responded_count += int(len(selected))
        cumulative_reward += paper_cumulative_reward(observed_selected, midpoint=acquisition_config.rating_midpoint)
        centered_cumulative_reward += true_centered_reward(observed_selected, midpoint=acquisition_config.rating_midpoint)
        observed = pd.concat([observed, observed_selected], ignore_index=True)
        candidates = candidates.drop(candidates.index[selected_pos]).reset_index(drop=True)

    return pd.DataFrame(rows)
