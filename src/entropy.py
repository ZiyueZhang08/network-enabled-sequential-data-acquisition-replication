from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> np.ndarray:
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    return 1.0 / np.sqrt(user_counts[users] + 1.0) + 1.0 / np.sqrt(item_counts[items] + 1.0)


def subspace_entropy_scores(
    candidates: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> np.ndarray:
    """Approximate observational entropy with row-space coherence.

    The paper's entropy term is based on the user/item subspaces after DLS. This
    MVP proxy uses QR leverage scores as a single-entry information-gain signal:
    entries involving high-leverage users/items are more informative for learning
    the low-rank matrix geometry than entries that are already well represented.
    """
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    q_user, _ = np.linalg.qr(user_factors)
    q_item, _ = np.linalg.qr(item_factors)
    user_lev = np.sum(q_user * q_user, axis=1)
    item_lev = np.sum(q_item * q_item, axis=1)
    return user_lev[users] + item_lev[items] + user_lev[users] * item_lev[items]


def coherence_lite_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    observed_penalty_weight: float = 0.5,
) -> np.ndarray:
    """Coherence-style single-entry entropy proxy.

    The paper's observational entropy is written in terms of user/item subspace
    coherence and cross-coherence. This light-weight version keeps the core
    single-entry signal, `mu_i(U_DLS) * mu_j(V)`, and subtracts a redundancy
    penalty for rows/columns that are already well represented in the observed
    matrix.
    """
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    q_user, _ = np.linalg.qr(user_factors)
    q_item, _ = np.linalg.qr(item_factors)
    user_mu = np.sum(q_user * q_user, axis=1)
    item_mu = np.sum(q_item * q_item, axis=1)
    coherence = user_mu[users] * item_mu[items]

    penalty_weight = max(float(observed_penalty_weight), 0.0)
    if penalty_weight <= 0.0 or observed.empty:
        return coherence

    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    user_redundancy = user_counts[users] / max(float(user_counts.max()), 1.0)
    item_redundancy = item_counts[items] / max(float(item_counts.max()), 1.0)
    observed_similarity = np.sqrt(user_redundancy * item_redundancy)
    scale = max(float(np.percentile(coherence, 90)) if len(coherence) else 0.0, 1e-12)
    return coherence - penalty_weight * observed_similarity * scale


def observational_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> np.ndarray:
    """Closer single-entry proxy for the paper's observational entropy.

    It combines leverage in the learned user/item subspaces with a matrix
    completion uncertainty term from sparse user/item observation counts. The
    log form mirrors one-step log-det information gain more closely than the
    older raw leverage proxy while staying cheap enough for full candidate sets.
    """
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    q_user, _ = np.linalg.qr(user_factors)
    q_item, _ = np.linalg.qr(item_factors)
    user_lev = np.sum(q_user * q_user, axis=1)
    item_lev = np.sum(q_item * q_item, axis=1)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    sparse_precision = 1.0 / np.sqrt((user_counts[users] + 1.0) * (item_counts[items] + 1.0))
    leverage_gain = user_lev[users] + item_lev[items] + user_lev[users] * item_lev[items]
    return np.log1p(leverage_gain * (1.0 + sparse_precision))


def observational_logdet_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    regularization: float = 1.0,
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """One-step log-det information gain for candidate observations.

    Each rating observation contributes an interaction feature
    `user_factor * item_factor`. The score approximates the marginal
    information gain from adding that row to a regularized design matrix:
    `log(1 + x^T (X_obs^T X_obs + lambda I)^-1 x)`.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    gain = np.sum(solved * x_cand, axis=1)
    scores = np.log1p(np.maximum(gain, 0.0))
    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return scores * weights


def network_observational_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    adjacency: np.ndarray,
    diffusion_prob: float,
    regularization: float = 1.0,
    max_neighbors: int = 8,
    normalization: str = "degree",
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """Network-aware one-step observational entropy.

    For a candidate (user, item), this approximates the expected information
    gain from observing that entry plus same-item candidate entries from its
    social neighbors. The small set gain is computed as
    `log det(I + X_S P X_S^T)`, where P is the regularized inverse design
    matrix from currently observed interaction features.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    single_gain = np.sum(solved * x_cand, axis=1)
    scores = np.log1p(np.maximum(single_gain, 0.0))
    degree = adjacency.astype(float).sum(axis=1)
    max_neighbors = max(int(max_neighbors), 0)
    p = max(float(diffusion_prob), 0.0)
    if p > 0:
        frame = pd.DataFrame({"row": np.arange(len(candidates), dtype=int), "item": items})
    else:
        frame = pd.DataFrame(columns=["row", "item"])
    for _, group in frame.groupby("item", sort=False):
        same_item_rows = group["row"].to_numpy(dtype=int)
        if len(same_item_rows) <= 1:
            continue
        same_item_users = users[same_item_rows]
        adj_sub = adjacency[np.ix_(same_item_users, same_item_users)].astype(bool)
        has_neighbors = adj_sub.any(axis=1)
        for local_pos in np.flatnonzero(has_neighbors):
            row = int(same_item_rows[local_pos])
            neigh_rows = same_item_rows[adj_sub[local_pos]]
            neigh_rows = neigh_rows[neigh_rows != row]
            if len(neigh_rows) == 0:
                continue
            if max_neighbors and len(neigh_rows) > max_neighbors:
                neigh_degrees = degree[users[neigh_rows]]
                order = np.argsort(neigh_degrees)[:max_neighbors]
                neigh_rows = neigh_rows[order]
            if normalization == "none":
                neighbor_weight = np.sqrt(p)
            else:
                neighbor_weight = np.sqrt(p / max(float(degree[users[row]]), 1.0))
            x_set = np.vstack([x_cand[row], x_cand[neigh_rows] * neighbor_weight])
            kernel = np.eye(x_set.shape[0]) + x_set @ precision @ x_set.T
            sign, logdet = np.linalg.slogdet(kernel)
            if sign > 0:
                scores[row] = max(scores[row], float(logdet))

    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return np.maximum(scores, 0.0) * weights


def network_observational_expected_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    adjacency: np.ndarray,
    diffusion_prob: float,
    regularization: float = 1.0,
    max_neighbors: int = 8,
    normalization: str = "degree",
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """Expected network observational log-det information gain.

    The focal candidate observation is treated as certain. Same-item social
    neighbors are treated as probabilistic observations and contribute their
    first-order expected marginal log-det gain. This is closer to the paper's
    network-enabled acquisition idea than the set-max proxy while remaining
    cheap enough for full candidate lists.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    single_gain = np.sum(solved * x_cand, axis=1)
    base_scores = np.log1p(np.maximum(single_gain, 0.0))
    scores = base_scores.copy()

    degree = adjacency.astype(float).sum(axis=1)
    max_neighbors = max(int(max_neighbors), 0)
    p = max(float(diffusion_prob), 0.0)
    if p > 0 and len(candidates) > 1:
        frame = pd.DataFrame({"row": np.arange(len(candidates), dtype=int), "item": items})
        for _, group in frame.groupby("item", sort=False):
            same_item_rows = group["row"].to_numpy(dtype=int)
            if len(same_item_rows) <= 1:
                continue
            same_item_users = users[same_item_rows]
            adj_sub = adjacency[np.ix_(same_item_users, same_item_users)].astype(bool)
            has_neighbors = adj_sub.any(axis=1)
            for local_pos in np.flatnonzero(has_neighbors):
                row = int(same_item_rows[local_pos])
                neigh_rows = same_item_rows[adj_sub[local_pos]]
                neigh_rows = neigh_rows[neigh_rows != row]
                if len(neigh_rows) == 0:
                    continue
                if max_neighbors and len(neigh_rows) > max_neighbors:
                    neigh_degrees = degree[users[neigh_rows]]
                    order = np.argsort(neigh_degrees)[:max_neighbors]
                    neigh_rows = neigh_rows[order]
                if normalization == "none":
                    response_weight = min(p, 1.0)
                else:
                    response_weight = min(p / max(float(degree[users[row]]), 1.0), 1.0)
                if response_weight <= 0:
                    continue
                expected_extra = 0.0
                for neigh_row in neigh_rows:
                    x_pair = np.vstack([x_cand[row], x_cand[int(neigh_row)]])
                    kernel = np.eye(2) + x_pair @ precision @ x_pair.T
                    sign, logdet = np.linalg.slogdet(kernel)
                    if sign > 0:
                        expected_extra += response_weight * max(float(logdet) - float(base_scores[row]), 0.0)
                scores[row] = float(base_scores[row]) + expected_extra

    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return np.maximum(scores, 0.0) * weights


def network_observational_stochastic_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    adjacency: np.ndarray,
    diffusion_prob: float,
    regularization: float = 1.0,
    max_neighbors: int = 8,
    normalization: str = "degree",
    samples: int = 8,
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """Monte Carlo network observational log-det information gain.

    The focal candidate is always included. Same-item social-neighbor
    candidates are sampled as probabilistic network observations, and the
    returned score is the average small-set log-det gain over deterministic
    per-row Monte Carlo masks.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    single_gain = np.sum(solved * x_cand, axis=1)
    base_scores = np.log1p(np.maximum(single_gain, 0.0))
    scores = base_scores.copy()

    degree = adjacency.astype(float).sum(axis=1)
    max_neighbors = max(int(max_neighbors), 0)
    n_samples = max(int(samples), 1)
    p = max(float(diffusion_prob), 0.0)
    if p > 0 and len(candidates) > 1:
        frame = pd.DataFrame({"row": np.arange(len(candidates), dtype=int), "item": items})
        for _, group in frame.groupby("item", sort=False):
            same_item_rows = group["row"].to_numpy(dtype=int)
            if len(same_item_rows) <= 1:
                continue
            same_item_users = users[same_item_rows]
            adj_sub = adjacency[np.ix_(same_item_users, same_item_users)].astype(bool)
            has_neighbors = adj_sub.any(axis=1)
            for local_pos in np.flatnonzero(has_neighbors):
                row = int(same_item_rows[local_pos])
                neigh_rows = same_item_rows[adj_sub[local_pos]]
                neigh_rows = neigh_rows[neigh_rows != row]
                if len(neigh_rows) == 0:
                    continue
                if max_neighbors and len(neigh_rows) > max_neighbors:
                    neigh_degrees = degree[users[neigh_rows]]
                    order = np.argsort(neigh_degrees)[:max_neighbors]
                    neigh_rows = neigh_rows[order]
                if normalization == "none":
                    inclusion_prob = min(p, 1.0)
                else:
                    inclusion_prob = min(p / max(float(degree[users[row]]), 1.0), 1.0)
                if inclusion_prob <= 0:
                    continue
                sample_scores = []
                for sample_idx in range(n_samples):
                    rng = np.random.default_rng((row + 1) * 1_000_003 + sample_idx * 9_176 + len(neigh_rows))
                    include = rng.random(len(neigh_rows)) < inclusion_prob
                    if not include.any():
                        sample_scores.append(float(base_scores[row]))
                        continue
                    x_set = np.vstack([x_cand[row], x_cand[neigh_rows[include]]])
                    kernel = np.eye(x_set.shape[0]) + x_set @ precision @ x_set.T
                    sign, logdet = np.linalg.slogdet(kernel)
                    sample_scores.append(float(logdet) if sign > 0 else float(base_scores[row]))
                scores[row] = float(np.mean(sample_scores))

    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return np.maximum(scores, 0.0) * weights


def network_observational_reliability_stochastic_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    adjacency: np.ndarray,
    diffusion_prob: float,
    regularization: float = 1.0,
    max_neighbors: int = 8,
    normalization: str = "degree",
    samples: int = 8,
    reliability_strength: float = 1.0,
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """Reliability-calibrated stochastic network observational log-det gain.

    This keeps the same Monte Carlo subset-logdet target as
    `network_observational_stochastic_entropy_scores`, but damps same-item
    social-neighbor inclusion probabilities when the focal or neighbor user is
    structurally unreliable. The calibration is intentionally soft: it reduces
    high-centrality amplification without removing network observations.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    single_gain = np.sum(solved * x_cand, axis=1)
    base_scores = np.log1p(np.maximum(single_gain, 0.0))
    scores = base_scores.copy()

    degree = adjacency.astype(float).sum(axis=1)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users).astype(float)
    positive_degree = degree[degree > 0]
    positive_counts = user_counts[user_counts > 0]
    degree_scale = float(np.median(positive_degree)) if len(positive_degree) else 1.0
    count_scale = float(np.median(positive_counts)) if len(positive_counts) else 1.0
    degree_scale = max(degree_scale, 1.0)
    count_scale = max(count_scale, 1.0)
    strength = max(float(reliability_strength), 0.0)

    degree_factor = np.sqrt(degree_scale / (degree + degree_scale))
    count_factor = np.sqrt((user_counts + 1.0) / (user_counts + count_scale + 1.0))
    user_reliability = np.power(np.clip(degree_factor * count_factor, 0.0, 1.0), strength)
    user_reliability = 0.5 + 0.5 * user_reliability

    max_neighbors = max(int(max_neighbors), 0)
    n_samples = max(int(samples), 1)
    p = max(float(diffusion_prob), 0.0)
    if p > 0 and len(candidates) > 1:
        frame = pd.DataFrame({"row": np.arange(len(candidates), dtype=int), "item": items})
        for _, group in frame.groupby("item", sort=False):
            same_item_rows = group["row"].to_numpy(dtype=int)
            if len(same_item_rows) <= 1:
                continue
            same_item_users = users[same_item_rows]
            adj_sub = adjacency[np.ix_(same_item_users, same_item_users)].astype(bool)
            has_neighbors = adj_sub.any(axis=1)
            for local_pos in np.flatnonzero(has_neighbors):
                row = int(same_item_rows[local_pos])
                row_user = int(users[row])
                neigh_rows = same_item_rows[adj_sub[local_pos]]
                neigh_rows = neigh_rows[neigh_rows != row]
                if len(neigh_rows) == 0:
                    continue
                if max_neighbors and len(neigh_rows) > max_neighbors:
                    neigh_reliability = user_reliability[users[neigh_rows]]
                    order = np.argsort(neigh_reliability)[::-1][:max_neighbors]
                    neigh_rows = neigh_rows[order]
                if normalization == "none":
                    base_prob = min(p, 1.0)
                else:
                    base_prob = min(p / max(float(degree[row_user]), 1.0), 1.0)
                inclusion_prob = np.clip(
                    base_prob * user_reliability[row_user] * user_reliability[users[neigh_rows]],
                    0.0,
                    1.0,
                )
                if not (inclusion_prob > 0).any():
                    continue
                sample_scores = []
                for sample_idx in range(n_samples):
                    rng = np.random.default_rng((row + 1) * 1_000_003 + sample_idx * 9_176 + len(neigh_rows))
                    include = rng.random(len(neigh_rows)) < inclusion_prob
                    if not include.any():
                        sample_scores.append(float(base_scores[row]))
                        continue
                    x_set = np.vstack([x_cand[row], x_cand[neigh_rows[include]]])
                    kernel = np.eye(x_set.shape[0]) + x_set @ precision @ x_set.T
                    sign, logdet = np.linalg.slogdet(kernel)
                    sample_scores.append(float(logdet) if sign > 0 else float(base_scores[row]))
                scores[row] = float(np.mean(sample_scores))

    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return np.maximum(scores, 0.0) * weights


def network_observational_weighted_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    adjacency: np.ndarray,
    diffusion_prob: float,
    regularization: float = 1.0,
    max_neighbors: int = 8,
    normalization: str = "degree",
    reliability_strength: float = 1.0,
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """Probability-weighted network observational log-det gain.

    This deterministic estimator treats the focal candidate as a certain
    observation and same-item social-neighbor candidates as expected
    observations. Neighbor features are weighted by the square root of a
    calibrated inclusion probability before computing the small-set log-det,
    which approximates an expected information matrix while avoiding the
    high-degree collapse seen in unnormalized stochastic subsets.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    single_gain = np.sum(solved * x_cand, axis=1)
    base_scores = np.log1p(np.maximum(single_gain, 0.0))
    scores = base_scores.copy()

    degree = adjacency.astype(float).sum(axis=1)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users).astype(float)
    positive_degree = degree[degree > 0]
    positive_counts = user_counts[user_counts > 0]
    degree_scale = max(float(np.median(positive_degree)) if len(positive_degree) else 1.0, 1.0)
    count_scale = max(float(np.median(positive_counts)) if len(positive_counts) else 1.0, 1.0)
    strength = max(float(reliability_strength), 0.0)
    degree_factor = np.sqrt(degree_scale / (degree + degree_scale))
    count_factor = np.sqrt((user_counts + 1.0) / (user_counts + count_scale + 1.0))
    user_reliability = np.power(np.clip(degree_factor * count_factor, 0.0, 1.0), strength)
    user_reliability = 0.5 + 0.5 * user_reliability

    score_scale = max(float(np.percentile(base_scores, 95)), 1e-8)
    info_weight = np.clip(base_scores / score_scale, 0.25, 2.0)
    max_neighbors = max(int(max_neighbors), 0)
    p = max(float(diffusion_prob), 0.0)
    if p > 0 and len(candidates) > 1:
        frame = pd.DataFrame({"row": np.arange(len(candidates), dtype=int), "item": items})
        for _, group in frame.groupby("item", sort=False):
            same_item_rows = group["row"].to_numpy(dtype=int)
            if len(same_item_rows) <= 1:
                continue
            same_item_users = users[same_item_rows]
            adj_sub = adjacency[np.ix_(same_item_users, same_item_users)].astype(bool)
            has_neighbors = adj_sub.any(axis=1)
            for local_pos in np.flatnonzero(has_neighbors):
                row = int(same_item_rows[local_pos])
                row_user = int(users[row])
                neigh_rows = same_item_rows[adj_sub[local_pos]]
                neigh_rows = neigh_rows[neigh_rows != row]
                if len(neigh_rows) == 0:
                    continue
                if normalization == "none":
                    base_prob = min(p, 1.0)
                else:
                    base_prob = min(p / max(float(degree[row_user]), 1.0), 1.0)
                probabilities = np.clip(
                    base_prob
                    * user_reliability[row_user]
                    * user_reliability[users[neigh_rows]]
                    * info_weight[neigh_rows],
                    0.0,
                    1.0,
                )
                if not (probabilities > 0).any():
                    continue
                if max_neighbors and len(neigh_rows) > max_neighbors:
                    order = np.argsort(probabilities * base_scores[neigh_rows])[::-1][:max_neighbors]
                    neigh_rows = neigh_rows[order]
                    probabilities = probabilities[order]
                weighted_neighbors = x_cand[neigh_rows] * np.sqrt(probabilities)[:, None]
                x_set = np.vstack([x_cand[row], weighted_neighbors])
                kernel = np.eye(x_set.shape[0]) + x_set @ precision @ x_set.T
                sign, logdet = np.linalg.slogdet(kernel)
                if sign > 0:
                    scores[row] = max(float(base_scores[row]), float(logdet))

    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return np.maximum(scores, 0.0) * weights


def network_neighbor_prediction_gain_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    adjacency: np.ndarray,
    diffusion_prob: float,
    regularization: float = 1.0,
    max_neighbors: int = 8,
    normalization: str = "degree",
    reliability_strength: float = 1.0,
    predicted_values: np.ndarray | None = None,
    rating_midpoint: float = 3.0,
    weighting: str = "none",
    weight_strength: float = 1.0,
) -> np.ndarray:
    """Network entropy from focal observations reducing neighbor uncertainty.

    Unlike set-logdet variants that reward the whole focal-plus-neighbor set,
    this estimates the marginal reduction in same-item social-neighbor
    predictive variance caused by observing the focal candidate. For ridge
    precision P and focal feature x, the reduction for neighbor feature z is
    `(z^T P x)^2 / (1 + x^T P x)`.
    """
    rank = user_factors.shape[1]
    reg = max(float(regularization), 1e-8)
    if observed.empty:
        precision = np.eye(rank) / reg
    else:
        obs_users = observed["user_id"].to_numpy(dtype=int)
        obs_items = observed["item_id"].to_numpy(dtype=int)
        x_obs = user_factors[obs_users] * item_factors[obs_items]
        gram = x_obs.T @ x_obs + reg * np.eye(rank)
        precision = np.linalg.pinv(gram)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    x_cand = user_factors[users] * item_factors[items]
    solved = x_cand @ precision
    self_gain = np.sum(solved * x_cand, axis=1)
    base_scores = np.log1p(np.maximum(self_gain, 0.0))
    scores = base_scores.copy()

    degree = adjacency.astype(float).sum(axis=1)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users).astype(float)
    positive_degree = degree[degree > 0]
    positive_counts = user_counts[user_counts > 0]
    degree_scale = max(float(np.median(positive_degree)) if len(positive_degree) else 1.0, 1.0)
    count_scale = max(float(np.median(positive_counts)) if len(positive_counts) else 1.0, 1.0)
    strength = max(float(reliability_strength), 0.0)
    degree_factor = np.sqrt(degree_scale / (degree + degree_scale))
    count_factor = np.sqrt((user_counts + 1.0) / (user_counts + count_scale + 1.0))
    user_reliability = np.power(np.clip(degree_factor * count_factor, 0.0, 1.0), strength)
    user_reliability = 0.5 + 0.5 * user_reliability

    max_neighbors = max(int(max_neighbors), 0)
    p = max(float(diffusion_prob), 0.0)
    if p > 0 and len(candidates) > 1:
        frame = pd.DataFrame({"row": np.arange(len(candidates), dtype=int), "item": items})
        for _, group in frame.groupby("item", sort=False):
            same_item_rows = group["row"].to_numpy(dtype=int)
            if len(same_item_rows) <= 1:
                continue
            same_item_users = users[same_item_rows]
            adj_sub = adjacency[np.ix_(same_item_users, same_item_users)].astype(bool)
            has_neighbors = adj_sub.any(axis=1)
            for local_pos in np.flatnonzero(has_neighbors):
                row = int(same_item_rows[local_pos])
                row_user = int(users[row])
                neigh_rows = same_item_rows[adj_sub[local_pos]]
                neigh_rows = neigh_rows[neigh_rows != row]
                if len(neigh_rows) == 0:
                    continue

                cross = x_cand[neigh_rows] @ solved[row]
                reductions = np.maximum((cross * cross) / max(1.0 + float(self_gain[row]), 1e-12), 0.0)
                if max_neighbors and len(neigh_rows) > max_neighbors:
                    order = np.argsort(reductions)[::-1][:max_neighbors]
                    neigh_rows = neigh_rows[order]
                    reductions = reductions[order]

                if normalization == "none":
                    base_prob = min(p, 1.0)
                else:
                    base_prob = min(p / max(float(degree[row_user]), 1.0), 1.0)
                probabilities = np.clip(
                    base_prob * user_reliability[row_user] * user_reliability[users[neigh_rows]],
                    0.0,
                    1.0,
                )
                if (probabilities > 0).any():
                    scores[row] = float(base_scores[row]) + float(np.sum(probabilities * np.log1p(reductions)))

    weights = _logdet_candidate_weights(
        candidates,
        observed,
        num_users,
        num_items,
        predicted_values=predicted_values,
        rating_midpoint=rating_midpoint,
        weighting=weighting,
        weight_strength=weight_strength,
    )
    return np.maximum(scores, 0.0) * weights


def _logdet_candidate_weights(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    predicted_values: np.ndarray | None,
    rating_midpoint: float,
    weighting: str,
    weight_strength: float,
) -> np.ndarray:
    mode = weighting.lower()
    if mode in {"none", ""}:
        return np.ones(len(candidates), dtype=float)

    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    user_counts = np.bincount(observed["user_id"].to_numpy(dtype=int), minlength=num_users)
    item_counts = np.bincount(observed["item_id"].to_numpy(dtype=int), minlength=num_items)
    strength = max(float(weight_strength), 0.0)
    weights = np.ones(len(candidates), dtype=float)

    if mode in {"reward", "reward_cold"}:
        if predicted_values is None:
            reward_weight = np.ones(len(candidates), dtype=float)
        else:
            positive = np.maximum(predicted_values.astype(float) - rating_midpoint, 0.0)
            denom = max(float(np.percentile(positive, 95)), 1e-8)
            reward_weight = 1.0 + strength * np.clip(positive / denom, 0.0, 2.0)
        weights *= reward_weight

    if mode in {"cold", "reward_cold"}:
        cold = 1.0 / np.sqrt(user_counts[users] + 1.0) + 1.0 / np.sqrt(item_counts[items] + 1.0)
        cold = cold / max(float(np.percentile(cold, 95)), 1e-8)
        weights *= 1.0 + strength * np.clip(cold, 0.0, 2.0)

    if mode not in {"reward", "cold", "reward_cold"}:
        raise ValueError(f"Unknown logdet weighting mode: {weighting}")
    return weights


def hybrid_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    coverage_weight: float = 0.5,
) -> np.ndarray:
    coverage = coverage_entropy_scores(candidates, observed, num_users, num_items)
    subspace = subspace_entropy_scores(candidates, user_factors, item_factors)
    return coverage_weight * _standardize(coverage) + (1.0 - coverage_weight) * _standardize(subspace)


def mixed_observational_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    coverage_weight: float = 0.25,
) -> np.ndarray:
    coverage = coverage_entropy_scores(candidates, observed, num_users, num_items)
    observational = observational_entropy_scores(candidates, observed, num_users, num_items, user_factors, item_factors)
    return coverage_weight * _standardize(coverage) + (1.0 - coverage_weight) * _standardize(observational)


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def network_entropy_scores(
    base_entropy: np.ndarray,
    candidates: pd.DataFrame,
    adjacency: np.ndarray,
    diffusion_prob: float,
    normalization: str = "degree",
) -> np.ndarray:
    """Approximate network information value by adding same-item neighbor uncertainty."""
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    out = base_entropy.astype(float).copy()
    frame = pd.DataFrame({"row": np.arange(len(candidates)), "user": users, "item": items})
    for _, group in frame.groupby("item", sort=False):
        rows = group["row"].to_numpy(dtype=int)
        if len(rows) <= 1:
            continue
        group_users = group["user"].to_numpy(dtype=int)
        adj_sub = adjacency[np.ix_(group_users, group_users)].astype(float)
        neigh_counts = adj_sub.sum(axis=1)
        has_neighbors = neigh_counts > 0
        if has_neighbors.any():
            neigh_mean = np.zeros(len(rows), dtype=float)
            if normalization == "none":
                neigh_mean[has_neighbors] = adj_sub[has_neighbors] @ base_entropy[rows]
            else:
                neigh_mean[has_neighbors] = (adj_sub[has_neighbors] @ base_entropy[rows]) / neigh_counts[has_neighbors]
            out[rows] += diffusion_prob * neigh_mean
    return out
