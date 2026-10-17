from __future__ import annotations

import numpy as np


def compute_dls_neighbors(network_embedding: np.ndarray, top_k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    n = network_embedding.shape[0]
    k = min(max(1, top_k), n)
    diff = network_embedding[:, None, :] - network_embedding[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    neighbor_idx = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
    neighbor_dist_sq = np.take_along_axis(dist_sq, neighbor_idx, axis=1)
    return neighbor_idx, neighbor_dist_sq


def smooth_dls_user_factors(
    user_factors: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_dist_sq: np.ndarray,
    sigma: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    n = user_factors.shape[0]
    smoothed = np.zeros_like(user_factors)
    effective_neighborhood = np.zeros(n)
    denom = max(2.0 * sigma * sigma, 1e-12)
    for u in range(n):
        idx = neighbor_idx[u]
        weights = np.exp(-neighbor_dist_sq[u] / denom)
        weights = weights / max(float(weights.sum()), 1e-12)
        smoothed[u] = weights @ user_factors[idx]
        effective_neighborhood[u] = 1.0 / max(float(np.sum(weights * weights)), 1e-12)
    return smoothed, effective_neighborhood


def shrink_dls_user_factors(
    raw_user_factors: np.ndarray,
    smoothed_user_factors: np.ndarray,
    effective_neighborhood: np.ndarray,
    user_degree: np.ndarray,
    top_k: int,
    strength: float = 0.6,
    user_observed_counts: np.ndarray | None = None,
    mode: str = "adaptive",
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink DLS smoothing toward raw MF factors when graph evidence is weak."""
    max_strength = float(np.clip(strength, 0.0, 1.0))
    if max_strength <= 0.0:
        return raw_user_factors, np.zeros(raw_user_factors.shape[0], dtype=float)

    degree = user_degree.astype(float)
    positive_degree = degree[degree > 0]
    degree_scale = float(np.median(positive_degree)) if len(positive_degree) else 1.0
    degree_confidence = degree / (degree + max(degree_scale, 1.0))
    effective_confidence = np.clip(effective_neighborhood / max(float(top_k), 1.0), 0.0, 1.0)
    if mode in {"fixed", "residual"}:
        confidence = np.ones_like(degree_confidence)
    elif mode == "adaptive":
        confidence = np.sqrt(np.clip(degree_confidence * effective_confidence, 0.0, 1.0))
    elif mode == "reliability":
        if user_observed_counts is None:
            count_confidence = np.ones_like(degree_confidence)
        else:
            counts = user_observed_counts.astype(float)
            positive_counts = counts[counts > 0]
            count_scale = float(np.median(positive_counts)) if len(positive_counts) else 1.0
            count_confidence = counts / (counts + max(count_scale, 1.0))
        confidence = np.clip(degree_confidence * effective_confidence * count_confidence, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown DLS shrinkage mode: {mode}")
    rho = max_strength * confidence
    mixed = (1.0 - rho[:, None]) * raw_user_factors + rho[:, None] * smoothed_user_factors
    return mixed, rho


def compute_dls_user_factors(
    user_factors: np.ndarray,
    network_embedding: np.ndarray,
    sigma: float = 0.4,
    top_k: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic Local Smoothness using Gaussian weights over graph-embedding neighbors."""
    neighbor_idx, neighbor_dist_sq = compute_dls_neighbors(network_embedding, top_k=top_k)
    return smooth_dls_user_factors(user_factors, neighbor_idx, neighbor_dist_sq, sigma=sigma)
