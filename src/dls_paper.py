from __future__ import annotations

import numpy as np


def gaussian_kernel_weights(neighbor_dist_sq: np.ndarray, sigma: float) -> np.ndarray:
    """Row-normalized Gaussian DLS weights from squared neighbor distances."""
    denom = max(2.0 * float(sigma) * float(sigma), 1e-12)
    logits = -np.asarray(neighbor_dist_sq, dtype=float) / denom
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def paper_gaussian_dls_user_factors(
    user_factors: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_dist_sq: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 3.3-3.4 style Gaussian DLS without residual blending."""
    weights = gaussian_kernel_weights(neighbor_dist_sq, sigma=sigma)
    smoothed = np.einsum("uk,ukr->ur", weights, user_factors[neighbor_idx])
    effective_neighborhood = 1.0 / np.maximum(np.sum(weights * weights, axis=1), 1e-12)
    return smoothed, effective_neighborhood


def full_gaussian_dls_user_factors(
    user_factors: np.ndarray,
    network_embedding: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-neighborhood Gaussian DLS for small/medium user graphs."""
    embedding = np.asarray(network_embedding, dtype=float)
    diff = embedding[:, None, :] - embedding[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    weights = gaussian_kernel_weights(dist_sq, sigma=sigma)
    smoothed = weights @ user_factors
    effective_neighborhood = 1.0 / np.maximum(np.sum(weights * weights, axis=1), 1e-12)
    return smoothed, effective_neighborhood


def scoped_gaussian_dls_user_factors(
    user_factors: np.ndarray,
    network_embedding: np.ndarray,
    neighbor_idx: np.ndarray | None,
    neighbor_dist_sq: np.ndarray | None,
    sigma: float,
    scope: str = "neighbors",
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian DLS with an explicit Eq. 3.3 full-user option.

    The paper's Eq. 3.3 normalizes over all users. Neighbor scope is retained as
    a computational approximation for larger graphs and for legacy baselines.
    """
    mode = str(scope or "neighbors").lower()
    if mode in {"full", "all", "all_users", "global", "eq3_3", "paper"}:
        return full_gaussian_dls_user_factors(user_factors, network_embedding, sigma=sigma)
    if mode in {"neighbors", "neighbours", "neighbor", "neighbour", "topk", "graph_edges"}:
        if neighbor_idx is None or neighbor_dist_sq is None:
            raise ValueError("Neighbor DLS scope requires neighbor_idx and neighbor_dist_sq.")
        return paper_gaussian_dls_user_factors(
            user_factors,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=sigma,
        )
    raise ValueError(f"Unknown paper_dls_scope: {scope}")
