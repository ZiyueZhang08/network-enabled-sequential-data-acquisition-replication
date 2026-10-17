from __future__ import annotations

import numpy as np
import pandas as pd

from .matrix_factorization import MatrixFactorization


def predicted_reward_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    user_factors: np.ndarray | None = None,
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    return model.predict_pairs(users, items, user_factors=user_factors)


def network_reward_scores(
    model: MatrixFactorization,
    candidates: pd.DataFrame,
    adjacency: np.ndarray,
    diffusion_prob: float,
    user_factors: np.ndarray | None = None,
    normalization: str = "none",
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    direct = model.predict_pairs(users, items, user_factors=user_factors)
    out = direct.copy()
    frame = pd.DataFrame({"row": np.arange(len(candidates)), "user": users, "item": items})
    for _, group in frame.groupby("item", sort=False):
        rows = group["row"].to_numpy(dtype=int)
        if len(rows) <= 1:
            continue
        group_users = group["user"].to_numpy(dtype=int)
        adj_sub = adjacency[np.ix_(group_users, group_users)].astype(float)
        neighbor_sum = adj_sub @ direct[rows]
        if normalization == "degree":
            degree = adj_sub.sum(axis=1)
            has_neighbors = degree > 0
            normalized = np.zeros_like(neighbor_sum)
            normalized[has_neighbors] = neighbor_sum[has_neighbors] / degree[has_neighbors]
            neighbor_sum = normalized
        elif normalization == "sqrt_degree":
            degree = np.sqrt(np.maximum(adj_sub.sum(axis=1), 1.0))
            neighbor_sum = neighbor_sum / degree
        out[rows] += diffusion_prob * neighbor_sum
    return out


def true_centered_reward(batch: pd.DataFrame, midpoint: float = 3.0) -> float:
    return float((batch["rating"].to_numpy(dtype=float) - midpoint).sum())
