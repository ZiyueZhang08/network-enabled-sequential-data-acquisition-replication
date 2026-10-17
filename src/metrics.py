from __future__ import annotations

import numpy as np
import pandas as pd

from .matrix_factorization import MatrixFactorization


def _prediction_errors(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    user_factors: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = ratings if mask is None else ratings.loc[np.asarray(mask, dtype=bool)]
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    truth = frame["rating"].to_numpy(dtype=float)
    pred = model.predict_pairs(users, items, user_factors=user_factors)
    return truth, truth - pred


def rmse(model: MatrixFactorization, ratings: pd.DataFrame, user_factors: np.ndarray | None = None) -> float:
    return standard_mask_rmse(model, ratings, user_factors=user_factors)


def pdf_literal_norm_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    user_factors: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """Literal placement from the PDF: norm(error) divided by mask size."""
    _, err = _prediction_errors(model, ratings, user_factors=user_factors, mask=mask)
    if len(err) == 0:
        return np.nan
    return float(np.sqrt(np.sum(err * err)) / len(err))


def standard_mask_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    user_factors: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """Conventional masked RMSE: sqrt(mean squared error)."""
    _, err = _prediction_errors(model, ratings, user_factors=user_factors, mask=mask)
    if len(err) == 0:
        return np.nan
    return float(np.sqrt(np.mean(err * err)))


def cumulative_reward(ratings: pd.DataFrame, midpoint: float = 3.0) -> float:
    """Paper acquisition reward: sum sigmoid(Y_i - midpoint)."""
    values = ratings["rating"].to_numpy(dtype=float) - midpoint
    values = np.clip(values, -40.0, 40.0)
    return float(np.sum(1.0 / (1.0 + np.exp(-values))))


def weighted_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    midpoint: float = 3.0,
    user_factors: np.ndarray | None = None,
) -> float:
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    truth = ratings["rating"].to_numpy(dtype=float)
    pred = model.predict_pairs(users, items, user_factors=user_factors)
    weights = np.maximum(truth - midpoint, 0.0) + 0.1
    return float(np.sqrt(np.sum(weights * (truth - pred) ** 2) / np.sum(weights)))


def reward_weighted_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    midpoint: float = 3.0,
    beta: float = 1.0,
    user_factors: np.ndarray | None = None,
) -> float:
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    truth = ratings["rating"].to_numpy(dtype=float)
    pred = model.predict_pairs(users, items, user_factors=user_factors)
    weights = 1.0 + max(float(beta), 0.0) * np.maximum(truth - midpoint, 0.0)
    return float(np.sqrt(np.sum(weights * (truth - pred) ** 2) / np.sum(weights)))


def sigmoid_reward_weighted_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    midpoint: float = 3.0,
    user_factors: np.ndarray | None = None,
) -> float:
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    truth = ratings["rating"].to_numpy(dtype=float)
    pred = model.predict_pairs(users, items, user_factors=user_factors)
    weights = 1.0 / (1.0 + np.exp(-(truth - midpoint)))
    return float(np.sqrt(np.sum(weights * (truth - pred) ** 2) / np.sum(weights)))


def linear_reward_norm_weighted_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    midpoint: float = 3.0,
    user_factors: np.ndarray | None = None,
) -> float:
    """Reward-weighted error norm where weights multiply the prediction error."""
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    truth = ratings["rating"].to_numpy(dtype=float)
    pred = model.predict_pairs(users, items, user_factors=user_factors)
    weights = np.maximum(truth - midpoint, 0.0) + 0.1
    return float(np.sqrt(np.sum((weights * (truth - pred)) ** 2) / len(truth)))


def reward_tail_proxy_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    midpoint: float = 3.0,
    user_factors: np.ndarray | None = None,
) -> float:
    """Diagnostic-only tail proxy using max(y - midpoint, 0) + 0.1 weights."""
    return linear_reward_norm_weighted_rmse(
        model,
        ratings,
        midpoint=midpoint,
        user_factors=user_factors,
    )


def positive_reward_norm_weighted_rmse(
    model: MatrixFactorization,
    ratings: pd.DataFrame,
    midpoint: float = 3.0,
    user_factors: np.ndarray | None = None,
) -> float:
    """Positive-reward-only variant of the weighted prediction-error norm."""
    users = ratings["user_id"].to_numpy(dtype=int)
    items = ratings["item_id"].to_numpy(dtype=int)
    truth = ratings["rating"].to_numpy(dtype=float)
    pred = model.predict_pairs(users, items, user_factors=user_factors)
    weights = np.maximum(truth - midpoint, 0.0)
    return float(np.sqrt(np.sum((weights * (truth - pred)) ** 2) / len(truth)))
