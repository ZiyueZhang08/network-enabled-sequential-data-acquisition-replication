from __future__ import annotations

import numpy as np
import pandas as pd


def _orthonormal_basis(factors: np.ndarray) -> np.ndarray:
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 2 or factors.shape[0] == 0 or factors.shape[1] == 0:
        return np.zeros((factors.shape[0], 0), dtype=float)
    u, singular_values, _ = np.linalg.svd(factors, full_matrices=False)
    if len(singular_values) == 0:
        return np.zeros((factors.shape[0], 0), dtype=float)
    tol = max(factors.shape) * np.finfo(float).eps * float(singular_values[0])
    rank = int(np.sum(singular_values > tol))
    return u[:, : max(rank, 1)]


def coherence_scores(user_factors_dls: np.ndarray, item_factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_user = _orthonormal_basis(user_factors_dls)
    q_item = _orthonormal_basis(item_factors)
    return np.sum(q_user * q_user, axis=1), np.sum(q_item * q_item, axis=1)


def entropy_coherence_base_scores(
    candidates: pd.DataFrame,
    user_factors_dls: np.ndarray,
    item_factors: np.ndarray,
) -> np.ndarray:
    users = candidates["user_id"].to_numpy(dtype=int)
    items = candidates["item_id"].to_numpy(dtype=int)
    mu_user, mu_item = coherence_scores(user_factors_dls, item_factors)
    return mu_user[users] * mu_item[items]


def _entry_kernel_features(frame: pd.DataFrame, q_user: np.ndarray, q_item: np.ndarray) -> np.ndarray:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    if len(frame) == 0 or q_user.shape[1] == 0 or q_item.shape[1] == 0:
        return np.zeros((len(frame), 0), dtype=float)
    return np.einsum("nr,ns->nrs", q_user[users], q_item[items], optimize=True).reshape(len(frame), -1)


def _precision_from_observed_kernel(
    observed: pd.DataFrame,
    q_user: np.ndarray,
    q_item: np.ndarray,
    regularization: float,
) -> np.ndarray | None:
    reg = max(float(regularization), 1e-8)
    rank = int(q_user.shape[1] * q_item.shape[1])
    if rank == 0:
        return None
    if observed.empty:
        return np.eye(rank, dtype=float) / reg
    observed_features = _entry_kernel_features(observed, q_user, q_item)
    gram = observed_features.T @ observed_features + reg * np.eye(rank, dtype=float)
    try:
        chol = np.linalg.cholesky(gram)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(rank, dtype=float)))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram)


def _posterior_variance(
    features: np.ndarray,
    precision: np.ndarray,
    regularization: float,
    batch_size: int,
) -> np.ndarray:
    reg = max(float(regularization), 1e-8)
    batch = max(int(batch_size), 1)
    out = np.zeros(features.shape[0], dtype=float)
    for start in range(0, features.shape[0], batch):
        block = features[start : start + batch]
        solved = block @ precision
        out[start : start + batch] = reg * np.sum(solved * block, axis=1)
    return out


def entropy_observational_lite_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors_dls: np.ndarray,
    item_factors: np.ndarray,
    regularization: float = 1.0,
    correction_weight: float = 1.0,
    shortlist_size: int = 20_000,
    batch_size: int = 4096,
) -> np.ndarray:
    """Coherence entropy with a Schur-complement observed-subspace correction.

    Candidate and observed entries are embedded with the paper-style
    cross-coherence kernel

        k((i,j),(i',j')) = <P_U e_i, P_U e_i'> <P_V e_j, P_V e_j'>.

    For a candidate entry, the returned value is the diagonal Schur complement
    base - nu^T (L_t + reg I)^-1 nu.  The calculation is done in the equivalent
    low-rank feature space rather than forming the full observed-by-observed
    kernel matrix.  When `shortlist_size` is positive, only the highest base
    coherence candidates receive the corrected score; the rest keep the
    first-order coherence value.
    """
    base = entropy_coherence_base_scores(candidates, user_factors_dls, item_factors)
    if len(candidates) == 0:
        return base
    if observed.empty or max(float(correction_weight), 0.0) == 0.0:
        return base

    q_user = _orthonormal_basis(user_factors_dls)
    q_item = _orthonormal_basis(item_factors)
    precision = _precision_from_observed_kernel(observed, q_user, q_item, regularization)
    if precision is None:
        return base

    if shortlist_size > 0 and int(shortlist_size) < len(candidates):
        size = max(int(shortlist_size), 1)
        selected_idx = np.argpartition(base, -size)[-size:]
        scores = base.copy()
    else:
        selected_idx = np.arange(len(candidates))
        scores = base.copy()

    selected_frame = candidates.iloc[selected_idx].reset_index(drop=True)
    features = _entry_kernel_features(selected_frame, q_user, q_item)
    posterior = _posterior_variance(features, precision, regularization, batch_size)
    posterior = np.clip(posterior, 0.0, base[selected_idx])
    weight = max(float(correction_weight), 0.0)
    corrected = base[selected_idx] - weight * (base[selected_idx] - posterior)
    scores[selected_idx] = np.clip(corrected, 0.0, base[selected_idx])
    return scores
