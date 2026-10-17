from __future__ import annotations

import numpy as np
import pandas as pd


def _orthonormal_basis(factors: np.ndarray) -> np.ndarray:
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 2:
        raise ValueError("factors must be a 2-D array")
    if factors.shape[0] == 0 or factors.shape[1] == 0:
        return np.zeros((factors.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(factors, full_matrices=False)
    if len(singular_values) == 0 or float(singular_values[0]) <= 0.0:
        return np.zeros((factors.shape[0], 0), dtype=float)
    tol = max(factors.shape) * np.finfo(float).eps * float(singular_values[0])
    rank = int(np.sum(singular_values > tol))
    if rank <= 0:
        return np.zeros((factors.shape[0], 0), dtype=float)
    return left[:, :rank]


def _entry_kernel_features(frame: pd.DataFrame, q_user: np.ndarray, q_item: np.ndarray) -> np.ndarray:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    if len(frame) == 0 or q_user.shape[1] == 0 or q_item.shape[1] == 0:
        return np.zeros((len(frame), 0), dtype=float)
    return np.einsum("nr,ns->nrs", q_user[users], q_item[items], optimize=True).reshape(len(frame), -1)


def _coherence_base(frame: pd.DataFrame, q_user: np.ndarray, q_item: np.ndarray) -> np.ndarray:
    users = frame["user_id"].to_numpy(dtype=int)
    items = frame["item_id"].to_numpy(dtype=int)
    user_mu = np.sum(q_user * q_user, axis=1)
    item_mu = np.sum(q_item * q_item, axis=1)
    return user_mu[users] * item_mu[items]


def _posterior_precision(
    observed: pd.DataFrame,
    q_user: np.ndarray,
    q_item: np.ndarray,
    lambda_noise: float,
) -> np.ndarray | None:
    rank = int(q_user.shape[1] * q_item.shape[1])
    if rank == 0:
        return None
    lam = max(float(lambda_noise), 1e-8)
    if observed.empty:
        return np.eye(rank, dtype=float) / lam
    observed_features = _entry_kernel_features(observed, q_user, q_item)
    gram = observed_features.T @ observed_features + lam * np.eye(rank, dtype=float)
    try:
        chol = np.linalg.cholesky(gram)
        identity = np.eye(rank, dtype=float)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, identity))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram)


def _conditional_variance(
    frame: pd.DataFrame,
    q_user: np.ndarray,
    q_item: np.ndarray,
    precision: np.ndarray,
    lambda_noise: float,
    batch_size: int,
) -> np.ndarray:
    lam = max(float(lambda_noise), 1e-8)
    batch = max(int(batch_size), 1)
    out = np.zeros(len(frame), dtype=float)
    for start in range(0, len(frame), batch):
        block = frame.iloc[start : start + batch]
        features = _entry_kernel_features(block, q_user, q_item)
        solved = features @ precision
        out[start : start + len(block)] = lam * np.sum(solved * features, axis=1)
    return out


def explicit_kernel_schur_conditional_variance_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors_dls: np.ndarray,
    item_factors: np.ndarray,
    lambda_noise: float = 1.0,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute the diagonal Schur complement through the observed kernel.

    This is the direct kernel-space form

        k_xx - k_xO @ (K_OO + lambda I)^-1 @ k_Ox,

    kept for audits and tests.  `schur_conditional_variance_entropy_scores`
    uses the algebraically equivalent low-rank feature-space form, which is
    much cheaper when the observed set is large.
    """
    if len(candidates) == 0:
        return np.zeros(0, dtype=float)

    q_user = _orthonormal_basis(user_factors_dls)
    q_item = _orthonormal_basis(item_factors)
    if q_user.shape[1] == 0 or q_item.shape[1] == 0:
        return np.zeros(len(candidates), dtype=float)

    base = _coherence_base(candidates, q_user, q_item)
    if observed.empty:
        return base

    lam = max(float(lambda_noise), 1e-8)
    observed_features = _entry_kernel_features(observed, q_user, q_item)
    observed_kernel = observed_features @ observed_features.T
    system = observed_kernel + lam * np.eye(len(observed_features), dtype=float)
    try:
        chol = np.linalg.cholesky(system)

        def solve(rhs: np.ndarray) -> np.ndarray:
            return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))

    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(system)

        def solve(rhs: np.ndarray) -> np.ndarray:
            return pinv @ rhs

    batch = max(int(batch_size), 1)
    out = np.zeros(len(candidates), dtype=float)
    for start in range(0, len(candidates), batch):
        block = candidates.iloc[start : start + batch]
        features = _entry_kernel_features(block, q_user, q_item)
        cross = features @ observed_features.T
        corrected = np.sum(cross * solve(cross.T).T, axis=1)
        block_base = base[start : start + len(block)]
        out[start : start + len(block)] = np.clip(block_base - corrected, 0.0, block_base)
    return out


def schur_conditional_variance_entropy_scores(
    candidates: pd.DataFrame,
    observed: pd.DataFrame,
    user_factors_dls: np.ndarray,
    item_factors: np.ndarray,
    lambda_noise: float = 1.0,
    shortlist_size: int = 20_000,
    batch_size: int = 4096,
) -> np.ndarray:
    """Eq. 3.6-style conditional variance in low-rank feature space.

    Each entry is represented as kron(q_user[u], q_item[i]), where q_user and
    q_item are orthonormal bases of the DLS user factors and item factors.  The
    returned score is

        lambda * phi.T @ (Phi_obs.T @ Phi_obs + lambda I)^-1 @ phi.

    When `shortlist_size` is positive, only the highest coherence-base
    candidates receive this Schur correction and non-shortlisted candidates get
    zero entropy score.  That keeps the approximation from mixing corrected and
    uncorrected information-value definitions.
    """
    if len(candidates) == 0:
        return np.zeros(0, dtype=float)

    q_user = _orthonormal_basis(user_factors_dls)
    q_item = _orthonormal_basis(item_factors)
    if q_user.shape[1] == 0 or q_item.shape[1] == 0:
        return np.zeros(len(candidates), dtype=float)

    base = _coherence_base(candidates, q_user, q_item)
    precision = _posterior_precision(observed, q_user, q_item, lambda_noise)
    if precision is None:
        return np.zeros(len(candidates), dtype=float)

    if shortlist_size > 0 and int(shortlist_size) < len(candidates):
        size = max(int(shortlist_size), 1)
        selected_idx = np.argpartition(base, -size)[-size:]
        scores = np.zeros(len(candidates), dtype=float)
    else:
        selected_idx = np.arange(len(candidates))
        scores = np.zeros(len(candidates), dtype=float)

    selected = candidates.iloc[selected_idx]
    values = _conditional_variance(
        selected,
        q_user,
        q_item,
        precision,
        lambda_noise=lambda_noise,
        batch_size=batch_size,
    )
    scores[selected_idx] = np.clip(values, 0.0, base[selected_idx])
    return scores
