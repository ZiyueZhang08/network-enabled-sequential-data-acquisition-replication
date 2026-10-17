from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MFConfig:
    rank: int = 12
    epochs: int = 35
    learning_rate: float = 0.035
    regularization: float = 0.035
    algorithm: str = "mf"


class MatrixFactorization:
    def __init__(self, num_users: int, num_items: int, config: MFConfig, seed: int):
        self.num_users = num_users
        self.num_items = num_items
        self.config = config
        rng = np.random.default_rng(seed)
        scale = 0.1
        self.user_factors = rng.normal(0, scale, size=(num_users, config.rank))
        self.item_factors = rng.normal(0, scale, size=(num_items, config.rank))
        self.implicit_factors = rng.normal(0, scale, size=(num_items, config.rank))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_mean = 3.0
        self._rng = rng
        self._user_histories: list[np.ndarray] | None = None

    def fit(self, ratings: pd.DataFrame) -> "MatrixFactorization":
        if ratings.empty:
            return self
        self._user_histories = self._build_user_histories(ratings)
        if self.config.algorithm.lower() == "svdpp":
            return self._fit_svdpp(ratings)
        return self._fit_mf(ratings)

    def _fit_mf(self, ratings: pd.DataFrame) -> "MatrixFactorization":
        users = ratings["user_id"].to_numpy(dtype=int)
        items = ratings["item_id"].to_numpy(dtype=int)
        values = ratings["rating"].to_numpy(dtype=float)
        self.global_mean = float(values.mean())

        lr = self.config.learning_rate
        reg = self.config.regularization
        order = np.arange(len(values))
        for _ in range(self.config.epochs):
            self._rng.shuffle(order)
            for idx in order:
                u = users[idx]
                i = items[idx]
                r = values[idx]
                pu = self.user_factors[u].copy()
                qi = self.item_factors[i].copy()
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + float(pu @ qi)
                err = r - pred
                self.user_bias[u] += lr * (err - reg * self.user_bias[u])
                self.item_bias[i] += lr * (err - reg * self.item_bias[i])
                self.user_factors[u] += lr * (err * qi - reg * pu)
                self.item_factors[i] += lr * (err * pu - reg * qi)
        return self

    def _fit_svdpp(self, ratings: pd.DataFrame) -> "MatrixFactorization":
        users = ratings["user_id"].to_numpy(dtype=int)
        items = ratings["item_id"].to_numpy(dtype=int)
        values = ratings["rating"].to_numpy(dtype=float)
        self.global_mean = float(values.mean())

        lr = self.config.learning_rate
        reg = self.config.regularization
        order = np.arange(len(values))
        histories = self._user_histories or self._build_user_histories(ratings)
        for _ in range(self.config.epochs):
            self._rng.shuffle(order)
            for idx in order:
                u = users[idx]
                i = items[idx]
                r = values[idx]
                hist = histories[u]
                norm = 1.0 / np.sqrt(len(hist)) if len(hist) else 0.0
                implicit_sum = self.implicit_factors[hist].sum(axis=0) * norm if len(hist) else 0.0
                pu = self.user_factors[u].copy()
                qi = self.item_factors[i].copy()
                effective_user = pu + implicit_sum
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i] + float(effective_user @ qi)
                err = r - pred

                self.user_bias[u] += lr * (err - reg * self.user_bias[u])
                self.item_bias[i] += lr * (err - reg * self.item_bias[i])
                self.user_factors[u] += lr * (err * qi - reg * pu)
                self.item_factors[i] += lr * (err * effective_user - reg * qi)
                if len(hist):
                    self.implicit_factors[hist] += lr * (err * qi * norm - reg * self.implicit_factors[hist])
        return self

    def _build_user_histories(self, ratings: pd.DataFrame) -> list[np.ndarray]:
        histories: list[np.ndarray] = []
        grouped = ratings.groupby("user_id")["item_id"].unique()
        for user_id in range(self.num_users):
            if user_id in grouped.index:
                histories.append(np.asarray(grouped.loc[user_id], dtype=int))
            else:
                histories.append(np.asarray([], dtype=int))
        return histories

    def user_representations(self) -> np.ndarray:
        if self.config.algorithm.lower() != "svdpp" or self._user_histories is None:
            return self.user_factors
        out = self.user_factors.copy()
        for user_id, hist in enumerate(self._user_histories):
            if len(hist):
                out[user_id] += self.implicit_factors[hist].sum(axis=0) / np.sqrt(len(hist))
        return out

    def predict_pairs_raw(
        self,
        users: np.ndarray,
        items: np.ndarray,
        user_factors: np.ndarray | None = None,
    ) -> np.ndarray:
        factors = self.user_representations() if user_factors is None else user_factors
        return (
            self.global_mean
            + self.user_bias[users]
            + self.item_bias[items]
            + np.sum(factors[users] * self.item_factors[items], axis=1)
        )

    def predict_pairs(self, users: np.ndarray, items: np.ndarray, user_factors: np.ndarray | None = None) -> np.ndarray:
        return np.clip(self.predict_pairs_raw(users, items, user_factors=user_factors), 1.0, 5.0)
