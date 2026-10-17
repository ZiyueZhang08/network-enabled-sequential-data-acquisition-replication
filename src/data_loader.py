from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RATING_COLUMNS = ["user_id", "item_id", "rating"]


@dataclass(frozen=True)
class RatingDataset:
    ratings: pd.DataFrame
    adjacency: np.ndarray
    num_users: int
    num_items: int
    name: str = "synthetic"
    train_ratings: pd.DataFrame | None = None
    test_ratings: pd.DataFrame | None = None
    user_features: np.ndarray | None = None


@dataclass(frozen=True)
class RatingSplit:
    initial: pd.DataFrame
    candidate: pd.DataFrame
    test: pd.DataFrame


def generate_synthetic_dataset(
    seed: int,
    num_users: int = 180,
    num_items: int = 120,
    rank: int = 8,
    num_communities: int = 6,
    observed_fraction: float = 0.18,
    within_edge_prob: float = 0.08,
    across_edge_prob: float = 0.006,
    noise_std: float = 0.35,
) -> RatingDataset:
    """Generate a low-rank rating matrix plus a homophilous user network."""
    rng = np.random.default_rng(seed)
    communities = rng.integers(0, num_communities, size=num_users)
    centers = rng.normal(0, 1.0, size=(num_communities, rank))
    user_latent = centers[communities] + rng.normal(0, 0.45, size=(num_users, rank))
    item_latent = rng.normal(0, 1.0, size=(num_items, rank))

    raw = user_latent @ item_latent.T / np.sqrt(rank)
    raw += rng.normal(0, noise_std, size=raw.shape)
    lo, hi = np.quantile(raw, [0.02, 0.98])
    ratings_matrix = 1.0 + 4.0 * np.clip((raw - lo) / (hi - lo), 0, 1)

    total_pairs = num_users * num_items
    observed_size = int(round(total_pairs * observed_fraction))
    flat_idx = rng.choice(total_pairs, size=observed_size, replace=False)
    users = flat_idx // num_items
    items = flat_idx % num_items
    ratings = pd.DataFrame(
        {
            "user_id": users.astype(int),
            "item_id": items.astype(int),
            "rating": ratings_matrix[users, items].astype(float),
        },
        columns=RATING_COLUMNS,
    )

    adjacency = np.zeros((num_users, num_users), dtype=bool)
    for i in range(num_users):
        same = communities[i] == communities[i + 1 :]
        probs = np.where(same, within_edge_prob, across_edge_prob)
        draws = rng.random(num_users - i - 1) < probs
        if draws.any():
            js = np.nonzero(draws)[0] + i + 1
            adjacency[i, js] = True
            adjacency[js, i] = True

    return RatingDataset(
        ratings=ratings,
        adjacency=adjacency,
        num_users=num_users,
        num_items=num_items,
        name="synthetic-low-rank-network",
    )


def split_ratings(
    ratings: pd.DataFrame,
    seed: int,
    test_ratio: float = 0.2,
    initial_fraction_of_train: float = 0.01,
) -> RatingSplit:
    rng = np.random.default_rng(seed)
    shuffled = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test = shuffled.iloc[:n_test].reset_index(drop=True)
    train = shuffled.iloc[n_test:].reset_index(drop=True)
    n_initial = max(1, int(round(len(train) * initial_fraction_of_train)))
    initial_idx = rng.choice(len(train), size=n_initial, replace=False)
    initial_mask = np.zeros(len(train), dtype=bool)
    initial_mask[initial_idx] = True
    initial = train.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True)
    candidate = train.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True)
    return RatingSplit(initial=initial, candidate=candidate, test=test[RATING_COLUMNS])


def split_dataset(
    dataset: RatingDataset,
    seed: int,
    test_ratio: float = 0.2,
    initial_fraction_of_train: float = 0.01,
    use_predefined_split: bool = True,
) -> RatingSplit:
    if use_predefined_split and dataset.train_ratings is not None and dataset.test_ratings is not None:
        rng = np.random.default_rng(seed)
        train = dataset.train_ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_initial = max(1, int(round(len(train) * initial_fraction_of_train)))
        initial_idx = rng.choice(len(train), size=n_initial, replace=False)
        initial_mask = np.zeros(len(train), dtype=bool)
        initial_mask[initial_idx] = True
        initial = train.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True)
        candidate = train.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True)
        return RatingSplit(initial=initial, candidate=candidate, test=dataset.test_ratings[RATING_COLUMNS])
    return split_ratings(
        dataset.ratings,
        seed=seed,
        test_ratio=test_ratio,
        initial_fraction_of_train=initial_fraction_of_train,
    )


def _symmetrized_topk_adjacency(sim: np.ndarray, top_k: int) -> np.ndarray:
    n = sim.shape[0]
    adjacency = np.zeros((n, n), dtype=bool)
    if n <= 1:
        return adjacency
    k = min(max(1, top_k), n - 1)
    sim = np.asarray(sim, dtype=float).copy()
    np.fill_diagonal(sim, -np.inf)
    for user in range(n):
        neigh = np.argpartition(sim[user], kth=n - k - 1)[-k:]
        adjacency[user, neigh] = True
        adjacency[neigh, user] = True
    np.fill_diagonal(adjacency, False)
    return adjacency


def _movielens_user_features_current(users: pd.DataFrame) -> np.ndarray:
    age = users["age"].to_numpy(dtype=float)
    age = ((age - age.mean()) / max(float(age.std()), 1.0))[:, None]
    gender = pd.get_dummies(users["gender"], dtype=float).to_numpy()
    occupation = pd.get_dummies(users["occupation"], dtype=float).to_numpy()
    features = np.hstack([age, gender, occupation])
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def _movielens_user_features_gcmc(users: pd.DataFrame) -> np.ndarray:
    age_max = max(float(users["age"].max()), 1.0)
    occupations = sorted(users["occupation"].unique().tolist())
    occupation_index = {occupation: idx for idx, occupation in enumerate(occupations, start=2)}
    features = np.zeros((len(users), 2 + len(occupation_index)), dtype=float)
    for row_idx, row in enumerate(users.itertuples(index=False)):
        features[row_idx, 0] = float(row.age) / age_max
        features[row_idx, 1] = 0.0 if row.gender == "M" else 1.0
        features[row_idx, occupation_index[row.occupation]] = 1.0
    return features


def load_movielens_100k(raw_dir: str | Path, network_mode: str = "current") -> RatingDataset:
    """Load local MovieLens 100K files if the user places them under data/raw/ml-100k.

    Expected files:
    - u.data: user item rating timestamp
    - u.user: optional user metadata for a similarity network
    """
    raw_path = Path(raw_dir)
    rating_path = raw_path / "u.data"
    if not rating_path.exists():
        raise FileNotFoundError(f"MovieLens 100K ratings not found: {rating_path}")

    ratings = pd.read_csv(
        rating_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    user_ids = {old: new for new, old in enumerate(sorted(ratings["user_id"].unique()))}
    item_ids = {old: new for new, old in enumerate(sorted(ratings["item_id"].unique()))}
    ratings["user_id"] = ratings["user_id"].map(user_ids).astype(int)
    ratings["item_id"] = ratings["item_id"].map(item_ids).astype(int)
    ratings = ratings[RATING_COLUMNS].reset_index(drop=True)

    num_users = len(user_ids)
    num_items = len(item_ids)
    adjacency = np.zeros((num_users, num_users), dtype=bool)
    user_features: np.ndarray | None = None
    user_path = raw_path / "u.user"
    if user_path.exists():
        users = pd.read_csv(
            user_path,
            sep="|",
            names=["raw_user_id", "age", "gender", "occupation", "zip"],
            engine="python",
        )
        users = users[users["raw_user_id"].isin(user_ids)].copy()
        users["user_id"] = users["raw_user_id"].map(user_ids).astype(int)
        users = users.sort_values("user_id")
        mode = network_mode.lower()
        if mode in {"current", "metadata_cosine", "metadata_l2"}:
            features = _movielens_user_features_current(users)
            sim = features @ features.T
        elif mode in {"gcmc", "gcmc_features", "monti_features"}:
            features = _movielens_user_features_gcmc(users)
            sim = features @ features.T
        elif mode in {"gcmc_cosine", "monti_cosine"}:
            features = _movielens_user_features_gcmc(users)
            features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            sim = features @ features.T
        else:
            raise ValueError(f"Unknown MovieLens network_mode: {network_mode}")
        row_adjacency = _symmetrized_topk_adjacency(sim, top_k=20)
        user_order = users["user_id"].to_numpy(dtype=int)
        adjacency[np.ix_(user_order, user_order)] = row_adjacency
        user_features = np.zeros((num_users, features.shape[1]), dtype=float)
        user_features[user_order] = features

    if not adjacency.any():
        user_items = ratings.groupby("user_id")["item_id"].apply(set)
        for i in range(num_users):
            items_i = user_items.get(i, set())
            if not items_i:
                continue
            sims: list[tuple[float, int]] = []
            for j in range(num_users):
                if i == j:
                    continue
                items_j = user_items.get(j, set())
                union = len(items_i | items_j)
                if union:
                    sims.append((len(items_i & items_j) / union, j))
            for _, j in sorted(sims, reverse=True)[:10]:
                adjacency[i, j] = True
                adjacency[j, i] = True

    return RatingDataset(
        ratings=ratings,
        adjacency=adjacency,
        num_users=num_users,
        num_items=num_items,
        name="movielens-100k",
        user_features=user_features,
    )


def _load_hdf5_mat_matrix(path: Path, field: str) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading MATLAB v7.3/HDF5 .mat files requires h5py. "
            "Install it with `python -m pip install h5py`."
        ) from exc

    with h5py.File(path, "r") as handle:
        if field not in handle:
            raise KeyError(f"Field {field!r} not found in {path}")
        # GCMC's Monti loader transposes dense matrices because MATLAB and
        # Python use different row/column-major conventions.
        return np.asarray(handle[field]).astype(np.float32).T


def _matrix_to_ratings(matrix: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    users, items = np.nonzero(mask)
    return pd.DataFrame(
        {
            "user_id": users.astype(int),
            "item_id": items.astype(int),
            "rating": matrix[users, items].astype(float),
        },
        columns=RATING_COLUMNS,
    )


def load_gcmc_mat(raw_dir: str | Path, name: str) -> RatingDataset:
    """Load GCMC/Monti Flixster or Douban `training_test_dataset.mat`.

    Expected fields are `M`, `Otraining`, `Otest`, and `W_users`.
    Flixster also contains `W_movies`, but this reproduction currently uses
    only the user graph because DU-Net is user-network enabled.
    """
    raw_path = Path(raw_dir)
    mat_path = raw_path / "training_test_dataset.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"GCMC dataset not found: {mat_path}")

    ratings_matrix = _load_hdf5_mat_matrix(mat_path, "M")
    train_mask = _load_hdf5_mat_matrix(mat_path, "Otraining") != 0
    test_mask = _load_hdf5_mat_matrix(mat_path, "Otest") != 0
    user_graph = _load_hdf5_mat_matrix(mat_path, "W_users") != 0

    active_users = np.nonzero(np.any(ratings_matrix != 0, axis=1))[0]
    active_items = np.nonzero(np.any(ratings_matrix != 0, axis=0))[0]
    ratings_matrix = ratings_matrix[np.ix_(active_users, active_items)]
    train_mask = train_mask[np.ix_(active_users, active_items)] & (ratings_matrix != 0)
    test_mask = test_mask[np.ix_(active_users, active_items)] & (ratings_matrix != 0)
    adjacency = user_graph[np.ix_(active_users, active_users)]
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)

    train_ratings = _matrix_to_ratings(ratings_matrix, train_mask)
    test_ratings = _matrix_to_ratings(ratings_matrix, test_mask)
    ratings = pd.concat([train_ratings, test_ratings], ignore_index=True)
    return RatingDataset(
        ratings=ratings,
        adjacency=adjacency,
        num_users=len(active_users),
        num_items=len(active_items),
        name=name,
        train_ratings=train_ratings,
        test_ratings=test_ratings,
    )


def load_dataset_from_config(config: dict) -> RatingDataset:
    dataset_config = config["dataset"]
    name = dataset_config["name"].lower()
    raw_dir = dataset_config["raw_dir"]
    if name == "movielens-100k":
        return load_movielens_100k(raw_dir, network_mode=dataset_config.get("network_mode", "current"))
    if name in {"flixster", "douban"}:
        return load_gcmc_mat(raw_dir, name)
    raise ValueError(f"Unknown dataset: {dataset_config['name']}")
