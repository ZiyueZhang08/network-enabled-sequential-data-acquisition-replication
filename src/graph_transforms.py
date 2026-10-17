from __future__ import annotations

import numpy as np


def _edge_list(adjacency: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices_from(adjacency, k=1)
    mask = adjacency[rows, cols].astype(bool)
    return np.column_stack([rows[mask], cols[mask]])


def edge_dropout(adjacency: np.ndarray, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graph = np.asarray(adjacency, dtype=bool).copy()
    edges = _edge_list(graph)
    if len(edges) == 0:
        return graph
    keep = rng.random(len(edges)) >= float(np.clip(rate, 0.0, 1.0))
    dropped = edges[~keep]
    graph[dropped[:, 0], dropped[:, 1]] = False
    graph[dropped[:, 1], dropped[:, 0]] = False
    np.fill_diagonal(graph, False)
    return graph


def edge_rewiring(adjacency: np.ndarray, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graph = np.asarray(adjacency, dtype=bool).copy()
    n = graph.shape[0]
    edges = _edge_list(graph)
    if len(edges) == 0 or n < 2:
        return graph
    n_rewire = int(round(len(edges) * float(np.clip(rate, 0.0, 1.0))))
    if n_rewire <= 0:
        return graph
    remove_idx = rng.choice(len(edges), size=min(n_rewire, len(edges)), replace=False)
    removed = edges[remove_idx]
    graph[removed[:, 0], removed[:, 1]] = False
    graph[removed[:, 1], removed[:, 0]] = False
    added = 0
    attempts = 0
    max_attempts = max(1000, n_rewire * 50)
    while added < len(removed) and attempts < max_attempts:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        attempts += 1
        if graph[i, j]:
            continue
        graph[i, j] = True
        graph[j, i] = True
        added += 1
    np.fill_diagonal(graph, False)
    return graph


def two_hop_densification(adjacency: np.ndarray, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graph = np.asarray(adjacency, dtype=bool).copy()
    n = graph.shape[0]
    edges = _edge_list(graph)
    if len(edges) == 0 or n < 2:
        return graph
    n_add = int(round(len(edges) * max(float(rate), 0.0)))
    if n_add <= 0:
        return graph
    common = graph.astype(np.int16) @ graph.astype(np.int16)
    np.fill_diagonal(common, 0)
    rows, cols = np.triu_indices(n, k=1)
    possible = (~graph[rows, cols]) & (common[rows, cols] > 0)
    candidates = np.column_stack([rows[possible], cols[possible]])
    if len(candidates) == 0:
        return graph
    weights = common[candidates[:, 0], candidates[:, 1]].astype(float)
    jitter = rng.random(len(candidates)) * 1e-6
    order = np.argsort(weights + jitter)[::-1][: min(n_add, len(candidates))]
    chosen = candidates[order]
    graph[chosen[:, 0], chosen[:, 1]] = True
    graph[chosen[:, 1], chosen[:, 0]] = True
    np.fill_diagonal(graph, False)
    return graph


def transform_user_graph(adjacency: np.ndarray, mode: str = "none", rate: float = 0.0, seed: int = 0) -> np.ndarray:
    mode = mode.lower()
    if mode in {"none", "original"}:
        return np.asarray(adjacency, dtype=bool).copy()
    if mode == "edge_dropout":
        return edge_dropout(adjacency, rate=rate, seed=seed)
    if mode == "rewiring":
        return edge_rewiring(adjacency, rate=rate, seed=seed)
    if mode == "densification":
        return two_hop_densification(adjacency, rate=rate, seed=seed)
    raise ValueError(f"Unknown graph transform mode: {mode}")
