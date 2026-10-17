from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def user_embedding(adjacency: np.ndarray, dim: int = 12, method: str = "spectral", seed: int = 0) -> np.ndarray:
    method = method.lower()
    if method == "node2vec":
        return node2vec_proxy_user_embedding(adjacency, dim=dim, seed=seed)
    if method == "sparse_node2vec":
        return sparse_node2vec_user_embedding(adjacency, dim=dim, seed=seed)
    if method in {"sparse_node2vec_biased", "node2vec_biased"}:
        return sparse_node2vec_biased_user_embedding(adjacency, dim=dim, seed=seed)
    if method in {"sparse_node2vec_bfs", "node2vec_bfs"}:
        return sparse_node2vec_biased_user_embedding(adjacency, dim=dim, seed=seed, return_param=1.0, inout_param=2.0)
    if method in {"sparse_node2vec_sgns", "node2vec_sgns", "sparse_node2vec_sgns_bfs", "node2vec_sgns_bfs"}:
        return sparse_node2vec_sgns_user_embedding(adjacency, dim=dim, seed=seed, return_param=1.0, inout_param=2.0)
    if method in {"sparse_node2vec_sgns_dfs", "node2vec_sgns_dfs"}:
        return sparse_node2vec_sgns_user_embedding(adjacency, dim=dim, seed=seed, return_param=1.0, inout_param=0.5)
    if method in {"sparse_node2vec_dfs", "node2vec_dfs"}:
        return sparse_node2vec_biased_user_embedding(adjacency, dim=dim, seed=seed, return_param=1.0, inout_param=0.5)
    if method in {"sparse_node2vec_return", "node2vec_return"}:
        return sparse_node2vec_biased_user_embedding(adjacency, dim=dim, seed=seed, return_param=0.25, inout_param=1.0)
    if method in {"sparse_node2vec_outward", "node2vec_outward"}:
        return sparse_node2vec_biased_user_embedding(adjacency, dim=dim, seed=seed, return_param=2.0, inout_param=0.25)
    if method == "elph":
        return elph_proxy_user_embedding(adjacency, dim=dim)
    if method == "elph_struct":
        return elph_structural_user_embedding(adjacency, dim=dim)
    if method in {"elph_role", "elph_role_struct"}:
        return elph_role_user_embedding(adjacency, dim=dim, seed=seed)
    return spectral_user_embedding(adjacency, dim=dim)


def spectral_user_embedding(adjacency: np.ndarray, dim: int = 12) -> np.ndarray:
    """Small dependency-free proxy for Node2Vec/ELPH in the MVP."""
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    adj = adjacency.astype(float)
    degree = adj.sum(axis=1)
    norm = np.sqrt(np.maximum(degree, 1.0))
    normalized = adj / norm[:, None] / norm[None, :]
    values, vectors = np.linalg.eigh(normalized)
    order = np.argsort(values)[::-1]
    keep = min(dim, n)
    emb = vectors[:, order[:keep]] * values[order[:keep]]
    if keep < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - keep)))
    degree_feature = (degree / max(float(degree.max()), 1.0))[:, None]
    emb[:, :1] = degree_feature
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def node2vec_proxy_user_embedding(
    adjacency: np.ndarray,
    dim: int = 12,
    seed: int = 0,
    num_walks: int = 6,
    walk_length: int = 12,
    window_size: int = 3,
) -> np.ndarray:
    """Dependency-free DeepWalk/Node2Vec-style proxy using PPMI + SVD.

    This is not a full biased Node2Vec implementation, but it captures the same
    random-walk co-occurrence signal without adding a heavy dependency.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    rng = np.random.default_rng(seed)
    neighbors = [np.flatnonzero(adjacency[i]) for i in range(n)]
    counts = np.zeros((n, n), dtype=float)
    for start in range(n):
        for _ in range(num_walks):
            walk = [start]
            cur = start
            for _step in range(walk_length - 1):
                neigh = neighbors[cur]
                if len(neigh) == 0:
                    break
                cur = int(rng.choice(neigh))
                walk.append(cur)
            for pos, src in enumerate(walk):
                lo = max(0, pos - window_size)
                hi = min(len(walk), pos + window_size + 1)
                for dst in walk[lo:hi]:
                    if dst != src:
                        counts[src, dst] += 1.0
    total = float(counts.sum())
    if total <= 0:
        return spectral_user_embedding(adjacency, dim=dim)
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    ppmi = np.maximum(np.log((counts * total + 1e-12) / np.maximum(row @ col, 1e-12)), 0.0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    keep = min(dim, u.shape[1])
    emb = u[:, :keep] * np.sqrt(s[:keep])
    if keep < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - keep)))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def sparse_node2vec_user_embedding(
    adjacency: np.ndarray,
    dim: int = 12,
    seed: int = 0,
    num_walks: int = 8,
    walk_length: int = 16,
    window_size: int = 4,
) -> np.ndarray:
    """Sparse DeepWalk/Node2Vec-style embedding via random-walk PPMI + sparse SVD."""
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    rng = np.random.default_rng(seed)
    neighbors = [np.flatnonzero(adjacency[i]) for i in range(n)]
    counts: dict[tuple[int, int], float] = {}
    for start in range(n):
        for _ in range(num_walks):
            walk = [start]
            cur = start
            for _step in range(walk_length - 1):
                neigh = neighbors[cur]
                if len(neigh) == 0:
                    break
                cur = int(rng.choice(neigh))
                walk.append(cur)
            for pos, src in enumerate(walk):
                lo = max(0, pos - window_size)
                hi = min(len(walk), pos + window_size + 1)
                for dst in walk[lo:hi]:
                    if dst != src:
                        key = (int(src), int(dst))
                        counts[key] = counts.get(key, 0.0) + 1.0
    if not counts:
        return spectral_user_embedding(adjacency, dim=dim)
    rows = np.fromiter((key[0] for key in counts.keys()), dtype=int)
    cols = np.fromiter((key[1] for key in counts.keys()), dtype=int)
    data = np.fromiter(counts.values(), dtype=float)
    cooc = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    total = float(cooc.sum())
    if total <= 0:
        return spectral_user_embedding(adjacency, dim=dim)
    row_sum = np.asarray(cooc.sum(axis=1)).ravel()
    col_sum = np.asarray(cooc.sum(axis=0)).ravel()
    coo = cooc.tocoo()
    denom = np.maximum(row_sum[coo.row] * col_sum[coo.col], 1e-12)
    ppmi_data = np.maximum(np.log((coo.data * total + 1e-12) / denom), 0.0)
    keep_mask = ppmi_data > 0
    if not np.any(keep_mask):
        return spectral_user_embedding(adjacency, dim=dim)
    ppmi = sparse.csr_matrix(
        (ppmi_data[keep_mask], (coo.row[keep_mask], coo.col[keep_mask])),
        shape=(n, n),
    )
    keep = min(dim, max(1, min(ppmi.shape) - 1))
    try:
        u, s, _ = svds(ppmi, k=keep, which="LM", random_state=seed)
        order = np.argsort(s)[::-1]
        emb = u[:, order] * np.sqrt(s[order])
    except Exception:
        return spectral_user_embedding(adjacency, dim=dim)
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def sparse_node2vec_biased_user_embedding(
    adjacency: np.ndarray,
    dim: int = 12,
    seed: int = 0,
    num_walks: int = 8,
    walk_length: int = 16,
    window_size: int = 4,
    return_param: float = 1.0,
    inout_param: float = 0.5,
) -> np.ndarray:
    """Sparse second-order Node2Vec-style embedding.

    The earlier `sparse_node2vec` is effectively DeepWalk. This variant keeps
    the implementation dependency-free but adds Node2Vec's second-order walk
    bias: returning to the previous node is weighted by `1 / p`, neighbors of
    the previous node by `1`, and outward moves by `1 / q`.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    rng = np.random.default_rng(seed)
    adj_bool = adjacency.astype(bool)
    neighbors = [np.flatnonzero(adj_bool[i]) for i in range(n)]
    counts: dict[tuple[int, int], float] = {}
    p = max(float(return_param), 1e-8)
    q = max(float(inout_param), 1e-8)

    for start in range(n):
        for _ in range(num_walks):
            walk = [start]
            prev = -1
            cur = start
            for _step in range(walk_length - 1):
                neigh = neighbors[cur]
                if len(neigh) == 0:
                    break
                if prev < 0:
                    cur_next = int(rng.choice(neigh))
                else:
                    weights = np.ones(len(neigh), dtype=float)
                    weights[neigh == prev] = 1.0 / p
                    outward = ~adj_bool[prev, neigh]
                    outward[neigh == prev] = False
                    weights[outward] = 1.0 / q
                    probs = weights / weights.sum()
                    cur_next = int(rng.choice(neigh, p=probs))
                prev, cur = cur, cur_next
                walk.append(cur)
            _accumulate_walk_context(walk, counts, window_size)

    return _sparse_ppmi_embedding(counts, n=n, dim=dim, seed=seed, fallback=adjacency)


def sparse_node2vec_sgns_user_embedding(
    adjacency: np.ndarray,
    dim: int = 12,
    seed: int = 0,
    num_walks: int = 4,
    walk_length: int = 12,
    window_size: int = 3,
    return_param: float = 1.0,
    inout_param: float = 2.0,
    epochs: int = 2,
    negative_samples: int = 5,
    batch_size: int = 4096,
    max_pairs: int = 250_000,
) -> np.ndarray:
    """Second-order Node2Vec walks trained with skip-gram negative sampling.

    This is the closest dependency-light local path to the original Node2Vec
    objective. It uses PyTorch when available and falls back to the sparse PPMI
    BFS-style proxy otherwise, keeping the graph-embedding API stable on minimal
    installations.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return sparse_node2vec_biased_user_embedding(
            adjacency,
            dim=dim,
            seed=seed,
            num_walks=max(num_walks, 1),
            walk_length=walk_length,
            window_size=window_size,
            return_param=return_param,
            inout_param=inout_param,
        )

    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    adj_bool = adjacency.astype(bool)
    neighbors = [np.flatnonzero(adj_bool[i]) for i in range(n)]
    rng = np.random.default_rng(seed)
    centers: list[int] = []
    contexts: list[int] = []
    p = max(float(return_param), 1e-8)
    q = max(float(inout_param), 1e-8)

    for start in range(n):
        for _ in range(max(int(num_walks), 1)):
            walk = [start]
            prev = -1
            cur = start
            for _step in range(max(int(walk_length), 1) - 1):
                neigh = neighbors[cur]
                if len(neigh) == 0:
                    break
                if prev < 0:
                    nxt = int(rng.choice(neigh))
                else:
                    weights = np.ones(len(neigh), dtype=float)
                    weights[neigh == prev] = 1.0 / p
                    outward = ~adj_bool[prev, neigh]
                    outward[neigh == prev] = False
                    weights[outward] = 1.0 / q
                    nxt = int(rng.choice(neigh, p=weights / weights.sum()))
                prev, cur = cur, nxt
                walk.append(cur)
            for pos, src in enumerate(walk):
                lo = max(0, pos - window_size)
                hi = min(len(walk), pos + window_size + 1)
                for dst in walk[lo:hi]:
                    if dst != src:
                        centers.append(int(src))
                        contexts.append(int(dst))

    if not centers:
        return spectral_user_embedding(adjacency, dim=dim)

    if len(centers) > max_pairs:
        keep = rng.choice(len(centers), size=max_pairs, replace=False)
        center_np = np.asarray(centers, dtype=np.int64)[keep]
        context_np = np.asarray(contexts, dtype=np.int64)[keep]
    else:
        center_np = np.asarray(centers, dtype=np.int64)
        context_np = np.asarray(contexts, dtype=np.int64)

    degree = adjacency.astype(float).sum(axis=1)
    neg_probs = np.power(np.maximum(degree, 1.0), 0.75)
    neg_probs = neg_probs / max(float(neg_probs.sum()), 1e-12)

    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    center_tensor = torch.as_tensor(center_np, dtype=torch.long)
    context_tensor = torch.as_tensor(context_np, dtype=torch.long)
    neg_prob_tensor = torch.as_tensor(neg_probs, dtype=torch.float32)

    center_emb = torch.nn.Embedding(n, dim)
    context_emb = torch.nn.Embedding(n, dim)
    scale = 0.5 / max(dim, 1)
    torch.nn.init.uniform_(center_emb.weight, -scale, scale)
    torch.nn.init.uniform_(context_emb.weight, -scale, scale)
    optimizer = torch.optim.Adam(
        list(center_emb.parameters()) + list(context_emb.parameters()),
        lr=0.025,
    )

    pair_count = int(center_tensor.numel())
    batch = max(int(batch_size), 1)
    neg_k = max(int(negative_samples), 1)
    for _epoch in range(max(int(epochs), 1)):
        order = torch.randperm(pair_count)
        for start in range(0, pair_count, batch):
            idx = order[start : start + batch]
            src = center_tensor[idx]
            dst = context_tensor[idx]
            src_vec = center_emb(src)
            dst_vec = context_emb(dst)
            pos_logits = torch.sum(src_vec * dst_vec, dim=1)
            neg = torch.multinomial(neg_prob_tensor, src.numel() * neg_k, replacement=True).view(src.numel(), neg_k)
            neg_vec = context_emb(neg)
            neg_logits = torch.bmm(neg_vec, src_vec.unsqueeze(2)).squeeze(2)
            loss = -F.logsigmoid(pos_logits).mean() - F.logsigmoid(-neg_logits).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    emb = center_emb.weight.detach().cpu().numpy() + context_emb.weight.detach().cpu().numpy()
    return _normalize_rows(emb)


def elph_proxy_user_embedding(adjacency: np.ndarray, dim: int = 12) -> np.ndarray:
    """Lightweight structural proxy for ELPH-style higher-order graph features."""
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    adj = adjacency.astype(float)
    degree = adj.sum(axis=1)
    deg_norm = degree / max(float(degree.max()), 1.0)
    two_hop = adj @ adj
    np.fill_diagonal(two_hop, 0.0)
    two_hop_count = two_hop.sum(axis=1)
    triangles = np.sum(adj * two_hop, axis=1) / 2.0
    possible = np.maximum(degree * (degree - 1.0) / 2.0, 1.0)
    clustering = triangles / possible
    structural = np.column_stack(
        [
            deg_norm,
            np.log1p(degree) / np.log1p(max(float(degree.max()), 1.0)),
            two_hop_count / max(float(two_hop_count.max()), 1.0),
            clustering,
        ]
    )
    spectral = spectral_user_embedding(adjacency, dim=max(dim - structural.shape[1], 1))
    emb = np.hstack([structural, spectral])
    if emb.shape[1] > dim:
        emb = emb[:, :dim]
    elif emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def elph_structural_user_embedding(adjacency: np.ndarray, dim: int = 12) -> np.ndarray:
    """Stronger ELPH-style structural features plus sparse random-walk context."""
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    adj = adjacency.astype(float)
    degree = adj.sum(axis=1)
    two_hop = adj @ adj
    np.fill_diagonal(two_hop, 0.0)
    two_hop_binary = two_hop > 0
    triangles = np.sum(adj * two_hop, axis=1) / 2.0
    possible = np.maximum(degree * (degree - 1.0) / 2.0, 1.0)
    neighbor_degree_sum = adj @ degree
    neighbor_degree_mean = neighbor_degree_sum / np.maximum(degree, 1.0)
    structural = np.column_stack(
        [
            degree,
            np.log1p(degree),
            two_hop_binary.sum(axis=1),
            np.log1p(two_hop.sum(axis=1)),
            triangles / possible,
            neighbor_degree_mean,
        ]
    )
    denom = np.maximum(np.std(structural, axis=0, keepdims=True), 1e-12)
    structural = (structural - np.mean(structural, axis=0, keepdims=True)) / denom
    walk_dim = max(dim - structural.shape[1], 1)
    walk = sparse_node2vec_user_embedding(adjacency, dim=walk_dim, seed=0, num_walks=4, walk_length=10)
    emb = np.hstack([structural, walk])
    if emb.shape[1] > dim:
        emb = emb[:, :dim]
    elif emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def elph_role_user_embedding(adjacency: np.ndarray, dim: int = 12, seed: int = 0) -> np.ndarray:
    """Role-oriented ELPH-style structural embedding.

    This emphasizes local structural equivalence rather than proximity alone:
    egonet size/density, neighbor-degree distribution, two-hop reach, triangle
    density, and a light biased-walk context are compressed with SVD.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.empty((0, dim))
    adj = adjacency.astype(float)
    adj_bool = adjacency.astype(bool)
    degree = adj.sum(axis=1)
    two_hop = adj @ adj
    np.fill_diagonal(two_hop, 0.0)
    two_hop_binary = two_hop > 0
    triangles = np.sum(adj * two_hop, axis=1) / 2.0
    possible_triangles = np.maximum(degree * (degree - 1.0) / 2.0, 1.0)
    clustering = triangles / possible_triangles

    neighbor_degree_sum = adj @ degree
    neighbor_degree_mean = neighbor_degree_sum / np.maximum(degree, 1.0)
    neighbor_degree_sq = adj @ (degree * degree)
    neighbor_degree_var = neighbor_degree_sq / np.maximum(degree, 1.0) - neighbor_degree_mean**2
    neighbor_degree_std = np.sqrt(np.maximum(neighbor_degree_var, 0.0))

    egonet_edges = np.zeros(n, dtype=float)
    egonet_size = degree + 1.0
    for node in range(n):
        ego_nodes = np.flatnonzero(adj_bool[node])
        if len(ego_nodes) > 1:
            egonet_edges[node] = float(np.triu(adj_bool[np.ix_(ego_nodes, ego_nodes)], k=1).sum())
    egonet_density = egonet_edges / np.maximum(degree * (degree - 1.0) / 2.0, 1.0)

    structural = np.column_stack(
        [
            degree,
            np.log1p(degree),
            two_hop_binary.sum(axis=1),
            np.log1p(two_hop.sum(axis=1)),
            clustering,
            neighbor_degree_mean,
            neighbor_degree_std,
            egonet_size,
            egonet_edges,
            egonet_density,
        ]
    )
    structural = _standardize_columns(structural)
    role_dim = min(max(dim, 2), structural.shape[1], n)
    try:
        u, s, _ = np.linalg.svd(structural, full_matrices=False)
        role = u[:, :role_dim] * s[:role_dim]
    except Exception:
        role = structural[:, :role_dim]

    walk_dim = max(dim - min(role.shape[1], max(dim // 2, 1)), 1)
    walk = sparse_node2vec_biased_user_embedding(
        adjacency,
        dim=walk_dim,
        seed=seed,
        num_walks=4,
        walk_length=10,
        window_size=3,
    )
    keep_role = max(dim - walk.shape[1], 1)
    emb = np.hstack([role[:, :keep_role], walk])
    if emb.shape[1] > dim:
        emb = emb[:, :dim]
    elif emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    return _normalize_rows(emb)


def _accumulate_walk_context(walk: list[int], counts: dict[tuple[int, int], float], window_size: int) -> None:
    for pos, src in enumerate(walk):
        lo = max(0, pos - window_size)
        hi = min(len(walk), pos + window_size + 1)
        for dst in walk[lo:hi]:
            if dst != src:
                key = (int(src), int(dst))
                counts[key] = counts.get(key, 0.0) + 1.0


def _sparse_ppmi_embedding(
    counts: dict[tuple[int, int], float],
    n: int,
    dim: int,
    seed: int,
    fallback: np.ndarray,
) -> np.ndarray:
    if not counts:
        return spectral_user_embedding(fallback, dim=dim)
    rows = np.fromiter((key[0] for key in counts.keys()), dtype=int)
    cols = np.fromiter((key[1] for key in counts.keys()), dtype=int)
    data = np.fromiter(counts.values(), dtype=float)
    cooc = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    total = float(cooc.sum())
    if total <= 0:
        return spectral_user_embedding(fallback, dim=dim)
    row_sum = np.asarray(cooc.sum(axis=1)).ravel()
    col_sum = np.asarray(cooc.sum(axis=0)).ravel()
    coo = cooc.tocoo()
    denom = np.maximum(row_sum[coo.row] * col_sum[coo.col], 1e-12)
    ppmi_data = np.maximum(np.log((coo.data * total + 1e-12) / denom), 0.0)
    keep_mask = ppmi_data > 0
    if not np.any(keep_mask):
        return spectral_user_embedding(fallback, dim=dim)
    ppmi = sparse.csr_matrix(
        (ppmi_data[keep_mask], (coo.row[keep_mask], coo.col[keep_mask])),
        shape=(n, n),
    )
    keep = min(dim, max(1, min(ppmi.shape) - 1))
    try:
        u, s, _ = svds(ppmi, k=keep, which="LM", random_state=seed)
        order = np.argsort(s)[::-1]
        emb = u[:, order] * np.sqrt(s[order])
    except Exception:
        return spectral_user_embedding(fallback, dim=dim)
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    return _normalize_rows(emb)


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    std = np.maximum(np.std(values, axis=0, keepdims=True), 1e-12)
    return (values - np.mean(values, axis=0, keepdims=True)) / std


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)
