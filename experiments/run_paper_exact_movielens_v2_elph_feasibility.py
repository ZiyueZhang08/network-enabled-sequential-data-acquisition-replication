from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_paper_faithful_movielens import make_u1_fallback_split
from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import load_movielens_100k
from src.graph_embedding import user_embedding
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir


BASE_PREFIX = "paper_exact_attempt_movielens_v2"
PREFIX = f"{BASE_PREFIX}_elph_feasibility"
CONFIG_PATH = ROOT / "configs" / f"{BASE_PREFIX}.json"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

DEPENDENCY_PATH = TABLE_DIR / f"{PREFIX}_dependencies.csv"
GRAPH_PATH = TABLE_DIR / f"{PREFIX}_movielens_graph.csv"
EMBEDDING_PATH = TABLE_DIR / f"{PREFIX}_embedding_distance.csv"
METRICS_PATH = TABLE_DIR / f"{PREFIX}_metrics.csv"
SUMMARY_PATH = TABLE_DIR / f"{PREFIX}.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}.md"

MAINLINE_VARIANT = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
PROTOCOL = "movielens_u1_fallback_table1_density"
STATIC_METHODS = ("elph_role", "elph_struct")
DEFAULT_DIAGNOSTIC_METHODS = ("elph_role",)

DEPENDENCIES = [
    ("elph", "direct import name for an official ELPH candidate", "official_candidate"),
    ("torch_geometric", "PyG backend often required by graph-neural ELPH implementations", "supporting_stack"),
    ("dgl", "DGL backend candidate for graph-neural implementations", "supporting_stack"),
    ("ogb", "Open Graph Benchmark utilities often used by ELPH reference code", "supporting_stack"),
    ("torch", "local neural backend", "supporting_stack"),
    ("networkx", "local graph utility backend", "supporting_stack"),
    ("sklearn", "local numeric/model-selection utility backend", "supporting_stack"),
    ("node2vec", "third-party Node2Vec graph-embedding package", "proxy_dependency"),
    ("pecanpy", "third-party fast Node2Vec graph-embedding package", "proxy_dependency"),
    ("gensim", "Word2Vec/SGNS backend used by many Node2Vec recipes", "proxy_dependency"),
]

METRIC_COLUMNS = [
    "pdf_literal_norm_rmse",
    "standard_mask_rmse",
    "cumulative_reward",
    "centered_cumulative_reward",
    "reward_tail_proxy_rmse",
    "rmse",
    "w_rmse",
    "observed_size",
    "candidate_size",
]


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            view[col] = view[col].fillna("").astype(str)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def build_dependency_audit() -> pd.DataFrame:
    rows = []
    for module_name, role, kind in DEPENDENCIES:
        available = package_available(module_name)
        if module_name == "elph":
            exactness_decision = "candidate_only_if_author_backend_validated" if available else "missing"
        elif kind == "supporting_stack":
            exactness_decision = "supporting_stack_only"
        else:
            exactness_decision = "proxy_only"
        rows.append(
            {
                "module": module_name,
                "available": bool(available),
                "kind": kind,
                "role": role,
                "install_attempted": False,
                "install_decision": "not_attempted_without_author_source_or_dependency_lock",
                "exactness_decision": exactness_decision,
            }
        )
    return pd.DataFrame(rows)


def pairwise_dist_sq(embedding: np.ndarray) -> np.ndarray:
    emb = np.asarray(embedding, dtype=float)
    diff = emb[:, None, :] - emb[None, :, :]
    return np.sum(diff * diff, axis=2)


def topk_excluding_self(embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    dist_sq = pairwise_dist_sq(embedding)
    np.fill_diagonal(dist_sq, np.inf)
    n = dist_sq.shape[0]
    k = min(max(1, int(top_k)), max(n - 1, 1))
    idx = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(dist_sq, idx, axis=1)
    order = np.argsort(dist, axis=1)
    sorted_idx = np.take_along_axis(idx, order, axis=1)
    sorted_dist = np.take_along_axis(dist, order, axis=1)
    return sorted_idx, sorted_dist


def mean_neighbor_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    overlaps = []
    jaccards = []
    for a, b in zip(left, right):
        aset = set(int(x) for x in a)
        bset = set(int(x) for x in b)
        inter = len(aset & bset)
        union = len(aset | bset)
        overlaps.append(inter / max(len(aset), 1))
        jaccards.append(inter / max(union, 1))
    return float(np.mean(overlaps)), float(np.mean(jaccards))


def distance_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_dist = pairwise_dist_sq(left)
    right_dist = pairwise_dist_sq(right)
    tri = np.triu_indices(left_dist.shape[0], k=1)
    a = left_dist[tri]
    b = right_dist[tri]
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def graph_distance_summary(embedding: np.ndarray, adjacency: np.ndarray, seed: int = 0) -> tuple[float, float]:
    dist_sq = pairwise_dist_sq(embedding)
    tri = np.triu_indices(dist_sq.shape[0], k=1)
    edge_mask = adjacency[tri].astype(bool)
    edge_dist = dist_sq[tri][edge_mask]
    nonedge_dist = dist_sq[tri][~edge_mask]
    if len(nonedge_dist) > 100_000:
        rng = np.random.default_rng(seed)
        nonedge_dist = rng.choice(nonedge_dist, size=100_000, replace=False)
    edge_mean = float(np.mean(edge_dist)) if len(edge_dist) else float("nan")
    nonedge_mean = float(np.mean(nonedge_dist)) if len(nonedge_dist) else float("nan")
    return edge_mean, nonedge_mean


def effective_neighborhood_full(embedding: np.ndarray, sigma: float) -> np.ndarray:
    dist_sq = pairwise_dist_sq(embedding)
    denom = max(2.0 * float(sigma) * float(sigma), 1e-12)
    logits = -dist_sq / denom
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return 1.0 / np.maximum(np.sum(weights * weights, axis=1), 1e-12)


def build_graph_audit(dataset, config: dict) -> pd.DataFrame:
    adjacency = np.asarray(dataset.adjacency, dtype=bool)
    degree = adjacency.sum(axis=1).astype(float)
    user_features = np.asarray(dataset.user_features, dtype=float)
    edge_count = float(degree.sum() / 2.0)
    possible_edges = dataset.num_users * (dataset.num_users - 1) / 2.0
    feature_norm = np.linalg.norm(user_features, axis=1)
    return pd.DataFrame(
        [
            {
                "dataset": dataset.name,
                "network_mode": config["dataset"].get("network_mode"),
                "network_embedding_mode": config["dataset"].get("network_embedding_mode"),
                "num_users": int(dataset.num_users),
                "raw_loaded_num_items": int(dataset.num_items),
                "protocol_num_items": int(config["split"].get("num_items", dataset.num_items)),
                "user_feature_dim": int(user_features.shape[1]),
                "graph_builder": "symmetrized_top20_similarity_over_gcmc_user_metadata_features",
                "edge_count_undirected": edge_count,
                "density": edge_count / max(possible_edges, 1.0),
                "degree_min": float(degree.min()),
                "degree_mean": float(degree.mean()),
                "degree_median": float(np.median(degree)),
                "degree_max": float(degree.max()),
                "feature_norm_min": float(feature_norm.min()),
                "feature_norm_mean": float(feature_norm.mean()),
                "feature_norm_max": float(feature_norm.max()),
            }
        ]
    )


def build_embedding_audit(dataset, config: dict) -> pd.DataFrame:
    adjacency = np.asarray(dataset.adjacency, dtype=bool)
    user_features = np.asarray(dataset.user_features, dtype=float)
    top_k = int(config["acquisition"]["dls_top_k"])
    rank = int(config["mf"]["rank"])
    sigma = float(config["acquisition"]["sigma"])

    embeddings: dict[str, tuple[str, np.ndarray, str]] = {
        "gcmc_user_features": (
            "v2_metadata_user_features",
            user_features,
            "current_v2_input_not_official_elph",
        )
    }
    for method in STATIC_METHODS:
        embeddings[method] = (
            f"local_{method}_proxy",
            user_embedding(adjacency, dim=rank, method=method, seed=0),
            "local_proxy_not_official_elph",
        )

    metadata_idx, _metadata_dist = topk_excluding_self(user_features, top_k=top_k)
    rows = []
    for method, (source, embedding, exactness) in embeddings.items():
        emb = np.asarray(embedding, dtype=float)
        idx, dist = topk_excluding_self(emb, top_k=top_k)
        overlap, jaccard = (1.0, 1.0) if method == "gcmc_user_features" else mean_neighbor_overlap(metadata_idx, idx)
        corr = 1.0 if method == "gcmc_user_features" else distance_correlation(user_features, emb)
        graph_hits = []
        for user, neighbors in enumerate(idx):
            graph_hits.append(float(np.mean(adjacency[user, neighbors])))
        edge_dist, nonedge_dist = graph_distance_summary(emb, adjacency)
        effective = effective_neighborhood_full(emb, sigma=sigma)
        norms = np.linalg.norm(emb, axis=1)
        rows.append(
            {
                "embedding_method": method,
                "embedding_source": source,
                "exactness_decision": exactness,
                "shape": f"{emb.shape[0]}x{emb.shape[1]}",
                "finite_share": float(np.isfinite(emb).mean()),
                "norm_min": float(norms.min()),
                "norm_mean": float(norms.mean()),
                "norm_max": float(norms.max()),
                "topk": int(top_k),
                "topk_overlap_with_metadata": float(overlap),
                "topk_jaccard_with_metadata": float(jaccard),
                "pairwise_dist_corr_with_metadata": float(corr),
                "graph_edge_hit_rate_in_embedding_topk": float(np.mean(graph_hits)),
                "embedding_topk_dist_sq_mean": float(np.mean(dist[np.isfinite(dist)])),
                "graph_edge_dist_sq_mean": float(edge_dist),
                "graph_nonedge_dist_sq_mean_sample": float(nonedge_dist),
                "full_dls_effective_n_mean_sigma": float(np.mean(effective)),
                "full_dls_effective_n_median_sigma": float(np.median(effective)),
            }
        )
    return pd.DataFrame(rows)


def completed_runs(existing: pd.DataFrame) -> set[tuple[str, int]]:
    if existing.empty:
        return set()
    return {
        (str(row.variant), int(row.seed))
        for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
    }


def append_metrics(metrics: pd.DataFrame, wrote_header: bool) -> bool:
    metrics.to_csv(METRICS_PATH, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
    return True


def run_proxy_diagnostic(config: dict, dataset, seeds: list[int], methods: list[str], resume: bool) -> pd.DataFrame:
    raw_dir = ROOT / config["dataset"]["raw_dir"]
    split_cfg = config["split"]
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    all_metrics: list[pd.DataFrame] = []
    existing = pd.read_csv(METRICS_PATH) if resume and METRICS_PATH.exists() else pd.DataFrame()
    if not existing.empty:
        all_metrics.append(existing)
    done = completed_runs(existing)
    wrote_header = bool(METRICS_PATH.exists() and not existing.empty)

    for seed in seeds:
        split = make_u1_fallback_split(
            raw_dir,
            seed=int(seed),
            initial_size=int(split_cfg.get("initial_size", 1364)),
            num_items=int(split_cfg.get("num_items", dataset.num_items)),
        )
        for method in methods:
            label = f"{MAINLINE_VARIANT}__{method}_proxy"
            if (label, int(seed)) in done:
                continue
            acquisition = replace(
                base_acquisition,
                selection="exp",
                selection_temperature=1.0,
                graph_embedding_method=str(method),
            )
            metrics = run_policy(
                policy="paper_faithful_dunet",
                split=split,
                adjacency=dataset.adjacency,
                num_users=int(split_cfg.get("num_users", dataset.num_users)),
                num_items=int(split_cfg.get("num_items", dataset.num_items)),
                mf_config=mf_config,
                acquisition_config=acquisition,
                seed=int(seed),
                network_embedding=None,
            )
            metrics.insert(0, "protocol", PROTOCOL)
            metrics.insert(1, "variant", label)
            metrics.insert(2, "seed", int(seed))
            metrics.insert(3, "embedding_method", str(method))
            metrics.insert(4, "embedding_exactness", "local_proxy_not_official_elph")
            metrics.insert(5, "initial_size", len(split.initial))
            metrics.insert(6, "test_size", len(split.test))
            all_metrics.append(metrics)
            wrote_header = append_metrics(metrics, wrote_header)

    return pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()


def summarize_final_rows(final: pd.DataFrame) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["variant", "policy", "embedding_method", "embedding_exactness"]
    for keys, group in final.groupby(group_cols, sort=False):
        variant, policy, embedding_method, embedding_exactness = keys
        row: dict[str, object] = {
            "variant": str(variant),
            "policy": str(policy),
            "embedding_method": str(embedding_method),
            "embedding_exactness": str(embedding_exactness),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            if metric not in group.columns:
                row[metric] = np.nan
                row[f"{metric}_ci95"] = np.nan
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[metric] = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_final_rows(seeds: list[int]) -> pd.DataFrame:
    path = TABLE_DIR / f"{BASE_PREFIX}_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(path)
    metrics = metrics[
        (metrics["variant"].astype(str) == MAINLINE_VARIANT)
        & (metrics["seed"].astype(int).isin([int(seed) for seed in seeds]))
    ].copy()
    if metrics.empty:
        return pd.DataFrame()
    final = metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1).copy()
    final["variant"] = f"{MAINLINE_VARIANT}__gcmc_user_features_baseline"
    final["embedding_method"] = "gcmc_user_features"
    final["embedding_exactness"] = "current_v2_mainline_metadata_not_official_elph"
    return final


def build_summary(proxy_metrics: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    finals = []
    baseline = baseline_final_rows(seeds)
    if not baseline.empty:
        finals.append(baseline)
    if not proxy_metrics.empty:
        proxy_final = proxy_metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
        finals.append(proxy_final)
    if not finals:
        return pd.DataFrame()

    final = pd.concat(finals, ignore_index=True, sort=False)
    summary = summarize_final_rows(final)
    if summary.empty:
        return summary
    baseline_summary = summary[summary["embedding_method"] == "gcmc_user_features"]
    if baseline_summary.empty:
        for metric in ["standard_mask_rmse", "cumulative_reward", "reward_tail_proxy_rmse", "w_rmse"]:
            summary[f"delta_{metric}_vs_metadata"] = np.nan
        return summary

    base = baseline_summary.iloc[0]
    for metric in ["standard_mask_rmse", "cumulative_reward", "reward_tail_proxy_rmse", "w_rmse"]:
        summary[f"delta_{metric}_vs_metadata"] = pd.to_numeric(summary[metric], errors="coerce") - float(base[metric])
    summary["exact_elph_reconstructed"] = "no"
    summary["diagnostic_decision"] = np.where(
        summary["embedding_method"].eq("gcmc_user_features"),
        "baseline_current_v2",
        "proxy_diagnostic_only_not_promoted",
    )
    return summary


def write_report(
    dependencies: pd.DataFrame,
    graph: pd.DataFrame,
    embeddings: pd.DataFrame,
    summary: pd.DataFrame,
    seeds: list[int],
    methods: list[str],
) -> None:
    official_available = bool(
        dependencies[
            (dependencies["module"] == "elph") & (dependencies["available"].astype(bool))
        ].shape[0]
    )
    graph_row = graph.iloc[0] if not graph.empty else None
    proxy_rows = summary[summary["embedding_method"].isin(methods)].copy() if not summary.empty else pd.DataFrame()
    if not proxy_rows.empty:
        best_proxy = proxy_rows.sort_values("standard_mask_rmse").iloc[0]
        diagnostic_sentence = (
            f"Best proxy by standard RMSE is `{best_proxy['embedding_method']}` with "
            f"standard RMSE {float(best_proxy['standard_mask_rmse']):.6g}, "
            f"reward {float(best_proxy['cumulative_reward']):.6g}, and "
            f"reward-tail RMSE {float(best_proxy['reward_tail_proxy_rmse']):.6g}."
        )
    else:
        diagnostic_sentence = "No proxy diagnostic rows were run."

    selected_embedding_cols = [
        "embedding_method",
        "embedding_source",
        "topk_overlap_with_metadata",
        "pairwise_dist_corr_with_metadata",
        "graph_edge_hit_rate_in_embedding_topk",
        "full_dls_effective_n_mean_sigma",
        "exactness_decision",
    ]
    selected_summary_cols = [
        "variant",
        "embedding_method",
        "seed_count",
        "standard_mask_rmse",
        "delta_standard_mask_rmse_vs_metadata",
        "cumulative_reward",
        "delta_cumulative_reward_vs_metadata",
        "reward_tail_proxy_rmse",
        "delta_reward_tail_proxy_rmse_vs_metadata",
        "exact_elph_reconstructed",
        "diagnostic_decision",
    ]

    lines = [
        "# MovieLens v2 ELPH/DLS Feasibility Audit",
        "",
        f"Generated by `experiments/run_paper_exact_movielens_v2_elph_feasibility.py` for seeds `{seeds}` and diagnostic methods `{methods}`.",
        "",
        "## Decision",
        "",
        "- Status: `source_blocked_proxy_diagnostic_complete`.",
        f"- Direct import of an `elph` module available locally: `{official_available}`.",
        "- Current v2 mainline uses GCMC-style MovieLens user metadata features as the DLS embedding input.",
        "- Local `elph_role` and `elph_struct` are dependency-free ELPH-style proxies from `src/graph_embedding.py`; they are not a recovered paper backend.",
        f"- {diagnostic_sentence}",
        "- Do not switch the exact-attempt mainline to a proxy embedding unless author source, supplement, or a validated dependency lock recovers the paper ELPH backend and parameters.",
        "",
        "## Plan Questions",
        "",
        "1. ELPH/official backend installability: no official paper ELPH backend is importable locally; no install was attempted without author source or a dependency lock.",
        "2. MovieLens input graph: the graph is a symmetrized top-20 similarity graph over GCMC-style user metadata features.",
        "3. User embedding generation: current metadata features and local proxy embeddings are finite and generated.",
        "4. Distance from metadata embedding: see the embedding-distance table below.",
        "5. Three-seed diagnostic: see the summary table below.",
        "",
        "## MovieLens Graph",
        "",
        md_table(graph if graph_row is not None else pd.DataFrame()),
        "",
        "## Dependency Audit",
        "",
        md_table(dependencies),
        "",
        "## Embedding Distance Audit",
        "",
        md_table(embeddings[selected_embedding_cols] if not embeddings.empty else pd.DataFrame()),
        "",
        "## Three-Seed Diagnostic",
        "",
        md_table(summary[selected_summary_cols] if not summary.empty else pd.DataFrame()),
        "",
        "## Artifacts",
        "",
        f"- `{DEPENDENCY_PATH.relative_to(ROOT)}`",
        f"- `{GRAPH_PATH.relative_to(ROOT)}`",
        f"- `{EMBEDDING_PATH.relative_to(ROOT)}`",
        f"- `{METRICS_PATH.relative_to(ROOT)}`",
        f"- `{SUMMARY_PATH.relative_to(ROOT)}`",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_DIAGNOSTIC_METHODS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_movielens_100k(
        ROOT / config["dataset"]["raw_dir"],
        network_mode=config["dataset"].get("network_mode", "gcmc_features"),
    )
    if dataset.user_features is None:
        raise ValueError("MovieLens dataset did not expose GCMC user features.")

    dependencies = build_dependency_audit()
    graph = build_graph_audit(dataset, config)
    embeddings = build_embedding_audit(dataset, config)
    dependencies.to_csv(DEPENDENCY_PATH, index=False)
    graph.to_csv(GRAPH_PATH, index=False)
    embeddings.to_csv(EMBEDDING_PATH, index=False)

    proxy_metrics = pd.read_csv(METRICS_PATH) if args.resume and METRICS_PATH.exists() else pd.DataFrame()
    if not args.skip_diagnostic:
        proxy_metrics = run_proxy_diagnostic(
            config=config,
            dataset=dataset,
            seeds=[int(seed) for seed in args.seeds],
            methods=[str(method) for method in args.methods],
            resume=bool(args.resume),
        )
    summary = build_summary(proxy_metrics, seeds=[int(seed) for seed in args.seeds])
    summary.to_csv(SUMMARY_PATH, index=False)
    write_report(
        dependencies=dependencies,
        graph=graph,
        embeddings=embeddings,
        summary=summary,
        seeds=[int(seed) for seed in args.seeds],
        methods=[str(method) for method in args.methods],
    )
    print(REPORT_PATH)
    print(DEPENDENCY_PATH)
    print(GRAPH_PATH)
    print(EMBEDDING_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
