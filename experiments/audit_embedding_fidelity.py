"""Audit graph-embedding fidelity options and current empirical evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"


DEPENDENCIES = [
    ("node2vec", "third-party Node2Vec package"),
    ("pecanpy", "fast sparse Node2Vec implementation"),
    ("gensim", "Word2Vec/SGNS backend often used by Node2Vec"),
    ("networkx", "graph utilities; not a Node2Vec trainer by itself"),
    ("karateclub", "graph embedding package with Node2Vec-style models"),
    ("torch", "local SGNS trainer backend"),
    ("torch_geometric", "PyG backend for ELPH/GraphSAGE-style models"),
    ("dgl", "DGL backend for graph neural models"),
    ("ogb", "Open Graph Benchmark utilities often used by ELPH code"),
    ("sklearn", "classical ML utilities"),
]


LOCAL_METHODS = [
    ("spectral", "normalized-adjacency spectral embedding", "confirmed strongest Douban candidate"),
    ("sparse_node2vec_bfs", "dependency-free second-order random walks + PPMI/SVD", "close but below spectral"),
    ("sparse_node2vec_sgns_bfs", "second-order random walks + local torch SGNS", "new fidelity check"),
    ("sparse_node2vec_sgns_dfs", "outward-biased local torch SGNS", "new negative check"),
    ("elph_role", "higher-order role/egonet proxy", "competitive but below spectral"),
]


def dependency_rows() -> list[dict[str, str | bool]]:
    rows = []
    for module, role in DEPENDENCIES:
        rows.append(
            {
                "kind": "dependency",
                "name": module,
                "available": importlib.util.find_spec(module) is not None,
                "role": role,
                "status": "available" if importlib.util.find_spec(module) is not None else "missing",
            }
        )
    return rows


def local_method_rows() -> list[dict[str, str | bool]]:
    rows = []
    summary_path = TABLE_DIR / "douban_svdpp_node2vec_sgns_3seeds_final_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    by_variant = {str(row.variant): row for row in summary.itertuples(index=False)} if not summary.empty else {}
    variant_map = {
        "spectral": "netobs_reliability_spectral",
        "sparse_node2vec_sgns_bfs": "netobs_reliability_node2vec_sgns_bfs",
        "sparse_node2vec_sgns_dfs": "netobs_reliability_node2vec_sgns_dfs",
        "elph_role": "netobs_reliability_elph_role",
    }
    previous_path = TABLE_DIR / "douban_svdpp_node2vec_pq_grid_3seeds_final_summary.csv"
    previous = pd.read_csv(previous_path) if previous_path.exists() else pd.DataFrame()
    previous_rows = {str(row.variant): row for row in previous.itertuples(index=False)} if not previous.empty else {}
    variant_map["sparse_node2vec_bfs"] = "netobs_reliability_node2vec_bfs"
    for method, role, status in LOCAL_METHODS:
        source = by_variant
        variant = variant_map.get(method, "")
        if method == "sparse_node2vec_bfs":
            source = previous_rows
        row = source.get(variant)
        rows.append(
            {
                "kind": "local_method",
                "name": method,
                "available": True,
                "role": role,
                "status": status,
                "variant": variant,
                "reward": "" if row is None else float(row.cumulative_reward),
                "w_rmse": "" if row is None else float(row.w_rmse),
                "entropy_pct": "" if row is None else float(row.selected_avg_entropy_percentile),
                "selected_degree": "" if row is None else float(row.selected_avg_user_degree),
            }
        )
    return rows


def write_report(rows: list[dict[str, object]]) -> None:
    deps = [row for row in rows if row["kind"] == "dependency"]
    methods = [row for row in rows if row["kind"] == "local_method"]
    dep_lines = ["| package | available | role |", "|---|---:|---|"]
    for row in deps:
        dep_lines.append(f"| `{row['name']}` | {row['available']} | {row['role']} |")

    method_lines = ["| method | reward | w-RMSE | entropy pct | selected degree | status |", "|---|---:|---:|---:|---:|---|"]
    for row in sorted(methods, key=lambda item: float(item["w_rmse"]) if item.get("w_rmse") != "" else 99.0):
        method_lines.append(
            f"| `{row['name']}` | {row.get('reward', '')} | {row.get('w_rmse', '')} | "
            f"{row.get('entropy_pct', '')} | {row.get('selected_degree', '')} | {row['status']} |"
        )

    report = f"""# Embedding Fidelity Feasibility Note

This note records the current state of the graph-embedding fidelity gap after
the local SGNS Node2Vec feasibility check.

## Dependency Audit

{chr(10).join(dep_lines)}

## Current Local Implementations

`src/graph_embedding.py` now includes both sparse PPMI/SVD Node2Vec-style
proxies and a local `torch` SGNS trainer:

{chr(10).join(method_lines)}

## Judgment

The environment does not currently provide a validated third-party Node2Vec or
ELPH stack (`node2vec`, `pecanpy`, `gensim`, `torch_geometric`, `dgl`, and `ogb`
are unavailable). `torch` is available, so the reproduction now includes a
dependency-light SGNS Node2Vec path. This closes the most realistic local
implementation branch without adding dependencies.

Empirically, SGNS-BFS is positive versus myopic but still weaker than the current
spectral and ELPH-role candidates on Douban. Therefore it should not be promoted
to 20 seeds.

## Next Decision

Stop local embedding-only sweeps unless a validated external implementation is
explicitly allowed later. The next executable reproduction work should move to
paper-specific benchmark definitions, while keeping `netobs_reliability_spectral`
as the current Douban faithful candidate.

CSV artifact: `outputs/tables/embedding_fidelity_audit.csv`.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "embedding_fidelity_feasibility.md").write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = dependency_rows() + local_method_rows()
    path = TABLE_DIR / "embedding_fidelity_audit.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    write_report(rows)
    print(f"Wrote {path}")
    print(f"Wrote {REPORT_DIR / 'embedding_fidelity_feasibility.md'}")


if __name__ == "__main__":
    sys.exit(main())
