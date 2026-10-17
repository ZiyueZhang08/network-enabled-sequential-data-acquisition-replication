# Douban Node2Vec SGNS Findings

This note records the graph-representation fidelity branch after local
entropy-gate sweeps were exhausted.

## What Was Added

`graph_embedding_method="sparse_node2vec_sgns_bfs"` and
`graph_embedding_method="sparse_node2vec_sgns_dfs"` now route to a local
second-order Node2Vec-style random-walk trainer with skip-gram negative sampling
when `torch` is available.

This is closer to the original Node2Vec objective than the earlier random-walk
PPMI/SVD proxy, while still avoiding new dependencies. If `torch` is unavailable,
the method falls back to the existing sparse PPMI Node2Vec-style implementation.

## Dependency Feasibility

The local environment has `torch`, `networkx`, and `sklearn`, but lacks the
main external graph-embedding stacks checked for this reproduction:
`node2vec`, `pecanpy`, `gensim`, `karateclub`, `torch_geometric`, `dgl`, and
`ogb`.

Therefore this branch closes the best local no-new-dependency Node2Vec path, but
it is still not a validated third-party Node2Vec package or the paper's true ELPH
module.

## Douban 3-Seed Result

Config: `configs/douban_svdpp_node2vec_sgns_3seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| netobs reliability ELPH-role | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Still the best non-spectral structural proxy. |
| Node2Vec SGNS BFS | 2978.00 | 0.6643 | 0.5623 | 1.2702 | Positive versus myopic, but weaker than spectral/ELPH-role. |
| Node2Vec SGNS DFS | 2949.33 | 0.6676 | 0.5949 | 1.2677 | More information-oriented, weaker accuracy. |
| myopic | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |

## Interpretation

SGNS-BFS behaves sensibly: it improves over myopic and avoids the worst outward
walk behavior. However, it does not beat the existing spectral candidate or the
ELPH-role proxy. SGNS-DFS increases entropy orientation but loses accuracy,
matching the earlier finding that outward exploration is noisy on sparse
Douban.

## Decision

Do not promote SGNS Node2Vec to 20 seeds. Keep it as useful implementation
infrastructure and as evidence that local embedding-only upgrades have been
exhausted under the current no-new-dependency constraint.

Next work should move to paper-specific benchmark definitions unless an external
validated Node2Vec/ELPH dependency is explicitly allowed later.

Primary artifacts:

- `outputs/tables/douban_svdpp_node2vec_sgns_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_node2vec_sgns_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_node2vec_sgns_3seeds_report.md`
- `reports/douban_svdpp_node2vec_sgns_3seeds_mechanism_diagnostics.md`
- `outputs/tables/embedding_fidelity_audit.csv`
