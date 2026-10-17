# Douban Node2Vec p/q Grid Findings

This note records the next graph-embedding fidelity check after the first
`sparse_node2vec_biased` and `elph_role` pilot.

## Implementation Added

- `graph_embedding_method=sparse_node2vec_bfs`: second-order Node2Vec-style walks with `p=1.0`, `q=2.0`, emphasizing local BFS-like neighborhoods.
- `graph_embedding_method=sparse_node2vec_dfs`: second-order walks with `p=1.0`, `q=0.5`, matching the previous outward-biased default.
- `graph_embedding_method=sparse_node2vec_return`: second-order walks with `p=0.25`, `q=1.0`, emphasizing return/local revisits.
- `graph_embedding_method=sparse_node2vec_outward`: second-order walks with `p=2.0`, `q=0.25`, strongly emphasizing outward exploration.

These remain dependency-free sparse Node2Vec-style implementations, not a
third-party exact Node2Vec package.

## Douban 3-Seed Result

Config: `configs/douban_svdpp_node2vec_pq_grid_3seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Still the best current embedding choice. |
| netobs reliability Node2Vec BFS | 2983.33 | 0.6633 | 0.5717 | 1.4160 | Closest Node2Vec setting; more information-oriented but more centrality-biased. |
| netobs reliability ELPH-role | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Still competitive with BFS-style Node2Vec. |
| netobs reliability Node2Vec return | 2968.67 | 0.6657 | 0.5641 | 1.2583 | Improves over myopic but not enough to promote. |
| netobs reliability Node2Vec DFS | 2965.33 | 0.6661 | 0.5681 | 1.2063 | Same behavior class as the earlier biased Node2Vec pilot. |
| myopic | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |
| netobs reliability Node2Vec outward | 2914.33 | 0.6718 | 0.5669 | 1.2662 | Negative: outward-biased walks underperform myopic on w-RMSE. |

## Interpretation

The best Node2Vec p/q choice for sparse Douban is BFS-like local structure, not
outward exploration. That is consistent with the sparse-graph diagnosis: overly
aggressive outward walks add noisy structural context, while local neighborhoods
remain useful. However, spectral embedding still has the best w-RMSE in this
3-seed check, so the current Douban headline candidate should remain
`netobs_reliability_spectral`.

## Decision

Do not promote Node2Vec p/q variants to 20 seeds yet. The useful next step is
to pair the best local structural candidates (`spectral`, `sparse_node2vec_bfs`,
and `elph_role`) with the next entropy upgrade, rather than running a 20-seed
Node2Vec-only confirmation.

Primary artifacts:

- `outputs/tables/douban_svdpp_node2vec_pq_grid_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_node2vec_pq_grid_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_node2vec_pq_grid_3seeds_report.md`
- `reports/douban_svdpp_node2vec_pq_grid_3seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_node2vec_pq_grid_3seeds_w_rmse.png`
