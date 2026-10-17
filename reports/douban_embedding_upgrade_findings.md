# Douban Embedding Upgrade Findings

This note records the next fidelity step after the 20-seed Douban netobs extension.

## Implementation Added

- `graph_embedding_method=sparse_node2vec_biased`: dependency-free second-order Node2Vec-style random walks with return/in-out bias.
- `graph_embedding_method=elph_role`: role-oriented structural features: degree, two-hop reach, triangle density, egonet density, neighbor-degree moments, and a light biased-walk context.

These are still local proxy implementations, but they are more faithful than the earlier DeepWalk-like `sparse_node2vec` and shallow `elph_struct` variants.

## Douban 3-Seed Result

Config: `configs/douban_svdpp_embedding_upgrade_3seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Still the best embedding choice in this pilot. |
| netobs reliability ELPH-role | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Clear improvement over older ELPH-struct, but not enough to beat spectral. |
| netobs reliability biased Node2Vec | 2965.33 | 0.6661 | 0.5681 | 1.2063 | Improves over sparse Node2Vec, but not enough to promote. |
| netobs reliability sparse Node2Vec | 2949.00 | 0.6672 | 0.5735 | 1.2545 | Positive but weaker than biased Node2Vec. |
| netobs reliability ELPH-struct | 2946.67 | 0.6681 | 0.5515 | 1.2862 | Older structural proxy is dominated by ELPH-role. |
| myopic | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong baseline. |

## Interpretation

The upgraded embeddings are directionally useful:

- `elph_role` improves w-RMSE from `0.6681` to `0.6633` relative to `elph_struct`;
- `sparse_node2vec_biased` improves w-RMSE from `0.6672` to `0.6661` relative to `sparse_node2vec`;
- both upgraded methods increase entropy-oriented selection compared with spectral.

But spectral embedding remains best in the 3-seed result. The current dependency-free Node2Vec/ELPH proxies should not be promoted to 20 seeds yet.

## Current Decision

Keep `netobs_reliability_spectral` as the Douban paper-faithful candidate. Treat `elph_role` as the next embedding candidate only if a later response-aware entropy or graph-density pass changes the ranking.

Primary artifacts:

- `outputs/tables/douban_svdpp_embedding_upgrade_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_embedding_upgrade_3seeds_mechanism_summary.csv`
- `outputs/figures/douban_svdpp_embedding_upgrade_3seeds_w_rmse.png`
- `reports/douban_svdpp_embedding_upgrade_3seeds_mechanism_diagnostics.md`
