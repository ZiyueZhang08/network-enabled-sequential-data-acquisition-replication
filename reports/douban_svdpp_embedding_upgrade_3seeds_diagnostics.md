# Diagnostics: douban_svdpp_embedding_upgrade_3seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5406 | 1.2343 | 0.1287 |
| netobs_reliability_spectral | 0.9996 | 0.5467 | 1.2252 | 0.1238 |
| netobs_reliability_sparse_node2vec | 0.9996 | 0.5735 | 1.2545 | 0.1342 |
| netobs_reliability_biased_node2vec | 0.9996 | 0.5681 | 1.2063 | 0.1313 |
| netobs_reliability_elph_struct | 0.9996 | 0.5515 | 1.2862 | 0.1307 |
| netobs_reliability_elph_role | 0.9996 | 0.5731 | 1.3427 | 0.1403 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_embedding_upgrade_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_embedding_upgrade_3seeds_diagnostics_by_cycle.csv`.
