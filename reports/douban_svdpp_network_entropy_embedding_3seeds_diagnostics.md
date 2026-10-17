# Diagnostics: douban_svdpp_network_entropy_embedding_3seeds

## Policy-Level Selection Diagnostics

| policy | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5406 | 1.2343 | 0.1287 |
| du_net_shrinkage_dls | 0.9994 | 0.6101 | 1.1224 | 0.1266 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_network_entropy_embedding_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_network_entropy_embedding_3seeds_diagnostics_by_cycle.csv`.
