# Diagnostics: douban_svdpp_network_entropy_spectral_20seeds

## Policy-Level Selection Diagnostics

| policy | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5435 | 1.2465 | 0.1688 |
| du_net_shrinkage_dls | 0.9996 | 0.5672 | 1.2373 | 0.1705 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_network_entropy_spectral_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_network_entropy_spectral_20seeds_diagnostics_by_cycle.csv`.
