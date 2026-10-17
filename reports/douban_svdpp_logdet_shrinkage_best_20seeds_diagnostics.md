# Diagnostics: douban_svdpp_logdet_shrinkage_best_20seeds

## Policy-Level Selection Diagnostics

| policy | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5435 | 1.2465 | 0.1688 |
| du_net | 0.9996 | 0.5385 | 1.8359 | 0.1787 |
| du_net_no_dls | 0.9996 | 0.5689 | 1.8632 | 0.1675 |
| du_net_logdet_shrinkage_dls | 0.9995 | 0.6435 | 1.0153 | 0.1621 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_logdet_shrinkage_best_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_logdet_shrinkage_best_20seeds_diagnostics_by_cycle.csv`.
