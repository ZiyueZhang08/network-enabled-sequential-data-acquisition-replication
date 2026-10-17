# Diagnostics: douban_svdpp_mechanism_matrix_3seeds

## Policy-Level Selection Diagnostics

| policy | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| du_net | 0.9996 | 0.5321 | 1.6682 | 0.1483 |
| du_net_no_dls | 0.9996 | 0.5727 | 1.6610 | 0.1192 |
| du_net_shrinkage_dls | 0.9996 | 0.5948 | 1.3943 | 0.1468 |
| du_net_logdet_entropy | 0.9813 | 0.6026 | 1.6413 | 0.1440 |
| du_net_logdet_shrinkage_dls | 0.9979 | 0.7072 | 0.9692 | 0.1075 |
| du_net_logdet_shrinkage_exp_tau05 | 0.9897 | 0.6801 | 1.1777 | 0.1448 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_mechanism_matrix_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_mechanism_matrix_3seeds_diagnostics_by_cycle.csv`.
