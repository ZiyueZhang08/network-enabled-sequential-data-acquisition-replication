# Diagnostics: douban_svdpp_reliability_stochastic_entropy_3seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5406 | 1.2343 | 0.1287 |
| netobs_reliability_topk | 0.9996 | 0.5467 | 1.2252 | 0.1238 |
| stochastic_netobs_p005 | 0.9996 | 0.3733 | 4.7875 | 0.1605 |
| reliability_stochastic_p005 | 0.9996 | 0.3703 | 4.8827 | 0.1613 |
| reliability_stochastic_p005_degree_norm | 0.9996 | 0.4212 | 3.8183 | 0.1408 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_reliability_stochastic_entropy_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_reliability_stochastic_entropy_3seeds_diagnostics_by_cycle.csv`.
