# Diagnostics: douban_svdpp_stochastic_network_entropy_3seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5406 | 1.2343 | 0.1287 |
| netobs_reliability_topk | 0.9996 | 0.5467 | 1.2252 | 0.1238 |
| stochastic_netobs_p001 | 0.9996 | 0.5653 | 1.2582 | 0.1255 |
| stochastic_netobs_p005 | 0.9996 | 0.3733 | 4.7875 | 0.1605 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_stochastic_network_entropy_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_stochastic_network_entropy_3seeds_diagnostics_by_cycle.csv`.
