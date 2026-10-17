# Diagnostics: douban_svdpp_graph_density_final_candidates_3seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| original_no_dls | 0.9996 | 0.5727 | 1.6610 | 0.1192 |
| original_netobs_reliability | 0.9996 | 0.5467 | 1.2252 | 0.1238 |
| drop50_netobs_reliability | 0.9996 | 0.5828 | 0.4705 | 0.1363 |
| rewire50_netobs_reliability | 0.9996 | 0.5807 | 0.8583 | 0.1353 |
| densify50_netobs_reliability | 0.9996 | 0.5613 | 2.3042 | 0.1298 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_graph_density_final_candidates_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_graph_density_final_candidates_3seeds_diagnostics_by_cycle.csv`.
