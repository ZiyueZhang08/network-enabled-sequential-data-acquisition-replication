# Diagnostics: douban_svdpp_graph_density_final_candidates_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| original_no_dls | 0.9996 | 0.5689 | 1.8632 | 0.1675 |
| original_netobs_reliability | 0.9996 | 0.5672 | 1.2373 | 0.1705 |
| drop50_netobs_reliability | 0.9996 | 0.6097 | 0.5292 | 0.1702 |
| rewire50_netobs_reliability | 0.9996 | 0.5652 | 0.9945 | 0.1714 |
| densify50_netobs_reliability | 0.9996 | 0.5563 | 2.4379 | 0.1633 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_graph_density_final_candidates_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_graph_density_final_candidates_20seeds_diagnostics_by_cycle.csv`.
