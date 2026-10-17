# Diagnostics: douban_svdpp_baselines_netobs_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| random | 0.4989 | 0.5007 | 1.0088 | 0.4580 |
| myopic | 0.9996 | 0.5435 | 1.2465 | 0.1688 |
| entropy | 0.4957 | 0.9996 | 1.3213 | 0.3982 |
| linear_ucb | nan | nan | nan | nan |
| low_rank_bandit | nan | nan | nan | nan |
| thompson_sampling | nan | nan | nan | nan |
| graph_social | 0.9996 | 0.5308 | 1.8429 | 0.1792 |
| current_du_net | 0.9996 | 0.5385 | 1.8359 | 0.1787 |
| netobs_reliability_spectral | 0.9996 | 0.5672 | 1.2373 | 0.1705 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_baselines_netobs_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_baselines_netobs_20seeds_diagnostics_by_cycle.csv`.
