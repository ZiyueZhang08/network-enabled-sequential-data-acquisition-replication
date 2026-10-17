# Diagnostics: flixster_svdpp_baselines_3seeds

## Policy-Level Selection Diagnostics

| policy | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| random | 0.5051 | 0.5006 | 10.1300 | 0.4465 |
| myopic | 0.9978 | 0.5857 | 8.4860 | 0.2002 |
| entropy | 0.4610 | 0.9978 | 9.9665 | 0.3220 |
| linear_ucb | nan | nan | nan | nan |
| low_rank_bandit | nan | nan | nan | nan |
| thompson_sampling | nan | nan | nan | nan |
| graph_social | 0.9978 | 0.5073 | 8.5795 | 0.1983 |
| du_net | 0.9977 | 0.5425 | 8.3952 | 0.1973 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\flixster_svdpp_baselines_3seeds_diagnostics_summary.csv`, `outputs\tables\flixster_svdpp_baselines_3seeds_diagnostics_by_cycle.csv`.
