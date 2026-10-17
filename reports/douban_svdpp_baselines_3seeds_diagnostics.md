# Diagnostics: douban_svdpp_baselines_3seeds

## Policy-Level Selection Diagnostics

| policy | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| random | 0.4982 | 0.5033 | 0.9957 | 0.4537 |
| myopic | 0.9996 | 0.5406 | 1.2343 | 0.1287 |
| entropy | 0.5042 | 0.9996 | 1.4080 | 0.4105 |
| linear_ucb | nan | nan | nan | nan |
| low_rank_bandit | nan | nan | nan | nan |
| thompson_sampling | nan | nan | nan | nan |
| graph_social | 0.9996 | 0.5161 | 1.7330 | 0.1403 |
| du_net | 0.9996 | 0.5321 | 1.6682 | 0.1483 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_baselines_3seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_baselines_3seeds_diagnostics_by_cycle.csv`.
