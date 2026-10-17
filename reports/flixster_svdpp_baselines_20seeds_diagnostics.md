# Diagnostics: flixster_svdpp_baselines_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| random | 0.5021 | 0.4994 | 10.2489 | 0.4422 |
| myopic | 0.9978 | 0.5875 | 8.6680 | 0.2030 |
| entropy | 0.4775 | 0.9978 | 10.3282 | 0.3289 |
| linear_ucb | nan | nan | nan | nan |
| low_rank_bandit | nan | nan | nan | nan |
| thompson_sampling | nan | nan | nan | nan |
| graph_social | 0.9978 | 0.5194 | 8.7002 | 0.2144 |
| du_net | 0.9977 | 0.5499 | 8.7145 | 0.2135 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\flixster_svdpp_baselines_20seeds_diagnostics_summary.csv`, `outputs\tables\flixster_svdpp_baselines_20seeds_diagnostics_by_cycle.csv`.
