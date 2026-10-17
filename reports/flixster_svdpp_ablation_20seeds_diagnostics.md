# Diagnostics: flixster_svdpp_ablation_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| current_du_net | 0.9977 | 0.5499 | 8.7145 | 0.2135 |
| no_dls | 0.9978 | 0.6116 | 8.6500 | 0.2038 |
| no_network | 0.9977 | 0.5499 | 8.7145 | 0.2135 |
| dynamic_sigma | 0.9977 | 0.7219 | 9.0768 | 0.1914 |
| no_entropy | 0.9978 | 0.5194 | 8.7002 | 0.2144 |
| no_reward | 0.4761 | 0.9978 | 10.3282 | 0.3289 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\flixster_svdpp_ablation_20seeds_diagnostics_summary.csv`, `outputs\tables\flixster_svdpp_ablation_20seeds_diagnostics_by_cycle.csv`.
