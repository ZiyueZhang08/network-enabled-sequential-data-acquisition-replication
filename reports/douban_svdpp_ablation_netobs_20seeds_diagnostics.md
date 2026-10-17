# Diagnostics: douban_svdpp_ablation_netobs_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| current_du_net | 0.9996 | 0.5385 | 1.8359 | 0.1787 |
| no_dls | 0.9996 | 0.5689 | 1.8632 | 0.1675 |
| no_network | 0.9996 | 0.5306 | 1.2569 | 0.1896 |
| dynamic_sigma | 0.9996 | 0.5353 | 1.7174 | 0.1738 |
| no_entropy | 0.9996 | 0.5308 | 1.8429 | 0.1792 |
| no_reward | 0.5023 | 0.9996 | 1.3271 | 0.3933 |
| reliability_dls | 0.9996 | 0.5720 | 1.2997 | 0.1726 |
| netobs_reliability_spectral | 0.9996 | 0.5672 | 1.2373 | 0.1705 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_ablation_netobs_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_ablation_netobs_20seeds_diagnostics_by_cycle.csv`.
