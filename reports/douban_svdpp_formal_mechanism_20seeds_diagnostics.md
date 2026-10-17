# Diagnostics: douban_svdpp_formal_mechanism_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5435 | 1.2465 | 0.1688 |
| current_du_net | 0.9996 | 0.5385 | 1.8359 | 0.1787 |
| no_dls | 0.9996 | 0.5689 | 1.8632 | 0.1675 |
| reliability_dls | 0.9996 | 0.5720 | 1.2997 | 0.1726 |
| weighted_logdet_reliability | 0.9996 | 0.6073 | 1.2590 | 0.1703 |
| netobs_reliability_spectral | 0.9996 | 0.5672 | 1.2373 | 0.1705 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_formal_mechanism_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_formal_mechanism_20seeds_diagnostics_by_cycle.csv`.
