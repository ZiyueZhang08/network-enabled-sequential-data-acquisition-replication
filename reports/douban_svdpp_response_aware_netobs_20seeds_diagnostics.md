# Diagnostics: douban_svdpp_response_aware_netobs_20seeds

## Policy-Level Selection Diagnostics

| variant | reward pct | entropy pct | user degree | cold-user share |
|---|---:|---:|---:|---:|
| myopic | 0.9996 | 0.5409 | 1.2925 | 0.1821 |
| reliability_dls | 0.9996 | 0.5805 | 1.2792 | 0.1816 |
| netobs_reliability_spectral | 0.9996 | 0.5782 | 1.2767 | 0.1910 |
| netobs_response_centered_str2 | 0.9996 | 0.5800 | 1.2127 | 0.1777 |

Interpretation guide:

- Higher reward percentile means the policy behaves more like myopic exploitation.
- Higher entropy percentile means stronger active-learning behavior.
- Higher user degree indicates centrality bias from network diffusion or DLS.
- Higher cold-user share indicates more cold-start exploration.

CSV artifacts: `outputs\tables\douban_svdpp_response_aware_netobs_20seeds_diagnostics_summary.csv`, `outputs\tables\douban_svdpp_response_aware_netobs_20seeds_diagnostics_by_cycle.csv`.
