# Mechanism Diagnostics: douban_svdpp_response_aware_netobs_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netobs_response_centered_str2 | 2360.35 | 0.6934 | 0.6927 | 0.6939 | 0.6889 | 0.7007 | 0.9996 | 0.5800 | 1.2127 |
| Reliability DLS | 2353.30 | 0.6948 | 0.6932 | 0.6963 | 0.6901 | 0.7021 | 0.9996 | 0.5805 | 1.2792 |
| Myopic | 2354.70 | 0.6953 | 0.6947 | 0.6957 | 0.6908 | 0.7025 | 0.9996 | 0.5409 | 1.2925 |
| NetObs + Reliability DLS | 2337.35 | 0.6969 | 0.6946 | 0.6991 | 0.6921 | 0.7045 | 0.9996 | 0.5782 | 1.2767 |

## Figures

- `outputs\figures\douban_svdpp_response_aware_netobs_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_response_aware_netobs_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_response_aware_netobs_20seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_response_aware_netobs_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_response_aware_netobs_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
