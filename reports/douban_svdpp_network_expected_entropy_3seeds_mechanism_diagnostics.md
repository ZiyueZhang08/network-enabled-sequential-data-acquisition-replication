# Mechanism Diagnostics: douban_svdpp_network_expected_entropy_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expected_netobs_reliability_spectral | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| NetObs + Reliability DLS | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |

## Figures

- `outputs\figures\douban_svdpp_network_expected_entropy_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_network_expected_entropy_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_network_expected_entropy_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_network_expected_entropy_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_network_expected_entropy_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
