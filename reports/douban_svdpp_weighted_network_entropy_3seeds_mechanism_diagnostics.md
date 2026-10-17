# Mechanism Diagnostics: douban_svdpp_weighted_network_entropy_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netobs_reliability_topk | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| weighted_netobs_bfs_p002 | 2966.33 | 0.6662 | 0.6662 | 0.6661 | 0.6637 | 0.6669 | 0.9996 | 0.5090 | 2.3765 |
| weighted_netobs_p001 | 2956.67 | 0.6667 | 0.6663 | 0.6671 | 0.6642 | 0.6683 | 0.9996 | 0.5707 | 1.2877 |
| weighted_netobs_p002 | 2951.00 | 0.6678 | 0.6679 | 0.6676 | 0.6652 | 0.6688 | 0.9996 | 0.5024 | 2.2562 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |
| weighted_netobs_p005_degree | 2865.67 | 0.6770 | 0.6727 | 0.6811 | 0.6753 | 0.6762 | 0.9996 | 0.4217 | 3.7360 |

## Figures

- `outputs\figures\douban_svdpp_weighted_network_entropy_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_weighted_network_entropy_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_weighted_network_entropy_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_weighted_network_entropy_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_weighted_network_entropy_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
