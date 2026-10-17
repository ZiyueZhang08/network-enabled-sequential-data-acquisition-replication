# Mechanism Diagnostics: douban_svdpp_graph_density_final_candidates_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_no_dls | 2863.05 | 0.6777 | 0.6771 | 0.6781 | 0.6740 | 0.6821 | 0.9996 | 0.5689 | 1.8632 |
| original_netobs_reliability | 2861.80 | 0.6778 | 0.6773 | 0.6781 | 0.6739 | 0.6829 | 0.9996 | 0.5672 | 1.2373 |
| densify50_netobs_reliability | 2858.50 | 0.6781 | 0.6780 | 0.6779 | 0.6742 | 0.6860 | 0.9996 | 0.5563 | 2.4379 |
| drop50_netobs_reliability | 2841.10 | 0.6803 | 0.6799 | 0.6805 | 0.6777 | 0.6865 | 0.9996 | 0.6097 | 0.5292 |
| rewire50_netobs_reliability | 2825.20 | 0.6826 | 0.6817 | 0.6833 | 0.6805 | 0.6860 | 0.9996 | 0.5652 | 0.9945 |

## Figures

- `outputs\figures\douban_svdpp_graph_density_final_candidates_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_20seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
