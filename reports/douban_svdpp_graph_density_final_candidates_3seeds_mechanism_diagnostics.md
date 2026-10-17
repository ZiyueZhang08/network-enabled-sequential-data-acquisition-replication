# Mechanism Diagnostics: douban_svdpp_graph_density_final_candidates_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_netobs_reliability | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| densify50_netobs_reliability | 2990.33 | 0.6629 | 0.6635 | 0.6621 | 0.6602 | 0.6682 | 0.9996 | 0.5613 | 2.3042 |
| original_no_dls | 2979.00 | 0.6643 | 0.6648 | 0.6637 | 0.6616 | 0.6650 | 0.9996 | 0.5727 | 1.6610 |
| rewire50_netobs_reliability | 2969.00 | 0.6650 | 0.6656 | 0.6641 | 0.6645 | 0.6651 | 0.9996 | 0.5807 | 0.8583 |
| drop50_netobs_reliability | 2874.33 | 0.6776 | 0.6783 | 0.6768 | 0.6749 | 0.6841 | 0.9996 | 0.5828 | 0.4705 |

## Figures

- `outputs\figures\douban_svdpp_graph_density_final_candidates_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_graph_density_final_candidates_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
