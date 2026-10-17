# Mechanism Diagnostics: douban_svdpp_node2vec_pq_grid_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NetObs + Reliability DLS | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| netobs_reliability_node2vec_bfs | 2983.33 | 0.6633 | 0.6624 | 0.6639 | 0.6606 | 0.6644 | 0.9996 | 0.5717 | 1.4160 |
| netobs_reliability_elph_role | 2987.00 | 0.6633 | 0.6638 | 0.6627 | 0.6609 | 0.6637 | 0.9996 | 0.5731 | 1.3427 |
| netobs_reliability_node2vec_return | 2968.67 | 0.6657 | 0.6658 | 0.6654 | 0.6629 | 0.6672 | 0.9996 | 0.5641 | 1.2583 |
| netobs_reliability_node2vec_dfs | 2965.33 | 0.6661 | 0.6663 | 0.6658 | 0.6633 | 0.6680 | 0.9996 | 0.5681 | 1.2063 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |
| netobs_reliability_node2vec_outward | 2914.33 | 0.6718 | 0.6702 | 0.6732 | 0.6686 | 0.6747 | 0.9996 | 0.5669 | 1.2662 |

## Figures

- `outputs\figures\douban_svdpp_node2vec_pq_grid_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_node2vec_pq_grid_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_node2vec_pq_grid_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_node2vec_pq_grid_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_node2vec_pq_grid_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
