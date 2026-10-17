# Mechanism Diagnostics: douban_svdpp_embedding_upgrade_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NetObs + Reliability DLS | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| netobs_reliability_elph_role | 2987.00 | 0.6633 | 0.6638 | 0.6627 | 0.6609 | 0.6637 | 0.9996 | 0.5731 | 1.3427 |
| netobs_reliability_biased_node2vec | 2965.33 | 0.6661 | 0.6663 | 0.6658 | 0.6633 | 0.6680 | 0.9996 | 0.5681 | 1.2063 |
| netobs_reliability_sparse_node2vec | 2949.00 | 0.6672 | 0.6662 | 0.6679 | 0.6643 | 0.6692 | 0.9996 | 0.5735 | 1.2545 |
| netobs_reliability_elph_struct | 2946.67 | 0.6681 | 0.6680 | 0.6680 | 0.6650 | 0.6705 | 0.9996 | 0.5515 | 1.2862 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |

## Figures

- `outputs\figures\douban_svdpp_embedding_upgrade_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_embedding_upgrade_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_embedding_upgrade_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_embedding_upgrade_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_embedding_upgrade_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
