# Mechanism Diagnostics: douban_svdpp_reward_frontier_guard_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netobs_reliability_topk | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| weighted_guard_q99_l020 | 2950.00 | 0.6657 | 0.6660 | 0.6654 | 0.6627 | 0.6684 | 0.9993 | 0.6467 | 1.0052 |
| netobs_guard_q99_l020 | 2925.67 | 0.6687 | 0.6681 | 0.6693 | 0.6655 | 0.6717 | 0.9992 | 0.6447 | 0.9920 |
| netobs_guard_q99_l010 | 2924.00 | 0.6705 | 0.6698 | 0.6711 | 0.6673 | 0.6731 | 0.9995 | 0.6003 | 1.2283 |
| weighted_guard_q99_l010 | 2922.33 | 0.6706 | 0.6701 | 0.6711 | 0.6676 | 0.6727 | 0.9995 | 0.5902 | 1.2222 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |
| netobs_guard_q95_l020 | 2899.67 | 0.6715 | 0.6705 | 0.6727 | 0.6686 | 0.6744 | 0.9988 | 0.6756 | 0.9803 |

## Figures

- `outputs\figures\douban_svdpp_reward_frontier_guard_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_reward_frontier_guard_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_reward_frontier_guard_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_reward_frontier_guard_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_reward_frontier_guard_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
