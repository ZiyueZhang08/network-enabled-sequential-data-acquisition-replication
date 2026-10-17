# Mechanism Diagnostics: douban_svdpp_reliability_stochastic_entropy_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netobs_reliability_topk | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |
| reliability_stochastic_p005_degree_norm | 2864.33 | 0.6773 | 0.6736 | 0.6808 | 0.6756 | 0.6764 | 0.9996 | 0.4212 | 3.8183 |
| stochastic_netobs_p005 | 2661.67 | 0.7032 | 0.6948 | 0.7113 | 0.7015 | 0.7038 | 0.9996 | 0.3733 | 4.7875 |
| reliability_stochastic_p005 | 2661.33 | 0.7039 | 0.6958 | 0.7117 | 0.7020 | 0.7049 | 0.9996 | 0.3703 | 4.8827 |

## Figures

- `outputs\figures\douban_svdpp_reliability_stochastic_entropy_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_reliability_stochastic_entropy_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_reliability_stochastic_entropy_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_reliability_stochastic_entropy_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_reliability_stochastic_entropy_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
