# Mechanism Diagnostics: douban_svdpp_formal_mechanism_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No DLS | 2863.05 | 0.6777 | 0.6771 | 0.6781 | 0.6740 | 0.6821 | 0.9996 | 0.5689 | 1.8632 |
| Reliability DLS | 2861.65 | 0.6778 | 0.6773 | 0.6781 | 0.6741 | 0.6824 | 0.9996 | 0.5720 | 1.2997 |
| NetObs + Reliability DLS | 2861.80 | 0.6778 | 0.6773 | 0.6781 | 0.6739 | 0.6829 | 0.9996 | 0.5672 | 1.2373 |
| Weighted Logdet + Reliability DLS | 2860.05 | 0.6780 | 0.6777 | 0.6781 | 0.6741 | 0.6831 | 0.9996 | 0.6073 | 1.2590 |
| DU-Net | 2838.65 | 0.6801 | 0.6789 | 0.6812 | 0.6765 | 0.6844 | 0.9996 | 0.5385 | 1.8359 |
| Myopic | 2844.60 | 0.6801 | 0.6798 | 0.6804 | 0.6761 | 0.6856 | 0.9996 | 0.5435 | 1.2465 |

## Figures

- `outputs\figures\douban_svdpp_formal_mechanism_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_formal_mechanism_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_formal_mechanism_20seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_formal_mechanism_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_formal_mechanism_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
