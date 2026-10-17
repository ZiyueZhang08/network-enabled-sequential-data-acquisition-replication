# Mechanism Diagnostics: douban_svdpp_batch_composition_20seeds

## Final and Selection Summary

| policy | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Myopic | 2844.60 | 0.6801 | 0.6798 | 0.6804 | 0.6761 | 0.6856 | 0.9996 | 0.5435 | 1.2465 |
| No DLS | 2818.00 | 0.6832 | 0.6813 | 0.6848 | 0.6798 | 0.6879 | 0.9991 | 0.6864 | 2.0138 |
| Reliability DLS | 2813.95 | 0.6833 | 0.6827 | 0.6837 | 0.6788 | 0.6901 | 0.9991 | 0.6889 | 0.8552 |
| Weighted Logdet + Reliability DLS | 2804.95 | 0.6843 | 0.6830 | 0.6853 | 0.6798 | 0.6910 | 0.9991 | 0.7258 | 0.8606 |

## Figures

- `outputs\figures\douban_svdpp_batch_composition_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_batch_composition_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_batch_composition_20seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_batch_composition_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_batch_composition_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
