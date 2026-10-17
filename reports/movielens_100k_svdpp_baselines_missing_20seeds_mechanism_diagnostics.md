# Mechanism Diagnostics: movielens_100k_svdpp_baselines_missing_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph_social | 2751.50 | 0.8300 | 0.7954 | 0.8365 | 0.8258 | 0.8172 | 0.9994 | 0.5485 | 26.7092 |
| thompson_sampling | 1657.80 | 1.0193 | 1.0865 | 1.0073 | 1.0135 | 1.0171 | nan | nan | nan |

## Figures

- `outputs\figures\movielens_100k_svdpp_baselines_missing_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_baselines_missing_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_baselines_missing_20seeds_selected_degree_bins.png`
- `outputs\figures\movielens_100k_svdpp_baselines_missing_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\movielens_100k_svdpp_baselines_missing_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
