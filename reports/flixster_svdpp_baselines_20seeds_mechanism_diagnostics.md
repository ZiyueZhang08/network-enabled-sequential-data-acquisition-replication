# Mechanism Diagnostics: flixster_svdpp_baselines_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DU-Net | 2919.65 | 0.6309 | 0.6417 | 0.6185 | 0.6320 | 0.6391 | 0.9977 | 0.5499 | 8.7145 |
| graph_social | 2911.68 | 0.6323 | 0.6424 | 0.6206 | 0.6334 | 0.6395 | 0.9978 | 0.5194 | 8.7002 |
| Myopic | 2845.38 | 0.6338 | 0.6479 | 0.6181 | 0.6409 | 0.6349 | 0.9978 | 0.5875 | 8.6680 |
| low_rank_bandit | 1807.17 | 0.8362 | 0.8716 | 0.8335 | 0.8838 | 0.7633 | nan | nan | nan |
| thompson_sampling | 1774.03 | 0.8462 | 0.8825 | 0.8436 | 0.8954 | 0.7706 | nan | nan | nan |
| random | 1520.83 | 0.9178 | 0.9661 | 0.8999 | 0.9777 | 0.8351 | 0.5021 | 0.4994 | 10.2489 |
| linear_ucb | 1392.40 | 0.9580 | 1.0058 | 0.8972 | 1.0231 | 0.8690 | nan | nan | nan |
| entropy | 1284.22 | 0.9984 | 1.0494 | 0.9290 | 1.0716 | 0.8973 | 0.4775 | 0.9978 | 10.3283 |

## Figures

- `outputs\figures\flixster_svdpp_baselines_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\flixster_svdpp_baselines_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\flixster_svdpp_baselines_20seeds_selected_degree_bins.png`
- `outputs\figures\flixster_svdpp_baselines_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\flixster_svdpp_baselines_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
