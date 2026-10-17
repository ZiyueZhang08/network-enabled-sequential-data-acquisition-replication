# Mechanism Diagnostics: flixster_svdpp_ablation_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DU-Net | 2919.65 | 0.6309 | 0.6417 | 0.6185 | 0.6320 | 0.6391 | 0.9977 | 0.5499 | 8.7145 |
| no_network | 2919.65 | 0.6309 | 0.6417 | 0.6185 | 0.6320 | 0.6391 | 0.9977 | 0.5499 | 8.7145 |
| no_entropy | 2911.68 | 0.6314 | 0.6419 | 0.6192 | 0.6327 | 0.6386 | 0.9978 | 0.5194 | 8.7002 |
| No DLS | 2831.20 | 0.6347 | 0.6476 | 0.6202 | 0.6433 | 0.6339 | 0.9978 | 0.6116 | 8.6500 |
| dynamic_sigma | 2770.05 | 0.6406 | 0.6569 | 0.6222 | 0.6530 | 0.6375 | 0.9977 | 0.7219 | 9.0768 |
| no_reward | 1284.22 | 0.9979 | 1.0490 | 0.9283 | 1.0716 | 0.8965 | 0.4761 | 0.9978 | 10.3283 |

## Figures

- `outputs\figures\flixster_svdpp_ablation_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\flixster_svdpp_ablation_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\flixster_svdpp_ablation_20seeds_selected_degree_bins.png`
- `outputs\figures\flixster_svdpp_ablation_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\flixster_svdpp_ablation_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
