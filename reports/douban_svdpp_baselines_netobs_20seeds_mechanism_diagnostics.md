# Mechanism Diagnostics: douban_svdpp_baselines_netobs_20seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NetObs + Reliability DLS | 2861.80 | 0.6778 | 0.6773 | 0.6781 | 0.6739 | 0.6829 | 0.9996 | 0.5672 | 1.2373 |
| DU-Net | 2838.65 | 0.6801 | 0.6789 | 0.6812 | 0.6765 | 0.6844 | 0.9996 | 0.5385 | 1.8359 |
| Myopic | 2844.60 | 0.6801 | 0.6798 | 0.6804 | 0.6761 | 0.6856 | 0.9996 | 0.5435 | 1.2465 |
| graph_social | 2835.00 | 0.6812 | 0.6799 | 0.6824 | 0.6775 | 0.6860 | 0.9996 | 0.5308 | 1.8429 |
| thompson_sampling | 1409.65 | 0.9417 | 0.9412 | 0.9436 | 0.9325 | 0.9653 | nan | nan | nan |
| linear_ucb | 1400.10 | 0.9424 | 0.9467 | 0.9366 | 0.9339 | 0.9651 | nan | nan | nan |
| low_rank_bandit | 1378.50 | 0.9487 | 0.9490 | 0.9473 | 0.9392 | 0.9732 | nan | nan | nan |
| random | 1370.35 | 0.9495 | 0.9486 | 0.9499 | 0.9406 | 0.9722 | 0.4989 | 0.5007 | 1.0088 |
| entropy | 1358.75 | 0.9543 | 0.9548 | 0.9537 | 0.9456 | 0.9774 | 0.4957 | 0.9996 | 1.3213 |

## Figures

- `outputs\figures\douban_svdpp_baselines_netobs_20seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_baselines_netobs_20seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_baselines_netobs_20seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_baselines_netobs_20seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_baselines_netobs_20seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
