# Mechanism Diagnostics: movielens_100k_svdpp_bandit_upgrade_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DU-Net | 2717.33 | 0.8229 | 0.8207 | 0.8242 | 0.8166 | 0.8051 | 0.9812 | 0.8582 | 25.8832 |
| graph_social | 2748.67 | 0.8246 | 0.7940 | 0.8310 | 0.8184 | 0.8119 | 0.9994 | 0.5184 | 26.2435 |
| mf_linear_ucb_a005 | 2702.33 | 0.8306 | 0.7950 | 0.8371 | 0.8239 | 0.8184 | nan | nan | nan |
| Myopic | 2704.00 | 0.8354 | 0.8061 | 0.8408 | 0.8284 | 0.8250 | 0.9994 | 0.5641 | 24.6485 |
| mf_thompson_n008 | 2682.33 | 0.8390 | 0.8162 | 0.8418 | 0.8319 | 0.8268 | nan | nan | nan |
| mf_low_rank_ucb_a005 | 2642.67 | 0.8424 | 0.8269 | 0.8441 | 0.8333 | 0.8335 | nan | nan | nan |
| legacy_low_rank_bandit | 1749.00 | 0.9978 | 1.0404 | 0.9872 | 0.9909 | 0.9870 | nan | nan | nan |
| legacy_thompson_sampling | 1640.67 | 1.0184 | 1.0766 | 1.0078 | 1.0068 | 1.0187 | nan | nan | nan |

## Figures

- `outputs\figures\movielens_100k_svdpp_bandit_upgrade_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_bandit_upgrade_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_bandit_upgrade_3seeds_selected_degree_bins.png`
- `outputs\figures\movielens_100k_svdpp_bandit_upgrade_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\movielens_100k_svdpp_bandit_upgrade_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
