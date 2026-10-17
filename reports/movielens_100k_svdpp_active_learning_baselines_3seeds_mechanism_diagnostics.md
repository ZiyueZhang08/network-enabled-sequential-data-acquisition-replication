# Mechanism Diagnostics: movielens_100k_svdpp_active_learning_baselines_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DU-Net | 2717.33 | 0.8229 | 0.8207 | 0.8242 | 0.8166 | 0.8051 | 0.9812 | 0.8582 | 25.8832 |
| graph_social | 2748.67 | 0.8246 | 0.7940 | 0.8310 | 0.8184 | 0.8119 | 0.9994 | 0.5184 | 26.2435 |
| mf_linear_ucb_a005 | 2702.33 | 0.8306 | 0.7950 | 0.8371 | 0.8239 | 0.8184 | nan | nan | nan |
| Myopic | 2704.00 | 0.8354 | 0.8061 | 0.8408 | 0.8284 | 0.8250 | 0.9994 | 0.5641 | 24.6485 |
| entropy | 1041.33 | 1.0536 | 1.0761 | 1.0475 | 1.0495 | 1.0426 | 0.4450 | 0.9994 | 23.9670 |
| active_hybrid_a020 | 267.67 | 1.1637 | 1.2233 | 1.1416 | 1.1524 | 1.1573 | nan | nan | nan |
| prediction_uncertainty | 229.33 | 1.1719 | 1.2534 | 1.1564 | 1.1673 | 1.1582 | nan | nan | nan |
| active_hybrid_a005 | 186.33 | 1.1866 | 1.2519 | 1.1657 | 1.1725 | 1.1791 | nan | nan | nan |
| leverage_uncertainty | 92.00 | 1.1868 | 1.2671 | 1.1530 | 1.1830 | 1.1636 | nan | nan | nan |

## Figures

- `outputs\figures\movielens_100k_svdpp_active_learning_baselines_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_active_learning_baselines_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_active_learning_baselines_3seeds_selected_degree_bins.png`
- `outputs\figures\movielens_100k_svdpp_active_learning_baselines_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\movielens_100k_svdpp_active_learning_baselines_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
