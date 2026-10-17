# Mechanism Diagnostics: movielens_100k_svdpp_social_rec_baselines_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DU-Net | 2717.33 | 0.8229 | 0.8207 | 0.8242 | 0.8166 | 0.8051 | 0.9812 | 0.8582 | 25.8832 |
| graph_social | 2748.67 | 0.8246 | 0.7940 | 0.8310 | 0.8184 | 0.8119 | 0.9994 | 0.5184 | 26.2435 |
| social_hybrid_w050 | 2719.67 | 0.8301 | 0.8022 | 0.8333 | 0.8231 | 0.8188 | 0.9994 | 0.7238 | 25.1178 |
| mf_linear_ucb_a005 | 2702.33 | 0.8306 | 0.7950 | 0.8371 | 0.8239 | 0.8184 | nan | nan | nan |
| social_hybrid_w025 | 2733.67 | 0.8306 | 0.8030 | 0.8353 | 0.8235 | 0.8185 | 0.9994 | 0.6567 | 24.6397 |
| Myopic | 2704.00 | 0.8354 | 0.8061 | 0.8408 | 0.8284 | 0.8250 | 0.9994 | 0.5641 | 24.6485 |
| social_hybrid_w100 | 2587.33 | 0.8425 | 0.8461 | 0.8422 | 0.8379 | 0.8309 | 0.9994 | 0.8586 | 25.3202 |
| social_neighbor_item | 2362.67 | 0.8678 | 0.8522 | 0.8688 | 0.8636 | 0.8602 | 0.9994 | 0.9112 | 25.2972 |

## Figures

- `outputs\figures\movielens_100k_svdpp_social_rec_baselines_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_social_rec_baselines_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\movielens_100k_svdpp_social_rec_baselines_3seeds_selected_degree_bins.png`
- `outputs\figures\movielens_100k_svdpp_social_rec_baselines_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\movielens_100k_svdpp_social_rec_baselines_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
