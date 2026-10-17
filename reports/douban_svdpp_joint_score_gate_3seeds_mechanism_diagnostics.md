# Mechanism Diagnostics: douban_svdpp_joint_score_gate_3seeds

## Final and Selection Summary

| variant | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spectral_netobs_linear | 2996.67 | 0.6626 | 0.6638 | 0.6611 | 0.6597 | 0.6641 | 0.9996 | 0.5467 | 1.2252 |
| spectral_joint_gate_005 | 2946.67 | 0.6660 | 0.6672 | 0.6647 | 0.6629 | 0.6690 | 0.9992 | 0.6629 | 0.9598 |
| Myopic | 2921.33 | 0.6710 | 0.6697 | 0.6723 | 0.6676 | 0.6741 | 0.9996 | 0.5406 | 1.2343 |
| spectral_joint_gate_010 | 2848.33 | 0.6780 | 0.6769 | 0.6793 | 0.6744 | 0.6828 | 0.9982 | 0.7352 | 0.7610 |
| spectral_joint_gate_020 | 2721.00 | 0.6945 | 0.6938 | 0.6951 | 0.6892 | 0.7035 | 0.9960 | 0.8643 | 0.4025 |
| spectral_joint_gate_040 | 2549.67 | 0.7192 | 0.7166 | 0.7223 | 0.7127 | 0.7318 | 0.9930 | 0.9419 | 0.2742 |

## Figures

- `outputs\figures\douban_svdpp_joint_score_gate_3seeds_cold_warm_w_rmse.png`
- `outputs\figures\douban_svdpp_joint_score_gate_3seeds_centrality_binned_w_rmse.png`
- `outputs\figures\douban_svdpp_joint_score_gate_3seeds_selected_degree_bins.png`
- `outputs\figures\douban_svdpp_joint_score_gate_3seeds_dls_shrinkage_over_cycles.png`
- `outputs\figures\douban_svdpp_joint_score_gate_3seeds_reward_entropy_percentiles.png`

## Interpretation

- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.
- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.
- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.
- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.
