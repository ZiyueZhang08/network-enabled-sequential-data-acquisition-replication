# Douban Mechanism Matrix: 3-Seed Findings

This experiment tests the next mechanism-level candidates after the frozen Douban result showed that current DU-Net is too close to myopic and that no-DLS can outperform DLS on the sparse Douban graph.

Config:

- `configs/douban_svdpp_mechanism_matrix_3seeds.json`

Artifacts:

- `outputs/tables/douban_svdpp_mechanism_matrix_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_mechanism_matrix_3seeds_metrics.csv`
- `outputs/tables/douban_svdpp_mechanism_matrix_3seeds_diagnostics_summary.csv`
- `outputs/figures/douban_svdpp_mechanism_matrix_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_mechanism_matrix_3seeds_cumulative_reward.png`

## Final-Cycle Results

| policy | reward | w-RMSE | reading |
|---|---:|---:|---|
| `du_net_no_dls` | 2979.00 | 0.6643 | Best 3-seed w-RMSE; confirms DLS noise remains a serious Douban issue. |
| `du_net_shrinkage_dls` | 2955.33 | 0.6666 | Shrinkage DLS recovers most of the no-DLS gain and beats current DU-Net. |
| `du_net_logdet_shrinkage_dls` | 2901.00 | 0.6715 | Strong information shift with good w-RMSE, but not as good as no-DLS/shrinkage alone. |
| `du_net` | 2871.67 | 0.6753 | Current tuned DU-Net baseline. |
| `du_net_logdet_shrinkage_exp_tau05` | 2822.00 | 0.6820 | Stochastic sampling hurts this 3-seed matrix. |
| `du_net_logdet_entropy` | 2792.33 | 0.6861 | Log-det entropy alone is too disruptive without DLS shrinkage. |

## Selection Diagnostics

| policy | reward percentile | entropy percentile | avg degree | cold-user share |
|---|---:|---:|---:|---:|
| `du_net` | 0.9996 | 0.5321 | 1.6682 | 0.1483 |
| `du_net_no_dls` | 0.9996 | 0.5727 | 1.6610 | 0.1192 |
| `du_net_shrinkage_dls` | 0.9996 | 0.5948 | 1.3943 | 0.1468 |
| `du_net_logdet_entropy` | 0.9813 | 0.6026 | 1.6413 | 0.1440 |
| `du_net_logdet_shrinkage_dls` | 0.9979 | 0.7072 | 0.9692 | 0.1075 |
| `du_net_logdet_shrinkage_exp_tau05` | 0.9897 | 0.6801 | 1.1777 | 0.1448 |

## Interpretation

The implementation changes worked mechanistically:

- `observational_logdet` makes the policy less purely reward-driven.
- `shrinkage_dls` improves over current full DLS on Douban.
- `logdet + shrinkage` raises entropy percentile from 0.5321 to 0.7072, which is the first clear sign that DU-Net is moving away from myopic acquisition.

The best short-run accuracy is still `du_net_no_dls`, so sparse-graph DLS remains the main bottleneck. The most promising candidate for a paper-faithful DU-Net direction is not plain logdet and not exp sampling; it is `logdet + shrinkage DLS`, followed by tuning shrinkage strength and lambda.

## Next Decision

Run a small grid before any 20-seed expansion:

- `dls_shrinkage`: 0.3, 0.5, 0.7, 0.9
- `lambda_entropy`: 0.1, 0.15, 0.2, 0.3
- `entropy_logdet_regularization`: 0.1, 1.0, 10.0
- selection: keep `topk` first; `exp_tau_0.5` is not currently strong enough.

Only expand to 20 seeds if a setting beats both current DU-Net and no-DLS in 3-seed w-RMSE or gives a materially better reward-information tradeoff with close w-RMSE.
