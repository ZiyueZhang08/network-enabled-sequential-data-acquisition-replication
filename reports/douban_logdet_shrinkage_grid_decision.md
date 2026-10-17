# Douban Logdet + Shrinkage Grid Decision

This note summarizes the two-stage Douban mechanism grid after adding `observational_logdet` entropy and adaptive `shrinkage_dls`.

## Stage 1: Shrinkage x Lambda

Config:

- `configs/douban_svdpp_logdet_shrinkage_stage1_3seeds.json`

Artifacts:

- `outputs/tables/douban_svdpp_logdet_shrinkage_stage1_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_logdet_shrinkage_stage1_3seeds_metrics.csv`
- `outputs/figures/douban_svdpp_logdet_shrinkage_stage1_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_logdet_shrinkage_stage1_3seeds_cumulative_reward.png`

Top variants:

| variant | shrinkage | lambda | reward | w-RMSE | reward pct | entropy pct |
|---|---:|---:|---:|---:|---:|---:|
| `s07_l010` | 0.7 | 0.10 | 2997.67 | 0.6605 | 0.9995 | 0.6362 |
| `s05_l015` | 0.5 | 0.15 | 2960.00 | 0.6649 | 0.9992 | 0.6728 |
| `s03_l010` | 0.3 | 0.10 | 2950.33 | 0.6671 | 0.9995 | 0.6517 |
| `s09_l010` | 0.9 | 0.10 | 2905.67 | 0.6713 | 0.9995 | 0.6200 |

Main reading:

- `s07_l010` is the first 3-seed setting that beats both current DU-Net and the previous no-DLS pilot on w-RMSE.
- Lower lambda is better for w-RMSE/reward in this logdet + shrinkage setting.
- Increasing lambda to 0.30 raises entropy percentile but hurts reward and w-RMSE.

## Stage 2: Logdet Regularization

Config:

- `configs/douban_svdpp_logdet_shrinkage_stage2_reg_3seeds.json`

Artifacts:

- `outputs/tables/douban_svdpp_logdet_shrinkage_stage2_reg_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_logdet_shrinkage_stage2_reg_3seeds_metrics.csv`
- `outputs/figures/douban_svdpp_logdet_shrinkage_stage2_reg_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_logdet_shrinkage_stage2_reg_3seeds_cumulative_reward.png`

Top variants:

| variant | reg | reward | w-RMSE | reward pct | entropy pct |
|---|---:|---:|---:|---:|---:|
| `s07_l010_r1` | 1.0 | 2997.67 | 0.6605 | 0.9995 | 0.6362 |
| `s07_l010_r10` | 10.0 | 2984.00 | 0.6625 | 0.9995 | 0.6536 |
| `s05_l015_r1` | 1.0 | 2960.00 | 0.6649 | 0.9992 | 0.6728 |
| `s05_l015_r10` | 10.0 | 2943.67 | 0.6668 | 0.9991 | 0.6898 |

Main reading:

- `entropy_logdet_regularization=1.0` remains the best setting.
- `reg=10.0` is close but slightly worse; `reg=0.1` is consistently worse.
- The best current Douban candidate is now `logdet + shrinkage DLS` with `dls_shrinkage=0.7`, `lambda_entropy=0.10`, `entropy_logdet_regularization=1.0`, `selection=topk`.

## Comparison to Current Baselines

| setting | seeds | reward | w-RMSE | status |
|---|---:|---:|---:|---|
| current DU-Net tuned | 3 | 2871.67 | 0.6753 | baseline |
| previous no-DLS pilot | 3 | 2979.00 | 0.6643 | strong sparse-graph baseline |
| best logdet + shrinkage | 3 | 2997.67 | 0.6605 | current best Douban candidate |
| current DU-Net tuned | 20 | 2836.90 | 0.6805 | frozen main |
| no-DLS confirmation | 20 | 2863.05 | 0.6777 | best 20-seed follow-up so far |

## Decision

I expanded the best 3-seed candidate to 20 seeds:

- `du_net_logdet_shrinkage_dls`
- `dls_shrinkage=0.7`
- `lambda_entropy=0.10`
- `entropy_logdet_regularization=1.0`
- `selection=topk`

20-seed config:

- `configs/douban_svdpp_logdet_shrinkage_best_20seeds.json`

20-seed result:

| policy | reward | w-RMSE | reward pct | entropy pct |
|---|---:|---:|---:|---:|
| `du_net_no_dls` | 2863.05 | 0.6777 | 0.9996 | 0.5689 |
| `du_net` | 2838.65 | 0.6801 | 0.9996 | 0.5385 |
| `myopic` | 2844.60 | 0.6801 | 0.9996 | 0.5435 |
| `du_net_logdet_shrinkage_dls` | 2821.15 | 0.6824 | 0.9995 | 0.6435 |

Final decision:

- The 3-seed `s07_l010` improvement did not survive 20-seed confirmation.
- `logdet + shrinkage` is mechanistically useful because it raises entropy percentile from roughly 0.54 to 0.64.
- It is not yet the new main Douban configuration because w-RMSE worsens to 0.6824.
- `du_net_no_dls` remains the best confirmed 20-seed Douban follow-up.

The next Douban step should not be another simple logdet weight search. It should change the information criterion so that higher information percentile targets implementation error more directly. Promising directions:

- item/user cold-start weighted logdet;
- test-weighted or popularity-weighted information gain;
- graph-reliability shrinkage where low-degree DLS is nearly off;
- separate reward-information batch composition instead of scalar z-score mixing.
