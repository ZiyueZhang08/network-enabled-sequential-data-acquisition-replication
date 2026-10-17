# paper_faithful_movielens_schur_loglambda_tuned_topk_v1

Frozen Figure 4a clean-room mainline.

## Main 20-seed result

- DU-Net standard RMSE: 1.088861
- Linear UCB concat standard RMSE: 1.106389
- Gap, Linear UCB minus DU-Net: 0.017528
- Visual paper-gap recovery against 0.026 rough reference: 67.4%
- Relative comparator improvement: 1.58%
- Standard RMSE win rate: 19/20
- PDF-literal RMSE win rate: 19/20
- Reward win rate: 0/20
- DU-Net cumulative reward: 1250.56
- Linear UCB cumulative reward: 1393.68

## Per-cycle check

- Positive RMSE gap cycles: 21/21 (100.0%)
- Early-cycle mean gap, cycles 0-4: 0.001642
- Final-cycle gap: 0.017528

## Tuning signature

```
 adaptive_lambda_entropy  sigma  count  mean_cycle  mean_standard_mask_rmse  mean_cumulative_reward  mean_selected_reward_percentile  mean_selected_entropy_percentile
                   100.0    0.1     38   11.315789                 1.089610              680.224640                         0.359605                          0.999362
                   300.0    0.1     38   11.973684                 1.096113              755.740068                         0.376945                          0.999361
                    30.0    0.1     28   12.857143                 1.098200              830.286124                         0.422408                          0.999360
                     1.0    0.1     20    2.000000                 1.091748              122.531829                         0.539757                          0.999308
                   300.0    0.4     20    8.250000                 1.099681              559.074226                         0.502052                          0.999364
                     0.1    0.1     16    7.812500                 1.100907              536.244130                         0.992149                          0.594258
                    30.0    0.2     16   14.062500                 1.096375              904.516605                         0.484684                          0.999359
                   100.0    1.6     16    7.812500                 1.090260              500.929820                         0.454633                          0.999365
                     0.0    0.1     15    5.333333                 1.096947              358.793754                         0.999366                          0.531434
                     0.0    0.2     15    7.000000                 1.096629              450.231366                         0.999365                          0.549075
```

## Trade-off signature

```
           variant  standard_mask_rmse  cumulative_reward  standard_mask_rmse_gap_vs_linucb  cumulative_reward_gap_vs_linucb
     linucb_concat            1.098598        1386.497556                          0.000000                         0.000000
  schur_lambda_100            1.081490        1231.872962                          0.017108                      -154.624594
  schur_lambda_300            1.082090        1234.688280                          0.016508                      -151.809276
   schur_lambda_30            1.082174        1231.152680                          0.016424                      -155.344876
    schur_lambda_3            1.082312        1238.213976                          0.016286                      -148.283580
   schur_lambda_10            1.082812        1231.724527                          0.015786                      -154.773029
    schur_lambda_1            1.083860        1253.729828                          0.014738                      -132.767728
  schur_lambda_0p1            1.116360        1533.788093                         -0.017762                       147.290537
    schur_lambda_0            1.121477        1546.114737                         -0.022879                       159.617181
schur_entropy_only            1.081620        1233.949492                          0.016978                      -152.548064
```

## Artifact index

- Metrics: `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_metrics.csv`
- Final summary: `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_final_summary.csv`
- `configs\paper_faithful_movielens_schur_loglambda_tuned_topk_v1.json`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_metrics.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_final_summary.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_per_seed_gaps.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_per_cycle_rmse_curve.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_lambda_sigma_logs.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_lambda_sigma_summary.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_fixed_lambda_reward_rmse_tradeoff.csv`
- `outputs\logs\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_pytest_core.txt`
- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_standard_rmse_curve.png`
- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_topk_v1_reward_rmse_tradeoff_fixed_lambda_sweep.png`
